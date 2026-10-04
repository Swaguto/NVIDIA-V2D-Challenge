#!/usr/bin/env python3
"""Track 3 pose estimation: stereo depth + open-vocab masks + CAD registration.

Per frame:
  1. rectify the ego stereo pair (ego_cam_c left / ego_cam_b right) and take metric depth
  2. Grounding DINO finds the episode's known objects from name-only prompts
  3. SAM2 turns those boxes into masks
  4. mask & valid depth -> an object point cloud in the rectified left camera frame
  5. register the public CAD by trimmed similarity ICP

Two things matter for both accuracy and runtime:

* Scale is fixed to 1. Depth is metric, and the CAD is at true physical scale -- measured:
  the pot lid fits identically at fixed scale 1.0 and at a free scale of 0.955, so there is
  nothing to gain from estimating scale and a real risk of it drifting. The official frame-0
  alignment is SE(3), not Sim(3), so scale error would be scored.

* ICP runs on the GPU and warm-starts from the previous frame's pose. Per-frame ICP from a
  PCA multistart is both slow and temporally incoherent, and the metrics want a smooth
  trajectory. Frame 0 (and any frame whose warm start converges poorly) falls back to the
  full PCA multistart.

Output is the per-frame JSON layout the official scorer reads:
  <out>/eNNN/ego_object_poses/eNNN/<object>/object_to_cam/ego_cam_c/TTTTTT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from trimesh import load as trimesh_load

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("V2D_REPO", str(REPO_ROOT / "scripts/eval")))
sys.path.insert(0, os.environ.get("V2D_RUNTIME", str(REPO_ROOT / "reconstruction/modules/v2d_stereo_depth/lib")))
from reference_loader import load_reference                        # noqa: E402
from stereo_depth import CALIB_PATH, backproject_full, load_rig, stereo_depth  # noqa: E402

GDINO = "IDEA-Research/grounding-dino-base"
SAM2 = "facebook/sam2.1-hiera-large"
_ROOT = Path(os.environ.get("V2D_PUBLIC", "/home/ubuntu/v2d_track3/data/hf/track_3/public"))
_VID = _ROOT / "videos/chunk-000"
_MESH = _ROOT / "mesh"
DEV = "cuda"


# ------------------------------------------------------------------ GPU ICP core
def _kabsch(src: torch.Tensor, dst: torch.Tensor, allow_scale: bool = False):
    """Rigid (or similarity) transform mapping src onto dst. Returns (s, R, t)."""
    cs = src.mean(0)
    cd = dst.mean(0)
    X = src - cs
    Y = dst - cd
    U, S, Vh = torch.linalg.svd(X.T @ Y)
    d = torch.sign(torch.linalg.det(Vh.T @ U.T))
    D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    R = Vh.T @ D @ U.T
    s = torch.ones((), dtype=src.dtype, device=src.device)
    if allow_scale:
        var = (X ** 2).sum()
        s = torch.where(var > 1e-16, (S[0] + S[1] + d * S[2]) / var.clamp_min(1e-16),
                        torch.ones_like(s))
    t = cd - s * (R @ cs)
    return s, R, t


def _trimmed_cost(moved, dst, trim):
    """Symmetric trimmed distance (cloud->mesh and mesh->cloud).

    A one-sided residual is minimised by collapse: shrinking every CAD point onto the cloud
    centroid drives it toward zero, which is how an earlier version "solved" registration
    with scale 0. The backward term is what makes the objective meaningful.
    """
    d_f = torch.cdist(moved, dst).min(dim=1).values
    d_b = torch.cdist(dst, moved).min(dim=1).values
    kf = min(d_f.numel(), max(1, int(trim * d_f.numel())))
    kb = min(d_b.numel(), max(1, int(trim * d_b.numel())))
    f = torch.topk(d_f, kf, largest=False).values.mean()
    b = torch.topk(d_b, kb, largest=False).values.mean()
    return f + b, f, b


def _icp(cad: torch.Tensor, dst: torch.Tensor, R0: torch.Tensor, t0: torch.Tensor,
         trim=0.6, iters=40, allow_scale=False, s0=1.0):
    s = torch.as_tensor(s0 if allow_scale else 1.0, dtype=cad.dtype, device=cad.device)
    R = R0.clone()
    t = t0.clone()
    best = None
    for _ in range(iters):
        moved = s * (cad @ R.T) + t
        cur, f, b = _trimmed_cost(moved, dst, trim)
        if best is None or cur < best[0]:
            best = (cur.item(), s.item(), R.clone(), t.clone())
        d_f, idx = torch.cdist(moved, dst).min(dim=1)
        k = min(d_f.numel(), max(16, int(trim * d_f.numel())))
        sel = torch.topk(d_f, k, largest=False).indices
        thr = torch.quantile(d_f[sel], 0.8)
        sel = sel[d_f[sel] <= thr]
        if sel.numel() < 16:
            break
        ns, nR, nt = _kabsch(moved[sel], dst[idx[sel]], allow_scale)
        new_s, new_R, new_t = ns * s, nR @ R, ns * (nR @ t) + nt
        delta = (abs(float(torch.log(ns.clamp_min(1e-9)))) + float((new_t - t).norm())
                 + float((new_R - R).norm()) * float(cad.std()))
        s, R, t = new_s, new_R, new_t
        if delta < 1e-7:
            break
    moved = s * (cad @ R.T) + t
    cur, f, b = _trimmed_cost(moved, dst, trim)
    if best is not None and best[0] < cur.item():
        _, s, R, t = best[0], best[1], best[2], best[3]
        cur = _trimmed_cost(s * (cad @ R.T) + t, dst, trim)[0]
    return float(s), R, t, float(cur)


def _pca_basis(pts: torch.Tensor) -> torch.Tensor:
    c = pts - pts.mean(0)
    _, _, Vh = torch.linalg.svd(c, full_matrices=False)
    V = Vh.T
    if torch.linalg.det(V) < 0:
        V = V.clone()
        V[:, 2] *= -1
    return V


def _init_rotations(src, dst):
    """PCA initialisations: signed permutations of the target's principal axes."""
    Vs, Vd = _pca_basis(src), _pca_basis(dst)
    out = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    S = torch.diag(torch.tensor([float(sx), float(sy), float(sz)],
                                                dtype=src.dtype, device=src.device))
                    P = torch.zeros((3, 3), dtype=src.dtype, device=src.device)
                    for i, p in enumerate(perm):
                        P[i, p] = 1.0
                    R = Vd @ S @ P @ Vs.T
                    if torch.linalg.det(R) > 0:
                        out.append(R)
    out.append(torch.eye(3, dtype=src.dtype, device=src.device))
    return out


def register(cad: torch.Tensor, dst: torch.Tensor, R0=None, t0=None,
             allow_scale=False, trim=0.6, coarse=900, refine=4000, n_seed=3, s0=1.0):
    """Return camera-space (scale, R, t, residual) in the original CAD frame.

    Warm starts use the same uncentered camera-space convention as the result.
    """
    if any(x.ndim != 2 or x.shape[1] != 3 or len(x) < 16 or not torch.isfinite(x).all()
           for x in (cad, dst)):
        raise ValueError("registration requires finite (N,3) clouds with at least 16 points")
    if not 0 < trim <= 1 or n_seed < 1 or min(coarse, refine) < 16 or s0 <= 0:
        raise ValueError("invalid registration trim, seed count, sample size, or scale")
    cad_center, cloud_center = cad.mean(0), dst.mean(0)
    cad = cad - cad_center
    dst = dst - cloud_center
    z = torch.zeros(3, dtype=cad.dtype, device=cad.device)

    def restore(result):
        s, R, t, residual = result
        return s, R, cloud_center + t - s * (R @ cad_center), residual

    if R0 is not None:
        scale0 = s0 if allow_scale else 1.0
        centered_t0 = scale0 * (R0 @ cad_center) + t0 - cloud_center if t0 is not None else z
        best = _icp(cad, dst, R0, centered_t0, trim=trim,
                    iters=40, allow_scale=allow_scale, s0=scale0)
        # a warm start that converged badly is not trustworthy
        if best[3] < 0.05:
            return restore(best)

    g = torch.Generator(device="cpu").manual_seed(0)

    def sub(x, n):
        if x.shape[0] <= n:
            return x
        idx = torch.randperm(x.shape[0], generator=g)[:n].to(x.device)
        return x[idx]

    cad_c, dst_c = sub(cad, coarse), sub(dst, coarse)
    scored = []
    for R in _init_rotations(cad_c, dst_c):
        s, Rr, t, res = _icp(cad_c, dst_c, R, z, trim=trim, iters=10,
                             allow_scale=allow_scale)
        scored.append((res, s, Rr, t))
    scored.sort(key=lambda x: x[0])
    cad_f = sub(cad, refine)
    best = None
    for _, scale_seed, Rr, t in scored[:n_seed]:
        s, R2, t2, res = _icp(cad_f, dst, Rr, t, trim=trim, iters=50,
                              allow_scale=allow_scale, s0=scale_seed)
        if best is None or res < best[3]:
            best = (s, R2, t2, res)
    return restore(best)


# -------------------------------------------------------------- cloud filtering
def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]


def trim_cloud(pts: np.ndarray, k: float | None = None, iters: int | None = None,
               method: str = "local_density") -> np.ndarray:
    """Remove isolated depth samples without erasing connected handles/rims.

    Distance from the object's center cannot distinguish a valid elongated
    feature from noise. Local neighbour spacing is rigid-transform invariant.
    The old radial method is retained only for frozen-baseline comparisons.
    """
    if method not in ("local_density", "radial"):
        raise ValueError("unknown cloud filter")
    if k is None:
        k = 3.5 if method == "local_density" else 2.5
    if iters is None:
        iters = 1 if method == "local_density" else 3
    p = np.asarray(pts)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("cloud filter requires finite (N,3) points")
    if not np.isfinite(k) or k <= 0 or not isinstance(iters, int) or iters < 1:
        raise ValueError("invalid cloud filter threshold or iteration count")
    for _ in range(iters):
        if len(p) < 80:
            break
        if method == "local_density":
            from scipy.spatial import cKDTree

            # Include self then remove it: eleven nearby samples measure density.
            d = cKDTree(p).query(p, k=12)[0][:, 1:].mean(axis=1)
            s = max(1.4826 * np.median(np.abs(d - np.median(d))), 1e-9)
            # Boundaries have fewer neighbours. On uniform sampling MAD can be
            # nearly zero, so retain spacing up to twice the interior median.
            threshold = max(np.median(d) + k * s, 2 * np.median(d))
        else:
            d = np.linalg.norm(p - np.median(p, 0), axis=1)
            s = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-9
            threshold = np.median(d) + k * s
        keep = d <= threshold
        if keep.sum() < 80:
            break
        p = p[keep]
    return p


# ------------------------------------------------------------------ perception
class Segmenter:
    def __init__(self, names, device=DEV):
        from transformers import (AutoProcessor, GroundingDinoForObjectDetection,
                                  Sam2Model, Sam2Processor)
        self.names = list(names)
        self.device = device
        self.prompt = ". ".join(n.replace("_", " ") for n in self.names) + "."
        self.gp = AutoProcessor.from_pretrained(GDINO)
        self.gm = GroundingDinoForObjectDetection.from_pretrained(GDINO).to(device).eval()
        self.sp = Sam2Processor.from_pretrained(SAM2)
        self.sm = Sam2Model.from_pretrained(SAM2).to(device).eval()
        self.phrase2idx = {n.replace("_", " "): i for i, n in enumerate(self.names)}
        self._order = sorted(self.phrase2idx, key=len, reverse=True)

    def _assign(self, labels):
        out = []
        for lab in labels:
            lab = lab.strip().lower().replace("_", " ")
            hit = None
            for ph in self._order:          # longest phrase first: "white pot lid" > "white pot"
                if ph in lab:
                    hit = self.phrase2idx[ph]
                    break
            # An unrelated detection is not evidence for the only requested object.
            # In single-object episodes the old fallback could turn a table into a pan.
            out.append(hit)
        return out

    @torch.inference_mode()
    def __call__(self, rgb, box_threshold=0.30):
        self.last_candidates = []
        inp = self.gp(images=rgb, text=self.prompt, return_tensors="pt").to(self.device)
        det = self.gp.post_process_grounded_object_detection(
            self.gm(**inp), inp.input_ids, threshold=box_threshold, text_threshold=0.25,
            target_sizes=[rgb.shape[:2]])[0]
        masks = np.zeros((len(self.names),) + rgb.shape[:2], bool)
        if len(det["boxes"]) == 0:
            return masks, []
        boxes = [[[float(v) for v in b] for b in det["boxes"].tolist()]]
        sinp = self.sp(images=rgb, input_boxes=boxes, return_tensors="pt").to(self.device)
        sout = self.sm(**sinp, multimask_output=False)
        raw = sout.pred_masks if hasattr(sout, "pred_masks") else sout.output_masks
        kw = {}
        if sinp.get("reshaped_input_sizes") is not None:
            kw["reshaped_input_sizes"] = sinp["reshaped_input_sizes"]
        m = np.asarray(self.sp.post_process_masks(
            raw, sinp["original_sizes"], binarize=True, **kw)[0].cpu())[:, 0]
        info = []
        for k, (lab, sc, bi) in enumerate(zip(det["text_labels"],
                                             det["scores"].tolist(),
                                             self._assign(det["text_labels"]))):
            self.last_candidates.append({
                "object": self.names[bi] if bi is not None else None,
                "label": lab, "score": float(sc), "box_xyxy": boxes[0][k],
                "mask_pixels": int(m[k].sum()),
            })
            if bi is None:
                info.append((None, lab, float(sc)))
                continue
            masks[bi] |= m[k]
            info.append((self.names[bi], lab, float(sc)))
        return masks, info


class PreparedMasks:
    """Consume frozen, hash-pinned observations without rerunning perception."""

    def __init__(self, path, episode, names, rig, require_no_gt=False):
        self.path = Path(path).resolve()
        self.raw = self.path.read_bytes()
        self.manifest = json.loads(self.raw)
        record = self.manifest
        if (record.get("schema_version") != 1 or record.get("episode") != episode
                or record.get("objects") != names or record.get("camera_frame") != rig.left
                or record.get("coordinate_frame") != "rectified_camera"
                or record.get("resolution") != list(rig.size)):
            raise ValueError("prepared mask identity/axes/resolution do not match this run")
        self.gt_use = record.get("gt_use")
        if self.gt_use not in ("none", "gt_assisted", "unknown") or require_no_gt and self.gt_use != "none":
            raise ValueError("prepared masks require compatible, explicit GT-use provenance")
        calibration_sha = hashlib.sha256(Path(CALIB_PATH).read_bytes()).hexdigest()
        if record.get("calibration_sha256") != calibration_sha:
            raise ValueError("prepared mask calibration hash differs from this run")
        for key in ("R1", "R2", "P1", "P2", "Q"):
            given = np.asarray(record.get("rectification", {}).get(key), dtype=float)
            expected = np.asarray(getattr(rig, key))
            if given.shape != expected.shape or not np.allclose(given, expected, rtol=0, atol=1e-9):
                raise ValueError(f"prepared mask rectification differs: {key}")
        if not isinstance(record.get("producer"), dict) or not isinstance(record.get("frames"), dict):
            raise ValueError("prepared masks need producer and source-frame records")
        self.names, self.size = names, tuple(rig.size)

    def load(self, frame, image_shape):
        if tuple(image_shape[:2]) != self.size[::-1]:
            raise ValueError("prepared masks and rectified RGB have different image sizes")
        records = self.manifest["frames"].get(str(frame), {})
        if not isinstance(records, dict) or set(records) - set(self.names):
            raise ValueError("prepared mask source frame has unknown object identities")
        masks = np.zeros((len(self.names),) + image_shape[:2], dtype=bool)
        info = []
        for i, name in enumerate(self.names):
            item = records.get(name)
            if item is None:
                info.append({"object": name, "status": "missing_prepared_mask"})
                continue
            path = (self.path.parent / item["path"]).resolve()
            if not path.is_relative_to(self.path.parent):
                raise ValueError("prepared mask path leaves its manifest root")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError(f"prepared mask hash differs: {frame}/{name}")
            if not content.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"prepared mask is not a PNG: {frame}/{name}")
            mask = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if (mask is None or mask.shape != image_shape[:2] or mask.dtype != np.uint8
                    or not np.isin(mask, [0, 255]).all()):
                raise ValueError(f"prepared mask must be a full-resolution binary PNG: {frame}/{name}")
            masks[i] = mask > 0
            info.append({"object": name, "status": "prepared", "sha256": item["sha256"]})
        return masks, info


def load_cad(name: str, n: int = 6000, seed: int = 0) -> np.ndarray:
    d = _MESH / name
    for cand in ("_visual", ""):
        for ext in (".glb", ".ply", ".obj", ".stl"):
            p = d / f"{name}{cand}{ext}"
            if p.exists():
                m = trimesh_load(str(p), force="mesh")
                if m.faces is not None and len(m.faces) and len(m.vertices) >= 16:
                    pts = np.asarray(m.sample(n, seed=seed) if hasattr(m, "sample")
                                     else m.vertices, np.float64)
                    if np.isfinite(pts).all():
                        return pts
    raise FileNotFoundError(f"no usable CAD for {name} under {_MESH}")


def read_frame(path: Path, idx: int):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx} of {path}")
    return img


def _mat_to_wxyz(R):
    """Rotation matrix -> quaternion in (w, x, y, z) order, Shepperd's method.

    The closed form w = 0.25*s with s = 2*sqrt(1-t) is only valid in the small-angle
    branch: it silently wrote identity for ~17% of rotations (trace ~= -1) and wrong
    angles for the rest, corrupting every predicted pose. Branch on the largest diagonal
    term instead, which is numerically stable over the full range.
    """
    R = np.asarray(R, np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--allow-scale", action="store_true")
    ap.add_argument("--no-warmstart", action="store_true")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--log", type=int, default=25)
    ap.add_argument("--max-residual", type=float, default=0.05,
                    help="reject registration above this symmetric trimmed distance in metres")
    ap.add_argument("--no-gt", action="store_true", help="never read reference parquet")
    ap.add_argument("--objects", default="", help="comma-separated membership, required with --no-gt")
    ap.add_argument("--frame-times", type=Path, help="JSON mapping video frame IDs to capture times")
    ap.add_argument("--mask-manifest", type=Path,
                    help="frozen rectified mask manifest; bypass DINO/SAM and preserve producer hashes")
    ap.add_argument("--seed", type=int, default=0, help="deterministic CAD surface sampling seed")
    ap.add_argument("--cloud-filter", choices=("local_density", "radial"), default="local_density",
                    help="local density preserves elongated features; radial replays the old baseline")
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        ap.error("output must be a fresh directory")
    if a.stride < 1 or a.frames < 0:
        ap.error("stride must be positive and frames nonnegative")

    ref = None if a.no_gt else load_reference(a.episode, _ROOT)
    names = [n.strip() for n in a.objects.split(",") if n.strip()]
    if not names and ref is not None:
        names = list(ref.object_names)
    if not names or len(names) != len(set(names)):
        ap.error("provide distinct --objects with --no-gt")
    rig = load_rig()
    prepared = PreparedMasks(a.mask_manifest, a.episode, names, rig, a.no_gt) if a.mask_manifest else None
    gt_use = (prepared.gt_use if prepared and prepared.gt_use != "none"
              else "none" if a.no_gt else "metadata_only")
    cads = {n: torch.tensor(load_cad(n, 6000, a.seed), dtype=torch.float32, device=DEV)
            for n in names}
    vlp = _VID / f"observation.images.{rig.left}" / f"episode_{a.episode:06d}.mp4"
    vrp = _VID / f"observation.images.{rig.right}" / f"episode_{a.episode:06d}.mp4"
    if ref is not None:
        frames = np.asarray(ref.frame_index, dtype=int)
        times = dict(zip(frames, ref.capture_time))
    else:
        counts = []
        for path in (vlp, vrp):
            capture = cv2.VideoCapture(str(path))
            counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            capture.release()
        if counts[0] <= 0 or counts[0] != counts[1]:
            ap.error("stereo videos require matching, nonempty frame counts")
        frames = np.arange(counts[0])
        times = {}
    if a.frame_times:
        times = {int(k): float(v) for k, v in json.loads(a.frame_times.read_text()).items()}
    if a.frames:
        frames = frames[:a.frames]
    steps = len(frames)
    seg = None if prepared else Segmenter(names)
    print(f"episode {a.episode}: {names} steps={steps} scale={'free' if a.allow_scale else 'fixed 1.0'}", flush=True)
    prev = {n: None for n in names}
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    rectification = {"episode": a.episode, "camera_frame": rig.left,
        "coordinate_frame": "rectified_camera", "baseline_m": rig.baseline_m,
        **{key: np.asarray(getattr(rig, key)).tolist() for key in ("R1", "R2", "P1", "P2", "Q")}}
    rectification_path = out / "rectification.json"
    rectification_path.write_text(json.dumps(rectification, indent=2))
    rectification_hash = hashlib.sha256(rectification_path.read_bytes()).hexdigest()
    report = {"episode": a.episode, "objects": names, "coordinate_frame": "rectified_camera",
        "camera_frame": rig.left, "gt_use": gt_use,
        "source_sha256_lf": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": DEV,
                    "device_name": torch.cuda.get_device_name() if DEV == "cuda" else "cpu"},
        "models": None if prepared else {"detector": GDINO, "segmenter": SAM2},
        "observation_source": ({"kind": "prepared_masks", "manifest_path": str(prepared.path),
            "manifest_sha256": hashlib.sha256(prepared.raw).hexdigest(),
            "gt_use": prepared.gt_use, "producer": prepared.manifest["producer"]} if prepared else {"kind": "dino_sam2"}),
        "rectification_sha256": rectification_hash,
        "params": {"stride": a.stride, "allow_scale": a.allow_scale,
                   "max_residual_m": a.max_residual, "box_threshold": a.box_threshold,
                   "cad_samples": 6000, "cad_seed": a.seed,
                   "cloud_filter": {"method": a.cloud_filter,
                       "mad_threshold": 3.5 if a.cloud_filter == "local_density" else 2.5,
                       "iterations": 1 if a.cloud_filter == "local_density" else 3,
                       "minimum_spacing_ratio": 2.0 if a.cloud_filter == "local_density" else None,
                       "neighbors": 11 if a.cloud_filter == "local_density" else None}},
        "frames": {}, "accepted_by_object": dict.fromkeys(names, 0), "input_errors": []}
    t0 = time.time()

    for t in frames[::a.stride]:
        t = int(t)
        try:
            il, ir = read_frame(vlp, t), read_frame(vrp, t)
        except RuntimeError as error:
            report["input_errors"].append({"frame_index": t, "error": str(error)})
            break
        pl, _ = rig.rectify_pair(il, ir)
        depth, valid = stereo_depth(rig, il, ir)
        try:
            masks, _info = (prepared.load(t, pl.shape) if prepared
                           else seg(cv2.cvtColor(pl, cv2.COLOR_BGR2RGB), a.box_threshold))
        except (ValueError, OSError) as error:
            report["input_errors"].append({"frame_index": t, "error": str(error)})
            break
        frame_report = {"timestamp": float(times[t]) if t in times else None,
                        "detections": _info,
                        "detection_candidates": getattr(seg, "last_candidates", []),
                        "objects": {}}
        report["frames"][str(t)] = frame_report
        pts = backproject_full(depth, valid, rig.P1)

        for bi, n in enumerate(names):
            prior = prev[n]
            # Every rejection path breaks the observed track. A later observation
            # must reinitialize rather than reuse a camera pose from before the gap.
            prev[n] = None
            m = masks[bi].ravel() & valid.ravel()
            detail = {"mask_pixels": int(masks[bi].sum()), "valid_depth_pixels": int(m.sum()),
                      "status": "rejected", "reason": "insufficient_masked_depth",
                      "warm_start_frame_index": None}
            frame_report["objects"][n] = detail
            d = out / f"e{a.episode:03d}" / "ego_object_poses" / f"e{a.episode:03d}" / n \
                / "object_to_cam" / rig.left
            d.mkdir(parents=True, exist_ok=True)
            if m.sum() < 300:
                continue
            downsampled = voxel_downsample(pts[m], 0.004)
            detail["downsampled_cloud_points"] = len(downsampled)
            cloud = trim_cloud(downsampled, method=a.cloud_filter)
            detail["filtered_cloud_points"] = len(cloud)
            if len(cloud) < 120:
                detail["reason"] = "insufficient_filtered_cloud"
                continue
            dst = torch.tensor(cloud, dtype=torch.float32, device=DEV)
            R0 = t0_ = None
            if not a.no_warmstart and prior is not None:
                Rp, tp, _, previous_scale, previous_frame = prior
                if t == previous_frame + a.stride:
                    R0, t0_ = Rp.clone(), tp.clone()
                    detail["warm_start_frame_index"] = previous_frame
            try:
                reg = register(cads[n], dst, R0=R0, t0=t0_, allow_scale=a.allow_scale,
                               s0=previous_scale if R0 is not None else 1.0)
            except Exception as error:                       # noqa: BLE001
                detail.update(reason="registration_error", error=f"{type(error).__name__}: {error}")
                print(f"    t={t} {n}: {type(error).__name__}: {error}", flush=True)
                continue
            if reg is None:
                detail["reason"] = "no_registration_result"
                continue
            s, R, tt, res = reg
            detail["residual_m"] = float(res) if np.isfinite(res) else None
            if not np.isfinite(res) or res > a.max_residual:
                detail["reason"] = "residual_gate"
                continue
            detail.update(status="observed", reason=None)
            report["accepted_by_object"][n] += 1
            prev[n] = (R, tt, res, s, t)   # camera-space warm start, including scale/source ID
            (d / f"{t:06d}.json").write_text(json.dumps({
                "translation": [float(x) for x in tt.cpu().numpy()],
                "rotation": [float(x) for x in _mat_to_wxyz(R.cpu().numpy())],
                "scale": float(s), "residual": float(res),
                "cloud_px": int(m.sum()),
                "episode": a.episode, "frame_index": t,
                "timestamp": float(times[t]) if t in times else None, "status": "observed",
                "coordinate_frame": "rectified_camera", "camera_frame": rig.left,
                "gt_use": gt_use, "quaternion_order": "wxyz",
                "rectification_sha256": rectification_hash,
                "observation_source": report["observation_source"],
                "mask_sha256": _info[bi]["sha256"] if prepared else None,
            }))
        if a.log and (t // a.stride) % a.log == 0:
            done = [prev[n][2] for n in names if prev[n] is not None]
            print(f"  t={t:>4}/{steps}  residual="
                  f"{[round(d, 4) for d in done]}  [{time.time() - t0:.0f}s]", flush=True)
    (out / "registration_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"done {a.episode} in {time.time() - t0:.0f}s; accepted={report['accepted_by_object']}; "
          f"report={out / 'registration_report.json'}", flush=True)
    if report["input_errors"] or not all(report["accepted_by_object"].values()):
        raise SystemExit("incomplete tracking: input read failed or an object has zero accepted poses; inspect registration_report.json")


if __name__ == "__main__":
    main()
