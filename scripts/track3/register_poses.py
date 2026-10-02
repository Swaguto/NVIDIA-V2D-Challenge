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
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from trimesh import load as trimesh_load

sys.path.insert(0, os.environ.get("V2D_REPO", "/home/ubuntu/repoeval"))
sys.path.insert(0, os.environ.get("V2D_RUNTIME", "/home/ubuntu"))
from reference_loader import load_reference                        # noqa: E402
from stereo_depth import backproject_full, load_rig, stereo_depth  # noqa: E402

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
    kf = max(16, int(trim * d_f.numel()))
    kb = max(16, int(trim * d_b.numel()))
    f = torch.topk(d_f, kf, largest=False).values.mean()
    b = torch.topk(d_b, kb, largest=False).values.mean()
    return f + b, f, b


def _icp(cad: torch.Tensor, dst: torch.Tensor, R0: torch.Tensor, t0: torch.Tensor,
         trim=0.6, iters=40, allow_scale=False):
    s = torch.ones((), dtype=cad.dtype, device=cad.device)
    R = R0.clone()
    t = t0.clone()
    best = None
    for _ in range(iters):
        moved = s * (cad @ R.T) + t
        cur, f, b = _trimmed_cost(moved, dst, trim)
        if best is None or cur < best[0]:
            best = (cur.item(), s.item(), R.clone(), t.clone())
        d_f, idx = torch.cdist(moved, dst).min(dim=1)
        k = max(16, int(trim * d_f.numel()))
        sel = torch.topk(d_f, k, largest=False).indices
        thr = torch.quantile(d_f[sel], 0.8)
        sel = sel[d_f[sel] <= thr]
        if sel.numel() < 16:
            break
        ns, nR, nt = _kabsch(moved[sel], dst[idx[sel]], allow_scale)
        delta = (abs(float(torch.log(ns.clamp_min(1e-9)))) + float((nt - t).norm())
                 + float((nR - R).norm()) * float(cad.std()))
        s, R, t = ns * s, nR @ R, nR @ t + nt
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
             allow_scale=False, trim=0.6, coarse=900, refine=4000, n_seed=3):
    """Similarity registration of CAD onto a cloud, coarse-to-fine, on GPU."""
    cad = cad - cad.mean(0)
    dst = dst - dst.mean(0)
    z = torch.zeros(3, dtype=cad.dtype, device=cad.device)

    if R0 is not None:
        best = _icp(cad, dst, R0, t0 if t0 is not None else z, trim=trim,
                    iters=40, allow_scale=allow_scale)
        # a warm start that converged badly is not trustworthy
        if best[3] < 0.05:
            return best

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
    for _, _, Rr, t in scored[:n_seed]:
        s, R2, t2, res = _icp(cad_f, dst, Rr, t, trim=trim, iters=50,
                              allow_scale=allow_scale)
        if best is None or res < best[3]:
            best = (s, R2, t2, res)
    return best


# -------------------------------------------------------------- cloud filtering
def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]


def trim_cloud(pts: np.ndarray, k: float = 2.5, iters: int = 3) -> np.ndarray:
    """Iterative statistical outlier removal."""
    p = pts
    for _ in range(iters):
        if len(p) < 80:
            break
        med = np.median(p, 0)
        d = np.linalg.norm(p - med, axis=1)
        s = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-9
        keep = d <= np.median(d) + k * s
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
            if hit is None:
                hit = 0 if len(self.names) == 1 else None
            out.append(hit)
        return out

    @torch.inference_mode()
    def __call__(self, rgb, box_threshold=0.30):
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
            if bi is None:
                continue
            masks[bi] |= m[k]
            info.append((self.names[bi], lab, float(sc)))
        return masks, info


def load_cad(name: str, n: int = 6000) -> np.ndarray:
    d = _MESH / name
    for cand in ("_visual", ""):
        for ext in (".glb", ".ply", ".obj", ".stl"):
            p = d / f"{name}{cand}{ext}"
            if p.exists():
                m = trimesh_load(str(p), force="mesh")
                if m.faces is not None and len(m.faces) and len(m.vertices) >= 16:
                    pts = np.asarray(m.sample(n) if hasattr(m, "sample")
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
    a = ap.parse_args()

    ref = load_reference(a.episode, _ROOT)
    names = list(ref.object_names)
    steps = a.frames or ref.steps
    rig = load_rig()
    cads = {n: torch.tensor(load_cad(n), dtype=torch.float32, device=DEV)
            for n in names}
    print(f"episode {a.episode}: {names}  steps={steps}  scale={'free' if a.allow_scale else 'fixed 1.0'}",
          flush=True)

    seg = Segmenter(names)
    vlp = _VID / f"observation.images.{rig.left}" / f"episode_{a.episode:06d}.mp4"
    vrp = _VID / f"observation.images.{rig.right}" / f"episode_{a.episode:06d}.mp4"
    prev = {n: None for n in names}
    out = a.out
    t0 = time.time()

    for t in range(0, steps, a.stride):
        try:
            il, ir = read_frame(vlp, t), read_frame(vrp, t)
        except RuntimeError:
            break
        pl, _ = rig.rectify_pair(il, ir)
        depth, valid = stereo_depth(rig, il, ir)
        masks, _info = seg(cv2.cvtColor(pl, cv2.COLOR_BGR2RGB), a.box_threshold)
        pts = backproject_full(depth, valid, rig.P1)

        for bi, n in enumerate(names):
            m = masks[bi].ravel() & valid.ravel()
            d = out / f"e{a.episode:03d}" / "ego_object_poses" / f"e{a.episode:03d}" / n \
                / "object_to_cam" / "ego_cam_c"
            d.mkdir(parents=True, exist_ok=True)
            if m.sum() < 300:
                continue
            cloud = trim_cloud(voxel_downsample(pts[m], 0.004))
            if len(cloud) < 120:
                continue
            dst = torch.tensor(cloud, dtype=torch.float32, device=DEV)
            R0 = t0_ = None
            if not a.no_warmstart and prev[n] is not None:
                Rp, tp, _ = prev[n]
                R0, t0_ = Rp.clone(), tp.clone()
            try:
                reg = register(cads[n], dst, R0=R0, t0=t0_, allow_scale=a.allow_scale)
            except Exception as error:                       # noqa: BLE001
                print(f"    t={t} {n}: {type(error).__name__}: {error}", flush=True)
                continue
            if reg is None:
                continue
            s, R, tt, res = reg
            prev[n] = (R, tt, res)   # keep on device: warm start feeds CUDA matmuls
            (d / f"{t:06d}.json").write_text(json.dumps({
                "translation": [float(x) for x in tt.cpu().numpy()],
                "rotation": [float(x) for x in _mat_to_wxyz(R.cpu().numpy())],
                "scale": float(s), "residual": float(res),
                "cloud_px": int(m.sum()),
            }))
        if a.log and (t // a.stride) % a.log == 0:
            done = [prev[n][2] for n in names if prev[n] is not None]
            print(f"  t={t:>4}/{steps}  residual="
                  f"{[round(d, 4) for d in done]}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"done {a.episode} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()