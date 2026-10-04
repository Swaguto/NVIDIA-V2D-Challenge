#!/usr/bin/env python3
"""FoundationPose baselines for Track 3, as an alternative to the ICP pipeline.

Requires the separate ``fp_venv`` (torch 2.1 + nvdiffrast + the FoundationPose weights);
it is not importable from the main venv. Typical invocation:

    FP_CUDNN=0 ~/fp_venv/bin/python scripts/track3/foundationpose_baseline.py \
        --mode reanchor --mask-root /home/ubuntu/masks_wide --out /home/ubuntu/pred_fpreanchor

Three modes, all reading masks only -- GT is never touched:

``static``    register once on the first inlier mask, hold that pose for the episode.
``track``     register once, then ``track_one`` for every remaining frame. Pure tracking
              drifts and loses the object on some episodes (AUC 0.0920).
``reanchor``  re-register on every frame that has a fresh inlier mask, track in between.
              This bounds drift to the gap between usable masks (~5 frames with
              ``masks_wide``) while still following real motion, with no tuning constants.

Mask selection is deliberately conservative: a frame is an anchor only if its area is at
least 400 px and within [0.35, 3.0] x the object's median mask area. That filter is what
raised anchor counts from 1/8 and 1/4 frames (sparse every-25th-frame sampling) to >= 5
for all nine objects.

**Quaternion order.** ``scripts/eval/score_track3_official.py`` reads
``rotation`` as ``[w, x, y, z]`` (see ``_q2m_wxyz_to_xyzw``). Earlier drafts of this script
wrote ``[x, y, z, w]``, which silently scores a wrong pose; ``quat_wxyz`` below is the
correct conversion. Use ``validate_submission.py`` after writing.
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
import trimesh

_ROOT = Path(os.environ.get("V2D_PUBLIC", "/home/ubuntu/v2d_track3/data/hf/track_3/public"))
_FP = os.environ.get("FOUNDERATIONPOSE", "/home/ubuntu/FoundationPose")
# stereo_depth is a real module in the tree, not a VM-only file; render_track.py does the
# same. Several older scripts here still import it from /home/ubuntu, which only resolves
# on one machine.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "reconstruction" / "modules" / "v2d_stereo_depth" / "lib"))
sys.path.insert(0, _FP)
sys.path.insert(0, f"{_FP}/mycpp/build")

# The A6000 in this VM ships a driver newer than torch 2.1's cuDNN expects; without this
# every conv raises an illegal-memory-access. cuDNN off costs speed, not correctness.
if os.environ.get("FP_CUDNN", "0") != "1":
    torch.backends.cudnn.enabled = False

import nvdiffrast.torch as dr                                        # noqa: E402
from Utils import set_logging_format, set_seed                         # noqa: E402
from estimater import FoundationPose                                  # noqa: E402
from learning.training.predict_pose_refine import PoseRefinePredictor  # noqa: E402
from learning.training.predict_score import ScorePredictor             # noqa: E402

from stereo_depth import load_rig, stereo_depth                        # noqa: E402


def inlier_masks(mdir: Path, lo: float = 0.35, hi: float = 3.0, min_px: int = 400):
    """{frame: mask_path} for frames whose area is a plausible inlier -> safe anchors."""
    ps = sorted(mdir.glob("*.png"))
    if not ps:
        return {}
    areas = []
    for p in ps:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        areas.append(int((m > 0).sum()) if m is not None else 0)
    med = float(np.median(areas))
    return {int(p.stem): p for p, a in zip(ps, areas)
            if a >= min_px and lo * med <= a <= hi * med}


def resolve_mesh(obj: str) -> Path:
    for suffix in ("", "_visual", "_collision"):
        p = _ROOT / "mesh" / obj / f"{obj}{suffix}.glb"
        if p.exists():
            return p
    raise FileNotFoundError(obj)


def quat_wxyz(R):
    """Rotation matrix -> [w, x, y, z], the order the official scorer expects."""
    m = np.asarray(R, float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = .25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, .25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, .25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, .25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


class Seq:
    """Sequential frame reader with random access by episode index."""

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        self.i = -1

    def get(self, idx: int):
        while self.i < idx:
            self.ok, self.fr = self.cap.read()
            self.i += 1
            if not self.ok:
                raise RuntimeError(f"eof at {self.i}")
        return self.fr.copy()

    def close(self):
        self.cap.release()


class FrameReader:
    """Rectified RGB + metric depth for an episode, read sequentially with random access.

    Both stereo streams must be decoded in order (OpenCV cannot seek these mp4s cheaply),
    so frames are pulled forward and only the current pair is kept in memory.
    """

    def __init__(self, rig, left: Path, right: Path):
        self.rig = rig
        self.sl, self.sr = Seq(left), Seq(right)
        self.cache: dict[int, tuple] = {}

    def __call__(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        if t not in self.cache:
            il, ir = self.sl.get(t), self.sr.get(t)
            pl, _ = self.rig.rectify_pair(il, ir)
            depth, valid = stereo_depth(self.rig, il, ir)
            self.cache.clear()
            self.cache[t] = (pl[:, :, ::-1].copy(),
                             np.where(valid, depth, 0.0).astype(np.float32))
        return self.cache[t]

    def close(self):
        self.sl.close()
        self.sr.close()


def run_object(est, K, frames, anchors, steps, mode):
    """-> ({frame: 4x4 pose}, n_register, n_track) for one object."""
    traj: dict[int, np.ndarray] = {}
    nreg = ntrk = 0
    # static registers once, on the first frame that actually has an inlier mask
    static_at = min(anchors) if anchors else None
    for t in range(steps):
        try:
            rgb, depth = frames(t)
        except RuntimeError:
            break
        if t in anchors or (mode == "static" and t == static_at):
            pose = est.register(K=K, rgb=rgb, depth=depth,
                                ob_mask=cv2.imread(str(anchors[t]), cv2.IMREAD_GRAYSCALE) > 0,
                                iteration=ITER)
            nreg += 1
            traj[t] = np.asarray(pose, float)
            if mode == "static":
                break                      # hold this one pose for the whole episode
        elif traj:
            pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=ITER)
            ntrk += 1
            traj[t] = np.asarray(pose, float)
        else:
            continue
    return traj, nreg, ntrk


ITER = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("static", "track", "reanchor"), default="reanchor")
    ap.add_argument("--episodes", default="0,11,18,23,39")
    ap.add_argument("--mask-root", required=True, type=Path,
                    help="dir with <eNNN>/<object>/*.png masks, every Nth frame")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    set_logging_format()
    set_seed(0)
    rig = load_rig()
    P = np.asarray(rig.P1, float)
    K = P[:3, :3] if P.shape == (3, 4) else P
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    from trimesh.visual import ColorVisuals

    # steps per episode, read from the videos we are about to open
    eps = [int(e) for e in a.episodes.split(",") if e.strip()]
    print(f"mode={a.mode} masks={a.mask_root} out={a.out}\n", flush=True)

    for ep in eps:
        vlp = _ROOT / "videos/chunk-000" / f"observation.images.{rig.left}" / f"episode_{ep:06d}.mp4"
        vrp = _ROOT / "videos/chunk-000" / f"observation.images.{rig.right}" / f"episode_{ep:06d}.mp4"
        obj_dirs = sorted(p for p in (a.mask_root / f"e{ep:03d}").iterdir() if p.is_dir())
        frames = FrameReader(rig, vlp, vrp)
        steps = int(cv2.VideoCapture(str(vlp)).get(cv2.CAP_PROP_FRAME_COUNT))

        for mdir in obj_dirs:
            obj = mdir.name
            anchors = inlier_masks(mdir)
            od = a.out / f"e{ep:03d}/ego_object_poses/e{ep:03d}" / obj / "object_to_cam/ego_cam_c"
            od.mkdir(parents=True, exist_ok=True)
            if not anchors:
                print(f"ep{ep:03d} {obj:22} NO ANCHORS -> skipped", flush=True)
                continue
            mesh = trimesh.load(str(resolve_mesh(obj)), force="mesh", process=False)
            mesh.visual = ColorVisuals(
                mesh=mesh,
                vertex_colors=np.tile(np.array([128, 128, 128], np.uint8), (len(mesh.vertices), 1)))
            est = FoundationPose(model_pts=np.asarray(mesh.vertices),
                                 model_normals=mesh.vertex_normals, mesh=mesh,
                                 scorer=scorer, refiner=refiner, debug=0,
                                 debug_dir="/tmp/fpdbg", glctx=glctx)
            t0 = time.time()
            traj, nreg, ntrk = run_object(est, K, frames, anchors, steps, a.mode)
            if not traj:
                print(f"ep{ep:03d} {obj:22} no frames", flush=True)
                continue
            first = min(traj)
            for t in range(0, first):
                traj[t] = traj[first]
            for t in range(steps):
                p = traj.get(t, traj[max(k for k in traj if k <= t)])
                (od / f"{t:06d}.json").write_text(json.dumps(
                    {"rotation": quat_wxyz(p[:3, :3]).tolist(),
                     "translation": [float(v) for v in p[:3, 3]]}))
            ts = np.array([traj[t][:3, 3] for t in sorted(traj)])
            step = np.linalg.norm(np.diff(ts, axis=0), axis=1) * 100
            print(f"ep{ep:03d} {obj:22} reg={nreg:3d} trk={ntrk:3d} "
                  f"range_cm={np.round((ts.max(0) - ts.min(0)) * 100, 1)} "
                  f"step_med={np.median(step):.2f} step_max={step.max():.2f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        frames.close()
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
