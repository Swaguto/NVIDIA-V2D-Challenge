#!/usr/bin/env python3
"""Probe: does ego stereo depth from RGB-only actually work on the public episodes?

Checks, on one episode:
  - rectification sanity (images must not be rotated/broken; note ego rotation_deg=180)
  - per-frame valid-depth fraction and metric range
  - depth stability across frames (a good sign the match is real, not noise)
  - writes a visual montage

There is no camera->world extrinsic, so GT cannot be projected into the camera frame for a
direct check; plausibility plus frame-to-frame stability is the available evidence here.
The real validation is the end-to-end official metric score after registration.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/ubuntu")
from stereo_depth import backproject, load_rig, stereo_depth   # noqa: E402

VID = Path(os.environ.get("V2D_VIDEOS",
                    "/home/ubuntu/v2d_track3/data/hf/track_3/public/videos/chunk-000"))
OUT = Path(os.environ.get("V2D_OUT", "/home/ubuntu/depth_probe")); OUT.mkdir(exist_ok=True, parents=True)

rig = load_rig()
print(f"pair {rig.left} <-> {rig.right}  baseline={rig.baseline_m:.6f} m  size={rig.size}")
print(f"rectified fx={rig.fx_rect:.3f}  cx={rig.P1[0,2]:.2f} cy={rig.P1[1,2]:.2f}")
print(f"Q-derived baseline = {-rig.Q[3,0]/rig.Q[0,0]:.6f} m")

episode = int(sys.argv[1]) if len(sys.argv) > 1 else 0
n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 6

cap_l = cv2.VideoCapture(str(VID / f"observation.images.{rig.left}" / f"episode_{episode:06d}.mp4"))
cap_r = cv2.VideoCapture(str(VID / f"observation.images.{rig.right}" / f"episode_{episode:06d}.mp4"))
total = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"episode {episode}: {total} frames")

panels, stats = [], []
idxs = np.linspace(0, total - 1, n_frames).astype(int)
for want in idxs:
    while True:
        ok_l, il = cap_l.read(); ok_r, ir = cap_r.read()
        if not ok_l or not ok_r:
            break
        if cap_l.get(cv2.CAP_PROP_POS_FRAMES) - 1 == want:
            break
    if not ok_l:
        break
    depth, valid = stereo_depth(rig, il, ir)
    frac = valid.mean()
    med = float(np.median(depth[valid])) if valid.any() else float("nan")
    pts, uv = backproject(depth, valid, rig.P1)
    # object-sized depth band: median distance of points in the middle 60% of the frame
    cy, cx = rig.size[1] // 2, rig.size[0] // 2
    box = (np.abs(uv[:, 0] - cx) < 0.3 * rig.size[0]) & (np.abs(uv[:, 1] - cy) < 0.3 * rig.size[1])
    med_c = float(np.median(pts[box, 2])) if box.sum() > 50 else float("nan")
    stats.append((int(want), frac, med, med_c, int(valid.sum())))
    print(f"  frame {want:>4}: valid={frac:6.1%}  median_z={med:.3f} m  "
          f"centre_z={med_c:.3f} m  n={valid.sum()}")

    vis = np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = 0.25, 1.4
    n = np.clip((depth - lo) / (hi - lo), 0, 1)
    vis[..., 2] = (n * 255).astype(np.uint8)
    vis[..., 0] = ((1 - n) * 255).astype(np.uint8)
    vis[~valid] = (40, 40, 40)
    panels.append(np.hstack([il, ir, vis]))
cap_l.release(); cap_r.release()

if panels:
    montage = np.vstack([cv2.resize(p, (0, 0), fx=0.55, fy=0.55) for p in panels])
    cv2.imwrite(str(OUT / f"ep{episode:03d}_stereo.png"), montage)
    print(f"\nwrote {OUT / f'ep{episode:03d}_stereo.png'}  "
          f"(raw-left | raw-right | depth, blue=far red=near, grey=invalid)")
    z = np.array([s[2] for s in stats])
    print(f"median depth across sampled frames: {np.nanmean(z):.3f} m "
          f"(std {np.nanstd(z):.3f}) — a stable value means the stereo match is real")