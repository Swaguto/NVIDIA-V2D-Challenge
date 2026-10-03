#!/usr/bin/env python3
"""Render a full-frame (no zoom) CAD tracking overlay for one object.

Panel-free: the raw rectified frame with the CAD silhouette outlined, so the object
stays in context. Configured to avoid the earlier 258-byte failure by taking the
writer dimensions from the actual frame instead of a hard-coded panel size.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/home/ubuntu")
from stereo_depth import load_rig  # noqa: E402

D = Path("/home/ubuntu/v2d_track3/data/hf/track_3/public")


def mesh_path(name):
    d = D / "mesh" / name
    for cand in (name, f"{name}_visual", f"{name}_collision"):
        p = d / f"{cand}.glb"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def silhouette(V, q, t, rig, shape):
    R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    X = (R @ (V - V.mean(0)).T).T + np.asarray(t)
    z = X[:, 2]
    ok = z > 1e-3
    u = (rig.P1[0, 0] * X[:, 0] / z + rig.P1[0, 2]).astype(np.int32)
    v = (rig.P1[1, 1] * X[:, 1] / z + rig.P1[1, 2]).astype(np.int32)
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    sel = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    m[v[sel], u[sel]] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--object", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=float, default=12.0)
    a = ap.parse_args()

    rig = load_rig()
    ep = f"e{a.episode:03d}"
    base = a.predictions / ep / "ego_object_poses" / ep / a.object / \
        "object_to_cam" / "ego_cam_c"
    files = sorted(base.glob("*.json"))
    if not files:
        raise SystemExit(f"no predictions in {base}")
    V = np.asarray(trimesh.load(str(mesh_path(a.object)),
                                force="mesh", process=False).vertices, np.float64)
    if len(V) > 120000:
        V = V[np.random.default_rng(0).choice(len(V), 120000, replace=False)]

    vroot = Path("/home/ubuntu/v2d_track3/data/hf/track_3/public/videos/chunk-000")
    vl = vroot / f"observation.images.{rig.left}" / f"episode_{a.episode:06d}.mp4"
    cap = cv2.VideoCapture(str(vl))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    shape = None
    writer = None
    written = 0
    while written < a.frames:
        ok, fr = cap.read()
        if not ok:
            break
        idx = a.start + written * a.stride
        f = base / f"{idx:06d}.json"
        if shape is None:
            shape = fr.shape[:2]
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     a.fps, (shape[1], shape[0]))
            if not writer.isOpened():
                raise SystemExit("VideoWriter failed to open")
        if not f.exists():
            written += 1
            continue
        r = json.loads(f.read_text())
        sil = silhouette(V, np.array(r["rotation"]), r["translation"], rig, shape)
        out = fr.copy()
        tint = out.copy()
        tint[sil > 0] = (255, 0, 255)
        out = np.where(sil[:, :, None] > 0,
                       (0.75 * out + 0.25 * tint).astype(np.uint8), out)
        cnts = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(out, cnts, -1, (255, 0, 255), 3)
        cv2.putText(out, f"{a.object}  f{idx}  z={r['translation'][2]:.2f}m",
                    (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        writer.write(out)
        written += 1
    writer.release()
    import os
    print(f"wrote {written} frames -> {a.out} "
          f"({os.path.getsize(a.out)} bytes)")


if __name__ == "__main__":
    main()