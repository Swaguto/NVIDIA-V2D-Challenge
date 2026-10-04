#!/usr/bin/env python3
"""Render a full-frame (no zoom) CAD tracking overlay for every object in a prediction.

Two bugs made earlier videos untrustworthy as diagnostics, and both are fixed here:

1. Frame/pose desync. The old loop read video frames sequentially (`cap.read()`) while
   advancing the *pose* index by `--stride`. On a 353-frame episode with --stride 5 the
   last output frame drew the pose from episode frame 350 onto the pixels from frame
   118 -- 232 frames apart. Frames are now consumed strictly in step with the episode
   index (read, then discard `stride - 1`), so image and pose are always the same frame.

2. Mesh origin. `register()` stores a camera-frame translation, and for a raw mesh vertex
   `p` the camera position is `s * R @ p + t` -- no re-centring. The old code projected
   `R @ (V - V.mean(0)) + t`, adding a constant `-R @ V.mean(0)` error of ~3 mm, i.e.
   1.4-2.2 px on every frame. Scale from the JSON is now applied too (it is 1.0 unless
   --allow-scale was used).

A missing prediction now renders the frame with an explicit "no pose" marker instead of
being dropped, because silently skipping frames is what let the two counters drift apart.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "reconstruction" / "modules" / "v2d_stereo_depth" / "lib"))
from stereo_depth import load_rig  # noqa: E402

DEFAULT_DATA = os.environ.get("V2D_PUBLIC", "/home/ubuntu/v2d_track3/data/hf/track_3/public")
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
PALETTE = [MAGENTA, CYAN, (0, 255, 255), (0, 165, 255), (0, 255, 0)]


def mesh_path(data, name):
    d = data / "mesh" / name
    for cand in (name, f"{name}_visual", f"{name}_collision"):
        p = d / f"{cand}.glb"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def silhouette(V, q, t, scale, rig, shape):
    """Project mesh vertices to a binary mask using the stored camera-frame pose."""
    R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    # pose is camera-frame for the *raw* mesh coordinates: X = s*R@p + t
    X = scale * (R @ V.T).T + np.asarray(t, np.float64)
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


def draw_object(out, sil, color, label):
    tint = out.copy()
    tint[sil > 0] = color
    out = np.where(sil[:, :, None] > 0,
                   (0.75 * out + 0.25 * tint).astype(np.uint8), out)
    cnts = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(out, cnts, -1, color, 3)
    return out, cnts


def put_label(out, lines):
    """Label with a solid backdrop; plain white text vanished on the light floor."""
    pad, y0, lh = 8, 12, 34
    h = lh * len(lines) + 2 * pad
    w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
            for s in lines) + 2 * pad
    cv2.rectangle(out, (8, 6), (8 + w, 6 + h), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(out, s, (8 + pad, y0 + pad + lh * i + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--object", nargs="+", default=None,
                    help="objects to draw; default = every object in the prediction")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", type=Path, default=Path(DEFAULT_DATA))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=float, default=12.0)
    a = ap.parse_args()

    rig = load_rig()
    ep = f"e{a.episode:03d}"
    ep_dir = a.predictions / ep / "ego_object_poses" / ep
    if not ep_dir.is_dir():
        raise SystemExit(f"no predictions in {ep_dir}")
    objects = a.object or sorted(p.name for p in ep_dir.iterdir() if p.is_dir())
    if not objects:
        raise SystemExit("no objects to draw")

    meshes, poses = {}, {}
    for name in objects:
        V = np.asarray(trimesh.load(str(mesh_path(a.data, name)),
                                    force="mesh", process=False).vertices, np.float64)
        if len(V) > 120000:
            V = V[np.random.default_rng(0).choice(len(V), 120000, replace=False)]
        meshes[name] = V
        poses[name] = ep_dir / name / "object_to_cam" / "ego_cam_c"

    vroot = a.data / "videos" / "chunk-000"
    vl = vroot / f"observation.images.{rig.left}" / f"episode_{a.episode:06d}.mp4"
    cap = cv2.VideoCapture(str(vl))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {vl}")
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # clamp so the requested window always lies inside the episode
    last = a.start + (a.frames - 1) * a.stride
    if a.start >= n_src:
        raise SystemExit(f"--start {a.start} >= episode length {n_src}")
    if last >= n_src:
        keep = (n_src - 1 - a.start) // a.stride + 1
        print(f"window {a.start}..{last} exceeds {n_src} frames; rendering {keep}")
        a.frames = keep

    # Land on a.start by counting from 0 rather than trusting a POS_FRAMES seek:
    # this is the exactness the old renderer was missing, so do not approximate it.
    for i in range(a.start):
        if not cap.read()[0]:
            raise SystemExit(f"episode has fewer than {a.start} frames")

    writer = None
    shape = None
    missed = []
    written = 0
    for k in range(a.frames):
        idx = a.start + k * a.stride
        # advance strictly in step with the episode index: read this frame, then
        # discard stride-1 so the next read lands exactly on idx + stride
        ok, fr = cap.read()
        if not ok:
            print(f"video ended early at episode frame {idx}")
            break
        for _ in range(a.stride - 1):
            if not cap.read()[0]:
                break

        if shape is None:
            shape = fr.shape[:2]
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     a.fps, (shape[1], shape[0]))
            if not writer.isOpened():
                raise SystemExit("VideoWriter failed to open")

        out = rig.rectify_pair(fr, fr)[0]
        lines = [f"ep{a.episode}  f{idx}/{n_src - 1}"]
        drawn = 0
        for ci, name in enumerate(objects):
            f = poses[name] / f"{idx:06d}.json"
            if not f.exists():
                missed.append((name, idx))
                continue
            r = json.loads(f.read_text())
            sc = float(np.atleast_1d(r.get("scale", [1.0]))[0])
            sil = silhouette(meshes[name], np.array(r["rotation"], np.float64),
                             r["translation"], sc, rig, shape)
            out, _ = draw_object(out, sil, PALETTE[ci % len(PALETTE)], name)
            lines.append(f"{name} z={r['translation'][2]:.2f}m")
            drawn += 1
        if drawn == 0:
            missed.append(("<all>", idx))
            cv2.putText(out, "NO PREDICTION FOR THIS FRAME", (shape[1] // 2 - 340,
                        shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (0, 0, 255), 4, cv2.LINE_AA)
        put_label(out, lines)
        writer.write(out)
        written += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not written:
        raise SystemExit("nothing rendered")
    print(f"wrote {written} frames ({a.start}..{a.start + (written - 1) * a.stride} "
          f"step {a.stride}) -> {a.out} ({os.path.getsize(a.out)} bytes)")
    if missed:
        uniq = sorted({i for _, i in missed})
        print(f"frames without a pose: {len(missed)} entries, indices {uniq[:20]}"
              f"{' ...' if len(uniq) > 20 else ''}")


if __name__ == "__main__":
    main()
