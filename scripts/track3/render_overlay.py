#!/usr/bin/env python3
"""Render an overlay video: CAD mesh drawn on the camera frame at the estimated pose.

This is the sanity check that matters most for pose work -- a good score can hide a
systematically wrong convention, but a visibly wrong overlay cannot. Draws the CAD wireframe
(solid) plus a semi-transparent silhouette fill, and optionally the segmented mask contour,
so registration quality is visible frame by frame.

Poses are in the rectified left-camera frame, which is also the frame the overlay image is in,
so no extrinsic is needed.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/ubuntu")
from stereo_depth import load_rig, stereo_depth  # noqa: E402


def load_mesh(name, root):
    """Load a CAD mesh. GLB is not readable by OpenCV, so go through trimesh."""
    import trimesh
    d = root / name
    for cand in ("_visual", ""):
        for ext in (".glb", ".ply", ".obj", ".stl"):
            p = d / f"{name}{cand}{ext}"
            if p.exists():
                m = trimesh.load(str(p), force="mesh", process=False)
                v = np.asarray(m.vertices, np.float32)
                f = np.asarray(m.faces, np.int32) if hasattr(m, "faces") else None
                return v, f
    return None, None


def mesh_to_rgb(pts, faces, alpha):
    """Vertex colours: hue varies with height so 3D structure is readable."""
    if pts is None or len(pts) == 0:
        return None
    z = pts[:, 2]
    z = (z - z.min()) / max(float(np.ptp(z)), 1e-9)
    img = np.zeros((len(pts), 1, 3), np.uint8)
    img[:, 0, 0] = (255 * (1 - z)).astype(np.uint8)   # B
    img[:, 0, 1] = (255 * z).astype(np.uint8)         # G
    img[:, 0, 2] = 255                                  # R
    return img


def draw_pose(img, R, t, P, mesh, colour, wire=True, thick=2):
    """Project CAD vertices with the rectified projection matrix and rasterise."""
    verts, faces = mesh
    if verts is None:
        return
    X = (R @ verts.T).T + t.reshape(1, 3)
    z = X[:, 2]
    good = z > 1e-3
    if good.sum() < 10:
        return
    px = (P[0, 0] * X[:, 0] / z + P[0, 2])
    py = (P[1, 1] * X[:, 1] / z + P[1, 2])
    proj = np.stack([px, py], 1)
    col = mesh_to_rgb(verts, faces, None)
    for i in range(len(verts)):
        if not good[i]:
            continue
        u, v = int(proj[i, 0]), int(proj[i, 1])
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            img[v, u] = col[i, 0]
    if faces is not None and wire:
        for f in faces[:4000]:
            idx = [j for j in f if good[j]]
            if len(idx) < 3:
                continue
            pts = proj[idx].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, colour, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--object", default=None)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--mesh-root", type=Path,
                    default=Path("/home/ubuntu/v2d_track3/data/hf/track_3/public/mesh"))
    ap.add_argument("--no-depth", action="store_true")
    a = ap.parse_args()

    rig = load_rig()
    ep = f"e{a.episode:03d}"
    base = a.predictions / ep / "ego_object_poses" / ep
    objects = sorted([p.name for p in base.iterdir() if p.is_dir()])
    if a.object:
        objects = [a.object]

    vid_root = Path("/home/ubuntu/v2d_track3/data/hf/track_3/public/videos/chunk-000")
    cap = cv2.VideoCapture(str(vid_root / f"observation.images.{rig.left}" /
                               f"episode_{a.episode:06d}.mp4"))
    cap_r = cv2.VideoCapture(str(vid_root / f"observation.images.{rig.right}" /
                                f"episode_{a.episode:06d}.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"),
                         a.fps, (W, H))
    print(f"{W}x{H} -> {a.out}  objects={objects}")

    meshes, pal = {}, {}
    for i, o in enumerate(objects):
        img, path = load_mesh(o, a.mesh_root)
        if img is None:
            print(f"  !! no mesh for {o}")
            meshes[o] = (None, None)
        else:
            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3] > 128
                v = np.stack([img[:, :, 2][alpha], img[:, :, 1][alpha],
                              img[:, :, 0][alpha]], 1).astype(np.float32) / 255
            else:
                v = img[:, :3][:, ::-1].reshape(-1, 3).astype(np.float32) / 255
            meshes[o] = (v, None)
            print(f"  {o}: {len(v)} verts")
        pal[o] = [(0, 0, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255)][i % 4]

    written = 0
    idx = 0
    while written < a.frames:
        ok, frame = cap.read()
        okr, right = cap_r.read()
        if not ok:
            break
        if idx % a.stride == 0 and idx >= a.start:
            info = []
            for o in objects:
                f = base / o / "object_to_cam" / "ego_cam_c" / f"{idx:06d}.json"
                if not f.exists():
                    continue
                rec = json.loads(f.read_text())
                t = np.array(rec["translation"], float)
                q = np.array(rec["rotation"], float)
                q /= np.linalg.norm(q)
                w, x, y, z = q
                R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                              [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
                draw_pose(frame, R, t, rig.P1, meshes[o], pal[o])
                info.append(f"{o} res={rec.get('residual', 0):.3f}")
            if not a.no_depth:
                d, _ = stereo_depth(rig, frame, right)
                d8 = np.clip((d / 2.0) * 255, 0, 255).astype(np.uint8)
                d8 = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
                m = d > 0
                frame[m] = (0.65 * frame[m] + 0.35 * d8[m]).astype(np.uint8)
            cv2.putText(frame, f"ep{a.episode} f{idx}  " + " | ".join(info),
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            vw.write(frame)
            written += 1
        idx += 1
    vw.release()
    print(f"wrote {written} frames (src index {a.start}..{idx})")


if __name__ == "__main__":
    main()