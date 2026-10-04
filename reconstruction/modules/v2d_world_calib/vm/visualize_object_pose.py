#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render-and-compare visualiser for fit_object_pose (issue #5).

Projects each object's mesh at the estimated object->world pose into the real
camera frames and composites:

  - magenta edges : SAM2 mask contour (2D supervision)
  - cyan   filled : mesh at the ESTIMATED pose (render shadow)
  - green  filled : mesh at the GT pose, only with --score-gt --show-gt

Per-frame overlay PNGs go under
``{work}/ego_object_poses/e{episode}/vis/{obj}/frames/`` and a contact sheet of
the requested frames is saved to ``.../vis/{obj}_overview.png``.

RGB lookup order for each frame (--rgb-root overrides):
work/rgb|frames|images/{cam}/{t}.png; falls back to a colour-mapped depth image
when RGB is unavailable for that frame.

Run on the VM inside the pipeline venv (CPU only; needs cv2 + trimesh):

  python -m v2d.world_calib.vm.visualize_object_pose \\
      --data-root ~/v2d_track3/data/hf/track_3/public \\
      --work ~/v2d_calib_work --episode 2 --objects white_pot \\
      --frames all --max-frames 8 --stride 3 --score-gt --show-gt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from v2d.world_calib.vm.fit_ego_rig import (
    _w_obj,
    ego_rig_matrix,
    js_to_mat,
    load_intrinsics,
    load_reference_loader,
    load_mesh,
)

COL_MASK = (255, 0, 255)
COL_EST = (255, 255, 0)
COL_GT = (0, 255, 0)
ALPHA = 0.55


def _load_rgb_first(potentials: list[Path]) -> np.ndarray | None:
    for p in potentials:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    return None


def _depth_cmap(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((720, 960, 3), dtype=np.uint8) + 32
    a = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if a is None:
        return np.zeros((720, 960, 3), dtype=np.uint8) + 32
    z = 65535.0 / np.maximum(a.astype(np.float64), 1.0) - 1.0
    z = np.clip((z - 0.1) / (2.0 - 0.1), 0, 1)
    return cv2.applyColorMap((255.0 * z).astype(np.uint8), cv2.COLORMAP_TURBO)


def _project(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    u = K[0, 0] * pts_cam[:, 0] / np.maximum(pts_cam[:, 2], 1e-3) + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / np.maximum(pts_cam[:, 2], 1e-3) + K[1, 2]
    return np.stack([u, v, pts_cam[:, 2] > 1e-3], axis=1)


def _render_mesh(img: np.ndarray, verts_cam: np.ndarray, faces: np.ndarray | None,
                 K: np.ndarray, color: tuple[int, int, int]) -> None:
    uv = _project(verts_cam, K)
    if faces is not None and len(faces):
        depth = render_depth(verts_cam, faces, K, img.shape[:2])
        visible = np.isfinite(depth)
        img[visible] = ((1 - ALPHA) * img[visible] + ALPHA * np.asarray(color)).astype(np.uint8)
    else:
        ok = uv[:, 2] > 0
        for x, y in uv[ok, :2].astype(np.int32):
            cv2.circle(img, (int(x), int(y)), 2, color, -1)


def render_depth(vertices, faces, K, shape):
    """Frontmost triangle depth with perspective interpolation, metres (+z forward)."""
    faces = np.asarray(faces, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3 or (faces < 0).any() or (faces >= len(vertices)).any():
        raise ValueError("faces must index the full, unreordered vertex array")
    height, width = shape
    depth = np.full(shape, np.inf)
    uv = _project(vertices, K)
    for face in faces:
        if not uv[face, 2].all():
            continue
        tri = uv[face, :2]
        x0, y0 = np.maximum(np.floor(tri.min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(tri.max(0)).astype(int), [width - 1, height - 1])
        if x1 < x0 or y1 < y0:
            continue
        x, y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        a, b, c = tri
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        wa = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
        wb = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
        wc = 1 - wa - wb
        inside = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9)
        invz = wa / vertices[face[0], 2] + wb / vertices[face[1], 2] + wc / vertices[face[2], 2]
        z = np.where(inside, 1 / np.maximum(invz, 1e-12), np.inf)
        block = depth[y0:y1 + 1, x0:x1 + 1]
        np.minimum(block, z, out=block)
    return depth


def camera_from_source(source, cam, left, right, rig):
    """Use exactly the camera artifact hashed by the tracker; derive the right view."""
    if not source:
        return None
    path = Path(source["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError(f"camera input changed since fitting: {path}")
    C = js_to_mat(json.loads(raw))
    if cam == left:
        return C
    if cam == right:
        return rig @ C
    raise ValueError(f"camera {cam} is not in the calibrated ego pair")


def _draw_mask(img: np.ndarray, mask_path: Path) -> None:
    if not mask_path.exists():
        return
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return
    cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, COL_MASK, 2)


def _parse_frames(spec: str) -> list[int] | None:
    if spec in ("", "all"):
        return None
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = (int(x) for x in tok.split("-", 1))
            out += list(range(a, b + 1))
        else:
            out.append(int(tok))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--objects", default="", help="comma list; default = fitted dirs")
    ap.add_argument("--cam", default="",
                    help="camera to render into (default: ego stereo left)")
    ap.add_argument("--rgb-root", default="",
                    help="rgb dir; default auto-search work/rgb|frames|images/{cam}")
    ap.add_argument("--frames", default="", help="e.g. 12, 5-9 or all (default all)")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--score-gt", action="store_true")
    ap.add_argument("--show-gt", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    work = Path(args.work)
    ep_dir = work / "ego_object_poses" / f"e{args.episode:03d}"
    if not ep_dir.exists():
        sys.exit(f"{ep_dir}: not fitted yet (run fit_object_pose first)")

    rig = json.loads((data_root / "meta" / "camera_calibration.json").read_text())
    cam_left, cam_right, M = ego_rig_matrix(rig)
    cam = args.cam or cam_left
    K, W, H = load_intrinsics(work / "intrinsics" / f"{cam}.json")

    obj_dirs = sorted(d for d in ep_dir.iterdir() if (d / "object_to_world").is_dir())
    obj_order = [o.strip() for o in args.objects.split(",") if o.strip()]
    if obj_order:
        obj_dirs = [d for d in obj_dirs if d.name in obj_order]

    ref = None
    if args.score_gt:
        loader = load_reference_loader(Path(__file__).resolve().parents[4])
        ref = loader.load_reference(args.episode, str(data_root))

    for obj_dir in obj_dirs:
        obj = obj_dir.name
        loader = load_reference_loader(Path(__file__).resolve().parents[4])
        glb = loader.object_mesh_dir([obj], str(data_root))[obj]
        mm = load_mesh(glb)
        mesh = np.asarray(mm.vertices, dtype=np.float64)
        faces = np.asarray(mm.faces, dtype=np.int64)

        if ref is not None:
            try:
                b = list(ref.object_names).index(obj)
            except ValueError:
                b = None
        else:
            b = None

        vis_dir = ep_dir / "vis" / obj
        frame_dir = vis_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        pose_files = sorted((obj_dir / "object_to_world").glob("[0-9]*.json"))
        frames = _parse_frames(args.frames)
        frames = frames or [int(p.stem) for p in pose_files]
        frames = sorted(set(frames))[:: max(args.stride, 1)][: args.max_frames]

        saved = 0
        for t in frames:
            rgb_pots = []
            if args.rgb_root:
                rgb_pots.append(Path(args.rgb_root) / f"{t:06d}.png")
            for base in ("rgb", "frames", "images"):
                rgb_pots.append(work / base / cam / f"{t:06d}.png")
            img = _load_rgb_first(rgb_pots)
            if img is None:
                img = _depth_cmap(work / "depth" / cam / f"{t:06d}.png")
            if img is None or img.size == 0:
                continue
            img = cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img

            C = None
            X = None
            pf = obj_dir / "object_to_world" / f"{t:06d}.json"
            if pf.exists():
                record = json.loads(pf.read_text())
                X = js_to_mat(record)
                C = camera_from_source(record.get("camera_source"), cam, cam_left, cam_right, M)

            _draw_mask(img, work / "masks" / cam / obj / "0" / f"{t:06d}.png")

            if C is not None and X is not None:
                verts_w = (X[:3, :3] @ mesh.T).T + X[:3, 3]
                verts_cam = (C[:3, :3] @ verts_w.T).T + C[:3, 3]
                _render_mesh(img, verts_cam, faces, K, COL_EST)

            ref_ids = np.asarray(getattr(ref, "frame_index", [])) if ref is not None else np.array([])
            matches = np.flatnonzero(ref_ids == t)
            row = int(matches[0]) if len(matches) == 1 else None
            if args.show_gt and C is not None and ref is not None and b is not None and row is not None:
                if bool(ref.visible[row, b]):
                    W_gt = _w_obj(np.asarray(ref.pose_xyzw[row, b], dtype=np.float64))
                    verts_w = (W_gt[:3, :3] @ mesh.T).T + W_gt[:3, 3]
                    verts_cam = (C[:3, :3] @ verts_w.T).T + C[:3, 3]
                    _render_mesh(img, verts_cam, faces, K, COL_GT)

            status = "EST+GT" if (args.show_gt and X is not None and C is not None
                                  and ref is not None and b is not None
                                  and row is not None and bool(ref.visible[row, b])) else \
                     "EST" if (X is not None and C is not None) else "no pose"
            info = f"ep {args.episode} {obj} cam {cam} f {t:06d} [{status}]"
            cv2.putText(img, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(frame_dir / f"{t:06d}.png"), img)
            saved += 1

        tiles = []
        for p in sorted(frame_dir.glob("[0-9]*.png")):
            im = cv2.imread(str(p))
            if im is None:
                continue
            tiles.append(im)
            if len(tiles) >= 12:
                break
        if tiles:
            cols = max(1, min(4, len(tiles)))
            rows = int(np.ceil(len(tiles) / cols))
            th, tw = tiles[0].shape[:2]
            canvas = np.zeros((th * rows, tw * cols, 3), dtype=np.uint8)
            for i, im in enumerate(tiles):
                r, c = divmod(i, cols)
                canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im
            out_png = vis_dir / f"{obj}_overview.png"
            cv2.imwrite(str(out_png), canvas)
            print(f"{obj}: {saved} frame(s) -> {out_png}")
        else:
            print(f"{obj}: no frames rendered (check --rgb-root / work rgb layout)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
