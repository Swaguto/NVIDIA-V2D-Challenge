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
    mesh_points_and_normals,
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
    if faces is not None and len(faces) and (faces < len(verts_cam)).all():
        cz = verts_cam[faces].mean(axis=1)
        order = np.argsort(-cz)
        ov = np.zeros_like(img)
        for f_ in faces[order]:
            pts = uv[f_]
            if not (pts[:, 2].all() and uv[f_, 2].all()):
                continue
            cv2.fillPoly(ov, [pts[:, :2].astype(np.int32)], color)
        cv2.addWeighted(ov, ALPHA, img, 1.0, 0.0, dst=img)
    else:
        ok = uv[:, 2] > 0
        for x, y in uv[ok, :2].astype(np.int32):
            cv2.circle(img, (int(x), int(y)), 2, color, -1)


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

    obj_dirs = sorted(d for d in ep_dir.iterdir() if d.is_dir())
    obj_order = [o.strip() for o in args.objects.split(",") if o.strip()]
    if obj_order:
        obj_dirs = [d for d in obj_dirs if d.name in obj_order]

    ref = None
    if args.score_gt:
        loader = load_reference_loader(Path(__file__).resolve().parents[4])
        ref = loader.load_reference(args.episode, str(data_root))

    for obj_dir in obj_dirs:
        obj = obj_dir.name
        mesh_src = data_root / "mesh" / obj
        glb = mesh_src / f"{obj}.glb"
        faces: np.ndarray | None = None
        try:
            import trimesh

            mm = trimesh.load(str(glb))
            if isinstance(mm, trimesh.Scene):
                mm = next(iter(mm.geometry.values()))
            faces = np.asarray(mm.faces, dtype=np.int64)
        except Exception:
            faces = None
        mesh, _ = mesh_points_and_normals(glb, 4000)
        mesh = np.asarray(mesh, dtype=np.float64)

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
            cpf = work / "ego_cam_poses" / obj / "world_to_cam" / cam / f"{t:06d}.json"
            if cpf.exists():
                C = js_to_mat(json.loads(cpf.read_text()))
            X = None
            pf = obj_dir / "object_to_world" / f"{t:06d}.json"
            if pf.exists():
                X = js_to_mat(json.loads(pf.read_text()))

            _draw_mask(img, work / "masks" / cam / obj / "0" / f"{t:06d}.png")

            if C is not None and X is not None:
                verts_w = (X[:3, :3] @ mesh.T).T + X[:3, 3]
                verts_cam = (C[:3, :3] @ verts_w.T).T + C[:3, 3]
                _render_mesh(img, verts_cam, faces, K, COL_EST)

            if args.show_gt and ref is not None and b is not None:
                if bool(ref.visible[t, b]):
                    W_gt = _w_obj(np.asarray(ref.pose_xyzw[t, b], dtype=np.float64))
                    verts_w = (W_gt[:3, :3] @ mesh.T).T + W_gt[:3, 3]
                    verts_cam = (C[:3, :3] @ verts_w.T).T + C[:3, 3]
                    _render_mesh(img, verts_cam, faces, K, COL_GT)

            status = "EST+GT" if (args.show_gt and X is not None and C is not None
                                  and ref is not None and b is not None
                                  and bool(ref.visible[t, b])) else \
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