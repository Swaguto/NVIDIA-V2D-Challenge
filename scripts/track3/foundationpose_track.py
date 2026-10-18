#!/usr/bin/env python3
"""Track one public CAD object with FoundationPose and rectified stereo depth.

Use masks from temporal_masks.py. Missing or depth-poor masks remain gaps; a
later world-frame inference stage may fill them with explicit provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("V2D_RUNTIME", str(ROOT / "reconstruction/modules/v2d_stereo_depth/lib")))
from stereo_depth import load_rig, stereo_depth  # noqa: E402


def mesh_for(data: Path, name: str) -> Path:
    folder = data / "mesh" / name
    for suffix in ("", "_visual", "_collision"):
        path = folder / f"{name}{suffix}.glb"
        if path.is_file():
            return path
    raise FileNotFoundError(f"CAD missing for {name}")


def run(episode: int, name: str, data: Path, masks: Path, out: Path,
        register_iterations=5, track_iterations=2, max_frames=0,
        reinit_every=0):
    target = (out / f"e{episode:03d}" / "ego_object_poses" / f"e{episode:03d}" /
              name / "object_to_cam" / "ego_cam_c")
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"object output must be fresh: {target}")
    # The FoundationPose installation and weights are supplied through the VM.
    import torch
    torch.backends.cudnn.enabled = False
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

    rig = load_rig()
    mesh_path = mesh_for(data, name)
    mesh_sha256 = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    # FoundationPose's renderer expects SimpleMaterial.image, while the public
    # GLBs carry PBRMaterial.baseColorTexture.
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        material = mesh.visual.material
        image = getattr(material, "baseColorTexture", None)
        if image is not None:
            mesh.visual.material = trimesh.visual.material.SimpleMaterial(image=image)
        else:
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh, vertex_colors=np.tile([128, 128, 128, 255], (len(mesh.vertices), 1)))
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    tracker = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                             mesh=mesh, scorer=scorer, refiner=refiner,
                             glctx=dr.RasterizeCudaContext(), debug=0)
    base = data / "videos/chunk-000"
    left = cv2.VideoCapture(str(base / f"observation.images.{rig.left}" /
                                f"episode_{episode:06d}.mp4"))
    right = cv2.VideoCapture(str(base / f"observation.images.{rig.right}" /
                                 f"episode_{episode:06d}.mp4"))
    if not left.isOpened() or not right.isOpened():
        raise FileNotFoundError(f"stereo episode {episode}")
    K = np.asarray(rig.P1[:3, :3], dtype=np.float64)
    directory = (out / f"e{episode:03d}" / "ego_object_poses" / f"e{episode:03d}" /
                 name / "object_to_cam" / rig.left)
    directory.mkdir(parents=True)
    last_frame = None
    last_pose = None
    last_registration = None
    observed, missing, rejected = [], [], []
    frame = 0
    while True:
        ok_l, raw_l = left.read()
        ok_r, raw_r = right.read()
        if ok_l != ok_r:
            raise ValueError("stereo frame-count mismatch")
        if not ok_l or (max_frames and frame >= max_frames):
            break
        rect_l, _ = rig.rectify_pair(raw_l, raw_r)
        rgb = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)
        depth, valid = stereo_depth(rig, raw_l, raw_r)
        mask_path = masks / name / f"{frame:06d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
        if mask is None or mask.shape != depth.shape or ((mask > 127) & valid).sum() < 100:
            missing.append(frame)
            last_frame = None
            frame += 1
            continue
        try:
            periodic = (reinit_every > 0 and last_registration is not None and
                        frame - last_registration >= reinit_every)
            if last_frame is None or periodic:
                pose = tracker.register(K=K, rgb=rgb, depth=depth, ob_mask=mask > 127,
                                        iteration=register_iterations)
                method = "register_periodic" if periodic and last_frame is not None else "register"
            else:
                pose = tracker.track_one(rgb=rgb, depth=depth, K=K,
                                         iteration=track_iterations)
                method = "track"
        except Exception as error:  # keep the batch moving; report each failure
            traceback.print_exc()
            rejected.append({"frame": frame, "error": f"{type(error).__name__}: {error}"})
            last_frame = None
            frame += 1
            continue
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(pose).all() or pose[2, 3] <= 0.05:
            rejected.append({"frame": frame, "error": "invalid pose"})
            last_frame = None
            frame += 1
            continue
        if last_pose is not None and last_frame is not None:
            step = np.linalg.inv(last_pose) @ pose
            if (np.linalg.norm(step[:3, 3]) > 0.15 or
                    Rotation.from_matrix(step[:3, :3]).magnitude() > np.deg2rad(40)):
                rejected.append({"frame": frame, "error": "implausible pose jump"})
                last_frame = None
                frame += 1
                continue
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        record = {"episode": episode, "frame_index": frame, "object": name,
                  "mesh_sha256": mesh_sha256,
                  "translation": pose[:3, 3].tolist(),
                  "rotation": quat[[3, 0, 1, 2]].tolist(),
                  "scale": 1.0, "status": "observed", "gt_use": "none",
                  "coordinate_frame": "rectified_camera", "camera_frame": rig.left,
                  "method": f"foundationpose_{method}",
                  "mask_pixels": int((mask > 127).sum()),
                  "depth_pixels": int(((mask > 127) & valid).sum())}
        (directory / f"{frame:06d}.json").write_text(json.dumps(record))
        observed.append(frame)
        last_frame, last_pose = frame, pose
        if method.startswith("register"):
            last_registration = frame
        if frame % 25 == 0:
            print(f"  FP {name} frame {frame}: {method}", flush=True)
        frame += 1
    left.release()
    right.release()
    report = {"episode": episode, "object": name, "frames": frame,
              "mesh_sha256": mesh_sha256,
              "observed": len(observed), "missing_mask_or_depth": missing,
              "rejected": rejected, "gt_use": "none",
              "reinit_every": reinit_every, "register_iterations": register_iterations,
              "track_iterations": track_iterations}
    (out / f"e{episode:03d}" / f"{name}_fp_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get(
        "V2D_PUBLIC", "/home/ubuntu/v2d_track3/data/hf/track_3/public")))
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--register-iterations", type=int, default=5)
    parser.add_argument("--track-iterations", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--reinit-every", type=int, default=0,
                        help="re-register from the current mask every N frames; 0 disables")
    args = parser.parse_args()
    print(run(args.episode, args.object, args.data, args.masks, args.out,
              args.register_iterations, args.track_iterations, args.max_frames,
              args.reinit_every), flush=True)


if __name__ == "__main__":
    main()
