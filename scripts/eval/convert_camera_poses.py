#!/usr/bin/env python3
"""Convert moving-camera CAD observations to one fixed world gauge.

T_camera_object = C(t) @ T_world_object. Original camera trajectories applied to
rectified observations require H1 @ C(t), with H1 containing the recorded R1.
This conversion does not estimate egomotion or remove GT dependence from cameras.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read_transform(record):
    q = np.asarray(record["rotation"], dtype=float)
    t = np.asarray(record["translation"], dtype=float)
    if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("transform requires finite xyz and nonzero wxyz rotation")
    if not np.allclose(record.get("scale", 1.0), 1.0):
        raise ValueError("fixed-world pose conversion requires metric scale 1")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = t
    return result


def camera_to_world(object_to_camera, camera_from_world, rectification=None):
    C = np.asarray(camera_from_world, dtype=float)
    if rectification is not None:
        H = np.eye(4)
        H[:3, :3] = rectification
        C = H @ C
    return np.linalg.inv(C) @ object_to_camera


def convert_episode(predictions, camera_root, out, episode, cam="ego_cam_c",
                    rectification=None, require_independent=False, rectification_sha256=None):
    predictions, camera_root, out = map(Path, (predictions, camera_root, out))
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be a fresh directory")
    source = predictions / f"e{episode:03d}" / "ego_object_poses" / f"e{episode:03d}"
    if not source.exists():
        source = predictions / "ego_object_poses" / f"e{episode:03d}"
    if not source.exists():
        raise FileNotFoundError(source)
    # Validate every conversion before writing, so missing cameras cannot create partial tracks.
    pending, sources = [], {}
    for obj in sorted(source.iterdir()):
        seen_frames = set()
        for pose_path in sorted((obj / "object_to_cam" / cam).glob("[0-9]*.json")):
            frame = int(pose_path.stem)
            if frame in seen_frames:
                raise ValueError(f"duplicate observation frame {frame}: {pose_path}")
            seen_frames.add(frame)
            pose = json.loads(pose_path.read_text())
            if pose.get("frame_index", frame) != frame or pose.get("episode", episode) != episode:
                raise ValueError(f"object frame/episode mismatch: {pose_path}")
            if pose.get("status", "observed") != "observed":
                continue
            camera_path = camera_root / cam / pose_path.name
            raw = camera_path.read_bytes()
            camera = json.loads(raw)
            if camera.get("frame_index", frame) != frame or camera.get("episode", episode) != episode:
                raise ValueError(f"camera frame/episode mismatch: {camera_path}")
            if camera.get("camera_frame", cam) != cam or pose.get("camera_frame", cam) != cam:
                raise ValueError("camera axes do not match the observations")
            camera_status = camera.get("status", "observed")
            if camera_status not in ("observed", "inferred"):
                raise ValueError(f"camera has invalid status: {camera_path}")
            if (pose.get("timestamp") is not None and camera.get("timestamp") is not None
                    and not np.isclose(pose["timestamp"], camera["timestamp"], rtol=0, atol=1e-9)):
                raise ValueError("camera/object timestamps do not match")
            gt_use = camera.get("gt_use", "unknown")
            observation_gt_use = pose.get("gt_use", "unknown")
            if require_independent and gt_use != "independent":
                raise ValueError(f"camera provenance is {gt_use}: {camera_path}")
            if require_independent and observation_gt_use not in ("none", "independent"):
                raise ValueError(f"observations lack GT-free inference provenance: {pose_path}")
            axes = pose.get("coordinate_frame", "original_camera")
            if pose.get("rectification_sha256") is not None and pose["rectification_sha256"] != rectification_sha256:
                raise ValueError("rectification manifest does not match the observation hash")
            camera_axes = camera.get("coordinate_frame", "original_camera")
            if axes not in ("original_camera", "rectified_camera") or camera_axes not in ("original_camera", "rectified_camera"):
                raise ValueError("camera observations and trajectory must declare their camera axes")
            H = None
            if axes != camera_axes:
                if axes != "rectified_camera" or camera_axes != "original_camera" or rectification is None:
                    raise ValueError("rectified observations require the camera's recorded R1 rotation")
                H = np.asarray(rectification, dtype=float)
                if H.shape != (3, 3) or not np.isfinite(H).all() or not np.allclose(H.T @ H, np.eye(3), atol=1e-10) or not np.isclose(np.linalg.det(H), 1, atol=1e-10):
                    raise ValueError("R1 must be a proper rotation")
            X = camera_to_world(read_transform(pose), read_transform(camera), H)
            q = Rotation.from_matrix(X[:3, :3]).as_quat()
            provenance = {"path": str(camera_path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                          "gt_use": gt_use, "camera_frame": cam}
            combined_gt_use = ("gt_assisted" if "gt_assisted" in (gt_use, observation_gt_use)
                               else "unknown" if "unknown" in (gt_use, observation_gt_use)
                               else "metadata_only" if observation_gt_use == "metadata_only" else gt_use)
            sources[str(frame)] = provenance
            result = {"rotation": q[[3, 0, 1, 2]].tolist(), "translation": X[:3, 3].tolist(),
                      "scale": [1.0] * 3, "episode": episode, "frame_index": frame,
                      "timestamp": pose.get("timestamp", camera.get("timestamp")),
                      "status": camera_status, "coordinate_frame": "fixed_world",
                      "gt_use": combined_gt_use, "camera_source": provenance,
                      "observation_gt_use": observation_gt_use,
                      "observation_sha256": hashlib.sha256(pose_path.read_bytes()).hexdigest()}
            if rectification_sha256 is not None:
                result["rectification_sha256"] = rectification_sha256
            dest = out / f"e{episode:03d}" / "ego_object_poses" / f"e{episode:03d}" / obj.name / "object_to_world" / pose_path.name
            pending.append((dest, result))
    if not pending:
        raise ValueError("no accepted camera-space observations found")
    for path, result in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    report = out / f"e{episode:03d}" / "ego_object_poses" / f"e{episode:03d}" / "conversion_report.json"
    report.write_text(json.dumps({"episode": episode, "poses": len(pending),
        "coordinate_frame": "fixed_world", "camera_sources": sources}, indent=2))
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--camera-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--cam", default="ego_cam_c")
    parser.add_argument("--rectification", type=Path, help="JSON containing R1 for this episode/calibration")
    parser.add_argument("--require-independent", action="store_true")
    args = parser.parse_args()
    R1 = json.loads(args.rectification.read_text())["R1"] if args.rectification else None
    rect_hash = hashlib.sha256(args.rectification.read_bytes()).hexdigest() if args.rectification else None
    if args.rectification:
        metadata = json.loads(args.rectification.read_text())
        if metadata.get("episode", args.episode) != args.episode or metadata.get("camera_frame", args.cam) != args.cam:
            raise ValueError("rectification episode/camera mismatch")
    print(f"converted {convert_episode(args.predictions, args.camera_root, args.out, args.episode, args.cam, R1, args.require_independent, rect_hash)} poses")


if __name__ == "__main__":
    main()
