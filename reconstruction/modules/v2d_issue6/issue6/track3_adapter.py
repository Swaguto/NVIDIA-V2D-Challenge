"""Prepare a frame-exact issue-6 bundle from public Track 3 stereo/CAD outputs.

The bundle uses the rectified stereo-left camera for color, depth, masks, and
object-to-camera poses. Reference object poses are never read by this adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .manifest import load, write_json


def observed_object_state(episode_root: Path, episode: int, name: str, frame: int, camera: str) -> dict:
    """Expose only a directly observed, matching-camera CAD pose to contacts."""
    episode_root = episode_root.resolve()
    pose = (
        episode_root
        / "poses"
        / f"e{episode:03d}"
        / "ego_object_poses"
        / f"e{episode:03d}"
        / name
        / "object_to_cam"
        / camera
        / f"{frame:06d}.json"
    )
    if not pose.is_file():
        return {"valid": False}
    record = json.loads(pose.read_text())
    if (
        record.get("status") != "observed"
        or record.get("gt_use") != "none"
        or record.get("camera_frame") != camera
        or record.get("coordinate_frame") != "rectified_camera"
        or record.get("frame_index") != frame
        or record.get("episode") != episode
    ):
        return {"valid": False}
    result = {"valid": True, "direction": "camera_from_object", "pose_path": str(pose)}
    mask = episode_root / "masks" / name / f"{frame:06d}.png"
    if mask.is_file():
        result["mask"] = str(mask)
    return result


def mesh_for(data: Path, name: str) -> Path:
    for suffix in ("", "_visual", "_collision"):
        candidate = data / "mesh" / name / f"{name}{suffix}.glb"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Public CAD mesh missing for {name}")


def export(
    data: Path,
    results: Path,
    episode: int,
    output: Path,
    start: int,
    frames: int,
    hand_source: dict,
    evaluation: dict | None = None,
) -> Path:
    # Input paths are relative to the caller, whereas paths inside the manifest
    # are resolved relative to its output directory. Preserve their locations.
    data, results, output = data.resolve(), results.resolve(), output.resolve()
    # Import heavy dependencies only when an actual dataset bundle is requested.
    import cv2
    import pyarrow.parquet as pq

    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo / "reconstruction/modules/v2d_stereo_depth/lib"))
    from stereo_depth import load_rig, stereo_depth

    if start < 0 or frames <= 0 or (output.exists() and any(output.iterdir())):
        raise ValueError("Choose nonnegative start, positive frames, and a fresh output directory")
    rig = load_rig(str(data / "meta/camera_calibration.json"))
    if rig.left != "ego_cam_c":
        raise ValueError(f"Track 3 pose output expects rectified ego_cam_c; got {rig.left}")
    ep = results / f"e{episode:03d}"
    parquet = data / "data/chunk-000" / f"episode_{episode:06d}.parquet"
    source = pq.ParquetFile(parquet)
    names = json.loads(source.schema_arrow.metadata[b"objects"])
    rows = pq.read_table(parquet, columns=["frame_index", "timestamp"]).to_pylist()
    selected = rows[start : start + frames]
    if len(selected) != frames or any(row["frame_index"] != start + i for i, row in enumerate(selected)):
        raise ValueError("Requested frames are absent or parquet frame IDs are noncontiguous")

    base = data / "videos/chunk-000"
    left = cv2.VideoCapture(str(base / f"observation.images.{rig.left}" / f"episode_{episode:06d}.mp4"))
    right = cv2.VideoCapture(str(base / f"observation.images.{rig.right}" / f"episode_{episode:06d}.mp4"))
    if not left.isOpened() or not right.isOpened():
        raise FileNotFoundError(f"Stereo videos for episode {episode}")
    left.set(cv2.CAP_PROP_POS_FRAMES, start)
    right.set(cv2.CAP_PROP_POS_FRAMES, start)
    output.mkdir(parents=True)
    for subdir in ("color", "depth", "valid"):
        (output / subdir).mkdir()
    intrinsics = {
        "id": rig.left,
        "convention": "rectified_pinhole",
        "width": int(rig.size[0]),
        "height": int(rig.size[1]),
        "fx": float(rig.P1[0, 0]),
        "fy": float(rig.P1[1, 1]),
        "cx": float(rig.P1[0, 2]),
        "cy": float(rig.P1[1, 2]),
    }
    manifest_frames = []
    try:
        for row in selected:
            frame = int(row["frame_index"])
            ok_l, raw_l = left.read()
            ok_r, raw_r = right.read()
            if not ok_l or not ok_r:
                raise ValueError(f"Stereo video ends before frame {frame}")
            rect_l, _ = rig.rectify_pair(raw_l, raw_r)
            depth, valid = stereo_depth(rig, raw_l, raw_r)
            if rect_l.shape[:2] != depth.shape or depth.shape != (intrinsics["height"], intrinsics["width"]):
                raise ValueError(f"Color/depth geometry mismatch at frame {frame}")
            stem = f"{frame:06d}"
            if not cv2.imwrite(str(output / "color" / f"{stem}.png"), rect_l):
                raise OSError(f"Could not write color frame {frame}")
            np.save(output / "depth" / f"{stem}.npy", np.where(valid, depth, np.nan))
            if not cv2.imwrite(str(output / "valid" / f"{stem}.png"), valid.astype(np.uint8) * 255):
                raise OSError(f"Could not write depth validity frame {frame}")
            manifest_frames.append(
                {
                    "id": frame,
                    "timestamp_s": float(row["timestamp"]),
                    "color": f"color/{stem}.png",
                    "depth_path": f"depth/{stem}.npy",
                    "depth_validity": f"valid/{stem}.png",
                    "objects": {name: observed_object_state(ep, episode, name, frame, rig.left) for name in names},
                }
            )
    finally:
        left.release()
        right.release()

    m = {
        "schema_version": 1,
        "episode": episode,
        "mode": "predicted_object_integration",
        "units": "metres",
        "camera_provenance": "predicted",
        "hand_provenance": "predicted",
        "color_camera": intrinsics,
        "depth": {
            "camera": intrinsics,
            "encoding": "metres",
            "color_from_depth": np.eye(4).tolist(),
            "provenance": "predicted",
        },
        "objects": [{"id": name, "mesh": str(mesh_for(data, name)), "provenance": "predicted"} for name in names],
        "hand_source": hand_source,
        "settings": {"minimum_pixels": 256, "contact_threshold_m": 0.02, "bridge_frames": 15},
        "frames": manifest_frames,
    }
    manifest = output / "manifest.json"
    if evaluation is not None:
        m["evaluation"] = evaluation
    write_json(manifest, m)
    load(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--hand-source-json", type=Path, help="HaMeR/Dyn-HaMR/import paths and settings")
    args = parser.parse_args()
    source = json.loads(args.hand_source_json.read_text()) if args.hand_source_json else {"backend": "hamer"}
    print(export(args.data, args.results, args.episode, args.output, args.start, args.frames, source))


if __name__ == "__main__":
    main()
