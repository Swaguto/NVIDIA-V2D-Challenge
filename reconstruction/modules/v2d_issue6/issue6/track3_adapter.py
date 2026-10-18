"""Prepare a frame-exact issue-6 bundle from public Track 3 stereo/CAD outputs.

The bundle uses the rectified stereo-left camera for color, depth, masks, and
object-to-camera poses. Reference object poses are never read by this adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .manifest import file_hash, load, write_json


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
    scored_kit: Path | None = None,
    camera_root: Path | None = None,
) -> Path:
    # Input paths are relative to the caller, whereas paths inside the manifest
    # are resolved relative to its output directory. Preserve their locations.
    data, results, output = data.resolve(), results.resolve(), output.resolve()
    # Import heavy dependencies only when an actual dataset bundle is requested.
    import cv2

    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo / "reconstruction/modules/v2d_stereo_depth/lib"))
    from stereo_depth import load_rig, stereo_depth

    if start < 0 or (frames <= 0 and scored_kit is None) or (output.exists() and any(output.iterdir())):
        raise ValueError("Choose nonnegative start, positive frames, and a fresh output directory")
    rig = load_rig(str(data / "meta/camera_calibration.json"))
    if rig.left != "ego_cam_c":
        raise ValueError(f"Track 3 pose output expects rectified ego_cam_c; got {rig.left}")
    ep = results / f"e{episode:03d}"
    if scored_kit is None:
        import pyarrow.parquet as pq
        parquet = data / "data/chunk-000" / f"episode_{episode:06d}.parquet"
        source = pq.ParquetFile(parquet)
        names = json.loads(source.schema_arrow.metadata[b"objects"])
        rows = pq.read_table(parquet, columns=["frame_index", "timestamp"]).to_pylist()
    else:
        sys.path.insert(0, str(repo / "scripts/track3"))
        from audit_scored_coverage import official_frames
        roster = json.loads((scored_kit / "data/track_3_evaluation_objects.json").read_text())["episodes"]
        if str(episode) not in roster:
            raise ValueError("Episode is outside the official scored roster")
        names = roster[str(episode)]
        count = official_frames(scored_kit, roster)[episode]
        if frames == 0:
            frames = count
        if start != 0 or frames != count:
            raise ValueError("Scored hand inputs require the complete original episode")
        # Scored data releases videos, not pose-label tables. Recover the clock
        # from the video; validate its actual decoded frame timestamps below.
        probe = cv2.VideoCapture(str(data / "videos/chunk-000" /
                                    f"observation.images.{rig.left}" / f"episode_{episode:06d}.mp4"))
        fps = probe.get(cv2.CAP_PROP_FPS)
        video_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        probe.release()
        if not np.isfinite(fps) or fps <= 0 or video_count != count:
            raise ValueError("Scored video clock/frame count differs from official roster")
        rows = [{"frame_index": i, "timestamp": i/fps} for i in range(count)]
    # Fail before decoding hundreds of depth frames when an object asset is absent.
    meshes = {name: mesh_for(data, name) for name in names}
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
            if scored_kit is not None:
                for capture in (left, right):
                    if not np.isclose(capture.get(cv2.CAP_PROP_POS_MSEC)/1000,
                                      row["timestamp"], atol=1e-5):
                        raise ValueError("Scored stereo timestamps are not on the declared clock")
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
            if camera_root is not None:
                camera_path = camera_root.resolve() / f"e{episode:03d}/camera" / rig.left / f"{stem}.json"
                record = json.loads(camera_path.read_text())
                if (record.get("episode") != episode or record.get("frame_index") != frame
                        or record.get("gt_use") != "independent"
                        or record.get("camera_frame") != rig.left):
                    raise ValueError("Camera evidence differs from scored frame")
                manifest_frames[-1].update(camera_pose_path=str(camera_path),
                                           camera_valid=record.get("status") == "observed")
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
        "objects": [{"id": name, "mesh": str(meshes[name]), "provenance": "predicted"} for name in names],
        "hand_source": hand_source,
        "settings": {"minimum_pixels": 256, "contact_threshold_m": 0.02, "bridge_frames": 15},
        "frames": manifest_frames,
    }
    manifest = output / "manifest.json"
    if evaluation is not None:
        m["evaluation"] = evaluation
    if scored_kit is not None:
        m["input_contract"] = dict(split="evaluation", object_slots="official roster order",
                                   timestamps="validated video presentation clock",
                                   reference_pose_tables_read=False,
                                   provided_geometry_is_not_claimed_as_reconstruction=True,
                                   official_roster_sha256=file_hash(scored_kit / "data/track_3_evaluation_objects.json"),
                                   official_sample_sha256=file_hash(scored_kit / "data/track_3_sample_submission.parquet"),
                                   calibration_sha256=file_hash(data / "meta/camera_calibration.json"),
                                   adapter_sha256=file_hash(Path(__file__)),
                                   stereo_videos_sha256={cam: file_hash(base / f"observation.images.{cam}" / f"episode_{episode:06d}.mp4") for cam in (rig.left, rig.right)})
        # Keep all generated/source geometry and observation links portable.
        for obj in m["objects"]:
            obj["mesh"] = os.path.relpath(obj["mesh"], output)
        for frame in m["frames"]:
            if "camera_pose_path" in frame:
                frame["camera_pose_path"] = os.path.relpath(frame["camera_pose_path"], output)
            for state in frame["objects"].values():
                for key in ("pose_path", "mask"):
                    if key in state:
                        state[key] = os.path.relpath(state[key], output)
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
    parser.add_argument("--frames", type=int, help="Public default 64; scored default complete episode")
    parser.add_argument("--scored-kit", type=Path, help="Official kit for label-free scored input creation")
    parser.add_argument("--camera-root", type=Path, help="Independent stereo camera results root")
    parser.add_argument("--hand-source-json", type=Path, help="HaMeR/Dyn-HaMR/import paths and settings")
    args = parser.parse_args()
    source = json.loads(args.hand_source_json.read_text()) if args.hand_source_json else {"backend": "hamer"}
    frames = args.frames if args.frames is not None else (0 if args.scored_kit else 64)
    print(export(args.data, args.results, args.episode, args.output, args.start, frames, source,
                 scored_kit=args.scored_kit, camera_root=args.camera_root))


if __name__ == "__main__":
    main()
