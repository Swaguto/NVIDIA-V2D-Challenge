"""Isolate public episode names and video clocks from reference pose tables.

Only the Parquet schema metadata is inspected. Pose columns are neither decoded
nor copied into the inference view. Freeze episode selection before evaluation;
freeze the method separately before running the declared held-out validation.
"""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make(source, metadata_source, episodes, out):
    source, metadata_source = source.resolve(), metadata_source.resolve()
    if out.exists():
        raise FileExistsError(out)
    if metadata_source.is_relative_to(source):
        raise ValueError(
            "Keep reference metadata tables outside the inference source tree"
        )
    if any((source / name).exists() for name in ("data", "mesh", "urdf")):
        raise ValueError(
            "Public inference view must expose videos and calibration only"
        )
    if (
        not episodes
        or len(set(episodes)) != len(episodes)
        or not all(type(ep) is int and ep >= 0 for ep in episodes)
    ):
        raise ValueError("Predeclare distinct nonnegative public episode IDs")
    report = {
        "schema": "v2d.public-video-validation-roster.v1",
        "split": "public",
        "episodes": {},
        "video_axes": {},
        "selected_episodes": list(episodes),
        "episode_selection_recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method_frozen": False,
        "held_out_results_used_for_selection": False,
        "reference_pose_values_read": False,
        "metadata_access": "Parquet schema metadata only; no table/column read",
        "metadata_tables_excluded_from_inference_mount": True,
        "metadata_source_files_sha256": {},
        "calibration_sha256": sha(source / "meta/camera_calibration.json"),
        "producer_sha256": sha(Path(__file__)),
    }
    for ep in episodes:
        path = metadata_source / f"episode_{ep:06d}.parquet"
        metadata = pq.ParquetFile(path).schema_arrow.metadata
        names = json.loads(metadata[b"objects"])
        if (
            not names
            or len(set(names)) != len(names)
            or not all(isinstance(n, str) and n.isidentifier() for n in names)
        ):
            raise ValueError("Invalid public object-name schema metadata")
        axes = {}
        for cam in ["ego_cam_a", "ego_cam_b", "ego_cam_c"]:
            video = (
                source
                / f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"
            )
            cap = cv2.VideoCapture(str(video))
            try:
                if not cap.isOpened():
                    raise ValueError("Missing declared public source video")
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                if count < 2 or not np.isfinite(fps) or fps <= 0:
                    raise ValueError("Invalid declared public video clock")
                axes[cam] = {"frames": count, "fps": fps, "sha256": sha(video)}
            finally:
                cap.release()
        left = axes["ego_cam_c"]
        if any(
            a["frames"] != left["frames"]
            or not np.isclose(a["fps"], left["fps"], rtol=0, atol=1e-8)
            for a in axes.values()
        ):
            raise ValueError("Public camera video clocks differ")
        report["episodes"][str(ep)] = names
        report["video_axes"][str(ep)] = {
            "frames": left["frames"],
            "fps": left["fps"],
            "cameras": axes,
        }
        report["metadata_source_files_sha256"][str(ep)] = sha(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "metadata-source", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    a = p.parse_args()
    print(json.dumps(make(a.source, a.metadata_source, a.episodes, a.out), indent=2))
