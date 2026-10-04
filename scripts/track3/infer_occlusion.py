#!/usr/bin/env python3
"""Fill world-frame pose gaps with explicit interpolation/contact provenance.

Short interior gaps are interpolated. A trailing gap is propagated with a
nearby object only when their recent relative transform was stable; otherwise
the last observed world pose is held. These are inferred poses, never reported
as visual observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def matrix(record):
    q = np.asarray(record["rotation"], float)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    pose[:3, 3] = np.asarray(record["translation"], float)
    return pose


def pose_record(pose, episode, frame, method, template):
    q = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return {"episode": episode, "frame_index": frame,
            "rotation": q[[3, 0, 1, 2]].tolist(),
            "translation": pose[:3, 3].tolist(), "scale": [1.0, 1.0, 1.0],
            "status": "inferred", "inference_method": method,
            "coordinate_frame": "fixed_world",
            "gt_use": template.get("gt_use", "unknown"),
            "source_observation": template.get("observation_sha256")}


def interpolate(left, right, fraction):
    result = np.eye(4)
    result[:3, 3] = (1 - fraction) * left[:3, 3] + fraction * right[:3, 3]
    keyframes = Rotation.from_matrix(np.stack([left[:3, :3], right[:3, :3]]))
    result[:3, :3] = Slerp([0, 1], keyframes)([fraction]).as_matrix()[0]
    return result


def stable_relation(target, anchor, end, window=5):
    frames = [t for t in range(max(0, end - 2 * window), end + 1)
              if t in target and t in anchor]
    if len(frames) < window:
        return None
    frames = frames[-window:]
    rel = [np.linalg.inv(anchor[t]) @ target[t] for t in frames]
    positions = np.stack([p[:3, 3] for p in rel])
    if (np.linalg.norm(positions, axis=1).mean() > 0.15 or
            np.max(np.linalg.norm(positions - positions.mean(0), axis=1)) > 0.025):
        return None
    rotations = Rotation.from_matrix(np.stack([p[:3, :3] for p in rel]))
    mean_rot = rotations.mean()
    spread = (mean_rot.inv() * rotations).magnitude()
    if spread.max() > np.deg2rad(15):
        return None
    relation = np.eye(4)
    relation[:3, :3] = mean_rot.as_matrix()
    relation[:3, 3] = positions.mean(0)
    return relation


def run(predictions: Path, out: Path, episode: int, objects: list[str], frames: int,
        max_interpolate: int = 15):
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output must be a fresh directory: {out}")
    base = predictions / f"e{episode:03d}" / "ego_object_poses" / f"e{episode:03d}"
    records = {}
    poses = {}
    for name in objects:
        folder = base / name / "object_to_world"
        rows = {int(path.stem): json.loads(path.read_text())
                for path in folder.glob("[0-9]*.json")}
        if not rows:
            raise ValueError(f"no observed poses for {name}")
        if any(t < 0 or t >= frames for t in rows):
            raise ValueError(f"out-of-range frame for {name}")
        records[name] = rows
        poses[name] = {t: matrix(row) for t, row in rows.items()}
    output = {name: dict(rows) for name, rows in records.items()}
    methods = {name: {} for name in objects}
    for name in objects:
        observed = sorted(poses[name])
        for frame in range(frames):
            if frame in output[name]:
                continue
            before = max((t for t in observed if t < frame), default=None)
            after = min((t for t in observed if t > frame), default=None)
            if before is not None and after is not None and after - before - 1 <= max_interpolate:
                pose = interpolate(poses[name][before], poses[name][after],
                                   (frame - before) / (after - before))
                method = "short_gap_interpolation"
                template = records[name][before]
            elif before is not None:
                pose = poses[name][before]
                method = "world_hold"
                template = records[name][before]
                for other in objects:
                    if other == name or frame not in poses[other]:
                        continue
                    relation = stable_relation(poses[name], poses[other], before)
                    if relation is not None:
                        pose = poses[other][frame] @ relation
                        method = f"stable_contact_with_{other}"
                        break
            else:
                pose = poses[name][after]
                method = "head_backfill"
                template = records[name][after]
            output[name][frame] = pose_record(pose, episode, frame, method, template)
            methods[name][method] = methods[name].get(method, 0) + 1
    for name in objects:
        destination = (out / f"e{episode:03d}" / "ego_object_poses" / f"e{episode:03d}" /
                       name / "object_to_world")
        destination.mkdir(parents=True, exist_ok=True)
        for frame, record in sorted(output[name].items()):
            (destination / f"{frame:06d}.json").write_text(json.dumps(record))
    report = {"episode": episode, "frames": frames,
              "observed": {name: len(records[name]) for name in objects},
              "inferred_by_method": methods,
              "gt_use": sorted({r.get("gt_use", "unknown") for rows in records.values()
                                for r in rows.values()})}
    (out / f"e{episode:03d}" / "inference_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    names = [name.strip() for name in args.objects.split(",") if name.strip()]
    print(json.dumps(run(args.predictions, args.out, args.episode, names, args.frames), indent=2))


if __name__ == "__main__":
    main()
