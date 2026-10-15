# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned, mask-preserving Track 3 reconstruction-to-retarget contract.

This pilot adapter uses the accepted refinement scene directly. It does not
reconstruct MANO from parameters, interpolate missing hands, or invent contacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = "v2d.track3.episode.v1"
JOINT_NAMES = ["wrist"] + [
    f"{finger}{j}"
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for j in range(1, 5)
]
SIDES = ("left", "right")


def digest(path: Path) -> str:
    """Hash an artifact incrementally with SHA256."""
    with path.open("rb") as handle:
        checksum = (
            hashlib.file_digest(handle, "sha256")
            if hasattr(hashlib, "file_digest")
            else None
        )
        if checksum is None:
            checksum = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                checksum.update(chunk)
        return checksum.hexdigest()


def write_json(path: Path, data: dict) -> None:
    """Write an indented UTF-8 JSON artifact with a final newline."""
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def rigid_pose(affine: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract a positive uniform object scale rather than putting it in SE(3)."""
    pose = np.asarray(affine, dtype=np.float64).copy()
    scale = np.linalg.norm(pose[:3, :3], axis=0)
    if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("Invalid affine pose")
    if not np.allclose(scale, scale[0], rtol=1e-5, atol=1e-8) or scale[0] <= 0:
        raise ValueError("Object scale must be positive and uniform")
    pose[:3, :3] /= scale[0]
    require_se3(pose)
    return pose, float(scale[0])


def require_se3(poses: np.ndarray) -> None:
    """Reject nonfinite, scaled or reflected rigid transforms."""
    poses = np.asarray(poses)
    if poses.shape[-2:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("Expected finite SE(3) poses")
    rotations = poses[..., :3, :3]
    if (
        not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(
            np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=1e-5
        )
        or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)
    ):
        raise ValueError("Pose is not a proper rigid transform")


def transform_points(poses: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map batched local points through column-vector world transforms."""
    return np.einsum("tij,tkj->tki", poses[:, :3, :3], points) + poses[:, None, :3, 3]


def wrist_rotation(record: dict, side: str) -> np.ndarray:
    """Refinement applies right MANO first, then mirrors camera X for left.

    Conjugation by that reflection gives a proper rotation acting on the
    corresponding mirrored canonical hand. Fingertip orientations are not
    inferred: the consumer uses positional fingertip tasks only.
    """
    rotation = Rotation.from_rotvec(
        np.asarray(record["mano"]["global_orient"])
    ).as_matrix()
    if side == "left":
        mirror = np.diag([-1.0, 1.0, 1.0])
        rotation = mirror @ rotation @ mirror
    return rotation


def export_bundle(  # noqa: PLR0917 - Preserve the frozen exporter interface.
    scene_path: Path,
    trial: Path,
    inputs_path: Path,
    out: Path,
    source_commit: str,
    episode: int,
    provenance: list[Path] | None = None,
) -> dict:
    """Freeze the complete scene, provenance and explicit unknown observations."""
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    receipt = json.loads((trial / "run-receipt.json").read_text(encoding="utf-8"))
    source_hash = digest(scene_path)
    if (
        source_hash != receipt["scene_sha256"]
        or digest(trial / "scene.npz") != source_hash
    ):
        raise ValueError("Scene does not match the completed native run")
    with np.load(scene_path, allow_pickle=False) as source:
        frames = source["frame_indices"].copy()
        if not np.array_equal(frames, inputs["frames"]) or not np.array_equal(
            frames, inputs["source_frames"]
        ):
            raise ValueError(
                "Scene does not cover the declared complete source sequence"
            )
        if list(map(str, source["hand_sides"])) != list(SIDES):
            raise ValueError("Unexpected hand identity/order")
        n = len(frames)
        fps = float(source["fps"])
        camera = source["refined_camera_w2c"].copy()
        require_se3(camera)
        c2w = np.linalg.inv(camera)
        arrays = {
            "frame_indices": frames,
            "timestamps_s": frames.astype(float) / fps,
            "camera_c2w": c2w,
            "intrinsics": source["K"].copy(),
            "image_size": source["image_size"].copy(),
            "hand_valid": np.zeros((n, 2), dtype=bool),
            "hand_directly_observed": np.zeros((n, 2), dtype=bool),
            "hand_joints_world": np.full((n, 2, 21, 3), np.nan),
            "hand_wrist_world": np.full((n, 2, 4, 4), np.nan),
            "hand_vertices_world": np.full((n, 2, 778, 3), np.nan),
            "contact_known": np.zeros((n, 2, 5), dtype=bool),
            "contact_probability": np.full((n, 2, 5), np.nan),
            "wrench_known": np.zeros((n, 2), dtype=bool),
            "wrench_world_N_Nm": np.full((n, 2, 6), np.nan),
        }
        index = {int(frame): i for i, frame in enumerate(frames)}
        frozen = {
            "scene.npz": scene_path,
            "run-receipt.json": trial / "run-receipt.json",
            "inputs.json": inputs_path,
        }
        for path in provenance or []:
            relative = "provenance/" + path.name
            if relative in frozen:
                raise ValueError("Duplicate provenance filename")
            frozen[relative] = path
        for side_id, side in enumerate(SIDES):
            hand_frames = source[f"hand_{side_id}_frame_indices"]
            records = {int(p.stem): p for p in (trial / "hands" / side).glob("*.json")}
            if set(records) != set(map(int, hand_frames)):
                raise ValueError(f"{side} scene and pose record frames disagree")
            rows = np.asarray([index[int(f)] for f in hand_frames])
            joints = source[f"refined_hand_{side_id}_joints"]
            vertices = source[f"refined_hand_{side_id}_verts"]
            arrays[f"hand_{side}_faces"] = source[f"hand_{side_id}_faces"].copy()
            for j, frame in enumerate(hand_frames):
                path = records[int(frame)]
                record = json.loads(path.read_text(encoding="utf-8"))
                frozen[f"hands/{side}/{path.name}"] = path
                if record.get("frame_idx") != int(frame) or record.get("is_right") != (
                    side == "right"
                ):
                    raise ValueError(f"Hand identity mismatch at {side}/{frame}")
                if not record.get("diagnostics", {}).get("metric_valid", False):
                    continue
                row = rows[j]
                arrays["hand_valid"][row, side_id] = True
                arrays["hand_directly_observed"][row, side_id] = (
                    record.get("directly_observed") is True
                )
                arrays["hand_joints_world"][row, side_id] = transform_points(
                    c2w[[row]], joints[j : j + 1]
                )[0]
                arrays["hand_vertices_world"][row, side_id] = transform_points(
                    c2w[[row]], vertices[j : j + 1]
                )[0]
                wrist = np.eye(4)
                wrist[:3, :3] = c2w[row, :3, :3] @ wrist_rotation(record, side)
                wrist[:3, 3] = arrays["hand_joints_world"][row, side_id, 0]
                arrays["hand_wrist_world"][row, side_id] = wrist
        objects = []
        for i, source_name in enumerate(source["object_names"]):
            name = str(source_name)
            pose_scales = [
                rigid_pose(matrix) for matrix in source[f"refined_object_{i}_o2c"]
            ]
            arrays[f"object_{i}_world"] = c2w @ np.stack([p for p, _ in pose_scales])
            arrays[f"object_{i}_scale"] = np.asarray([s for _, s in pose_scales])
            arrays[f"object_{i}_vertices"] = source[f"object_{i}_verts"].copy()
            arrays[f"object_{i}_faces"] = source[f"object_{i}_faces"].copy()
            for path in (trial / "objects" / name).glob("*.json"):
                frozen[f"objects/{name}/{path.name}"] = path
            objects.append(
                {
                    "name": name,
                    "index": i,
                    "pose_source": "issue7_refined",
                    "geometry_provenance": "provided_development_CAD",
                    "submission_mesh_eligible": False,
                }
            )
        for path in (trial / "camera").glob("*.json"):
            frozen[f"camera/{path.name}"] = path
        out.mkdir(parents=True)
        sources = []
        for relative, original in sorted(frozen.items()):
            target = out / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = digest(original)
            shutil.copyfile(original, target)
            if digest(target) != before or digest(original) != before:
                raise ValueError(f"Source changed during freeze: {original}")
            sources.append(
                {
                    "path": f"source/{relative}",
                    "sha256": before,
                    "bytes": target.stat().st_size,
                }
            )
        np.savez_compressed(out / "episode.npz", **arrays)
    manifest = {
        "schema": SCHEMA,
        "episode": episode,
        "source_commit": source_commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frames": frames.tolist(),
        "fps": fps,
        "units": "metres",
        "quaternion_order": "wxyz",
        "transforms": "column vectors; named source_to_destination",
        "world": "Issue7 SLAM world in the background reference camera gauge",
        "camera_convention": "rectified OpenCV camera: X right, Y down, Z forward",
        "hand_sides": list(SIDES),
        "mano_joint_names": JOINT_NAMES,
        "wrist_rotation": "MANO global orientation; left mirror conjugation; then camera_to_world",
        "fingertip_orientation": "unknown; retargeting uses position-only tip tasks",
        "unknown_policy": "false validity and NaN geometry; no interpolation or contact fabrication",
        "contacts": "all fingertip contacts unknown; coarse source-review labels are not fingertip probabilities",
        "wrenches": "unknown; world-frame force XYZ in newtons, torque XYZ in newton-metres; never derived from proximity",
        "objects": objects,
        "object_observation_provenance": inputs.get("object_provenance", {}),
        "camera_provenance": inputs.get("camera_provenance", {}),
        "payload": {"path": "episode.npz", "sha256": digest(out / "episode.npz")},
        "sources": sources,
        "exporter_sha256": digest(Path(__file__)),
        "learned_policy_rollout": False,
    }
    write_json(out / "manifest.json", manifest)
    summary = validate_bundle(out)
    write_json(out / "validation.json", summary)
    return summary


def validate_bundle(root: Path) -> dict:
    """Check artifact hashes, coordinates, masks and unknown contact data."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("units") != "metres":
        raise ValueError("Unknown schema or units")
    if (
        manifest["hand_sides"] != list(SIDES)
        or manifest["mano_joint_names"] != JOINT_NAMES
    ):
        raise ValueError("Ambiguous hand or MANO joint ordering")
    for record in manifest["sources"] + [manifest["payload"]]:
        path = root / record["path"]
        if (
            not path.resolve().is_relative_to(root.resolve())
            or digest(path) != record["sha256"]
        ):
            raise ValueError(f"Hash or path integrity failure: {record['path']}")
    with np.load(root / manifest["payload"]["path"], allow_pickle=False) as data:
        frames = data["frame_indices"]
        n = len(frames)
        if (
            n == 0
            or not np.array_equal(frames, manifest["frames"])
            or not np.all(np.diff(frames) == 1)
        ):
            raise ValueError("Incomplete or unordered source sequence")
        fps = manifest["fps"]
        if (
            not np.isfinite(fps)
            or fps <= 0
            or not np.allclose(data["timestamps_s"], frames / fps)
        ):
            raise ValueError("Invalid timestamps or FPS")
        if data["camera_c2w"].shape != (n, 4, 4):
            raise ValueError("Camera coverage mismatch")
        require_se3(data["camera_c2w"])
        if (
            data["intrinsics"].shape != (3, 3)
            or not np.isfinite(data["intrinsics"]).all()
            or np.any(np.diag(data["intrinsics"]) <= 0)
        ):
            raise ValueError("Invalid camera calibration")
        if data["image_size"].shape != (2,) or np.any(data["image_size"] <= 0):
            raise ValueError("Invalid image size")
        valid = data["hand_valid"]
        if valid.shape != (n, 2) or valid.dtype != np.bool_:
            raise ValueError("Invalid hand validity mask")
        if data["hand_directly_observed"].shape != valid.shape or np.any(
            data["hand_directly_observed"] & ~valid
        ):
            raise ValueError("Invalid direct-observation mask")
        shapes = {
            "hand_joints_world": (n, 2, 21, 3),
            "hand_vertices_world": (n, 2, 778, 3),
            "hand_wrist_world": (n, 2, 4, 4),
        }
        for key, shape in shapes.items():
            if (
                data[key].shape != shape
                or not np.isfinite(data[key][valid]).all()
                or not np.isnan(data[key][~valid]).all()
            ):
                raise ValueError(f"Geometry and unknown mask disagree: {key}")
        require_se3(data["hand_wrist_world"][valid])
        if not np.allclose(
            data["hand_wrist_world"][valid, :3, 3],
            data["hand_joints_world"][valid, 0],
            atol=1e-6,
        ):
            raise ValueError("Wrist pose disagrees with MANO joint zero")
        known = data["contact_known"]
        probabilities = data["contact_probability"]
        if (
            known.shape != (n, 2, 5)
            or known.dtype != np.bool_
            or probabilities.shape != known.shape
        ):
            raise ValueError("Invalid contact shape")
        if np.any(known) or not np.isnan(probabilities).all():
            raise ValueError("Pilot bundle cannot manufacture fingertip contacts")
        if data["wrench_known"].shape != (n, 2) or data["wrench_world_N_Nm"].shape != (
            n,
            2,
            6,
        ):
            raise ValueError("Invalid wrench shape")
        if (
            np.any(data["wrench_known"])
            or not np.isnan(data["wrench_world_N_Nm"]).all()
        ):
            raise ValueError("Pilot bundle cannot manufacture measured wrenches")
        for obj in manifest["objects"]:
            i = obj["index"]
            require_se3(data[f"object_{i}_world"])
            if data[f"object_{i}_world"].shape != (n, 4, 4) or data[
                f"object_{i}_scale"
            ].shape != (n,):
                raise ValueError("Incomplete object trajectory")
            if not np.isfinite(data[f"object_{i}_scale"]).all() or np.any(
                data[f"object_{i}_scale"] <= 0
            ):
                raise ValueError("Invalid object scale")
        return {
            "status": "validated",
            "schema": SCHEMA,
            "frames": n,
            "hand_valid_frames": {
                side: int(valid[:, i].sum()) for i, side in enumerate(SIDES)
            },
            "hand_unknown_frames": {
                side: int((~valid[:, i]).sum()) for i, side in enumerate(SIDES)
            },
            "known_fingertip_contacts": 0,
            "objects": [o["name"] for o in manifest["objects"]],
            "source_files_verified": len(manifest["sources"]),
            "manifest_sha256": digest(root / "manifest.json"),
            "payload_sha256": manifest["payload"]["sha256"],
        }


def main() -> None:
    """Run the command with validated inputs and explicit output artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--provenance", type=Path, action="append", default=[])
    parser.add_argument("--episode", type=int, default=2)
    args = parser.parse_args()
    if args.validate:
        result = validate_bundle(args.validate)
    else:
        if any(
            x is None
            for x in (args.scene, args.trial, args.inputs, args.out, args.source_commit)
        ):
            parser.error(
                "export requires --scene, --trial, --inputs, --out and --source-commit"
            )
        result = export_bundle(
            args.scene,
            args.trial,
            args.inputs,
            args.out,
            args.source_commit,
            args.episode,
            args.provenance,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
