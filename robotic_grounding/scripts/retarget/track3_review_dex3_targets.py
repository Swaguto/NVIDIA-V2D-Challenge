# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit actual saved Dex3 site distances without changing recovered hands.

Legacy FrameTask translation-log norms can depend on unused tip orientations.
This tool measures Euclidean world distances and a morphology lower bound,
preserves missing/direct-observation masks and never accepts physical contact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source/robotic_grounding"))
from robotic_grounding.retarget.track3_bundle import (  # noqa: E402
    JOINT_NAMES,
    SIDES,
    digest,
    validate_bundle,
    write_json,
)
from robotic_grounding.retarget.track3_model_lock import (  # noqa: E402
    verify_robot_assets,
)

DEFAULT_TARGETS = ("wrist", "thumb4", "index4", "middle4")


def array_content_digest(arrays: dict[str, np.ndarray]) -> str:
    """Hash numeric content independently of NPZ ZIP metadata and byte order."""
    result = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        if value.dtype.kind not in "biuf":
            raise ValueError("Numeric and boolean review arrays required")
        dtype = value.dtype.newbyteorder("<")
        header = json.dumps(
            {"name": name, "dtype": dtype.str, "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        content = np.ascontiguousarray(value.astype(dtype, copy=False)).tobytes()
        result.update(len(header).to_bytes(8, "big"))
        result.update(header)
        result.update(len(content).to_bytes(8, "big"))
        result.update(content)
    return result.hexdigest()


def distribution(values: np.ndarray) -> dict:
    """Keep a hand with no supported samples explicitly unknown."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def minimum_long_finger_separation(urdf: Path, side: str) -> float:
    """Derive a distance bound only for the verified parallel-axis Dex3 model."""
    xml = ET.parse(urdf).getroot()
    base_z = []
    for finger in ("index", "middle"):
        for index in (0, 1):
            joint = xml.find(f"joint[@name='{side}_hand_{finger}_{index}_joint']")
            if joint is None:
                raise ValueError("Dex3 long-finger joint missing")
            axis_element = joint.find("axis")
            if axis_element is None:
                raise ValueError("Dex3 long-finger axis missing")
            parent = joint.find("parent")
            child = joint.find("child")
            expected_parent = (
                f"{side}_hand_palm_link"
                if index == 0
                else f"{side}_hand_{finger}_0_link"
            )
            if (
                parent is None
                or child is None
                or parent.attrib["link"] != expected_parent
                or child.attrib["link"] != f"{side}_hand_{finger}_{index}_link"
            ):
                raise ValueError("Unexpected Dex3 long-finger chain")
            origin = joint.find("origin")
            if origin is None:
                raise ValueError("Dex3 joint origin missing")
            xyz = np.fromstring(origin.attrib["xyz"], sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            axis = np.fromstring(axis_element.attrib["xyz"], sep=" ")
            if not np.array_equal(rpy, [0, 0, 0]) or not np.array_equal(
                axis, [0, 0, 1]
            ):
                raise ValueError("Parallel-axis Dex3 distance bound unavailable")
            if index == 0:
                base_z.append(float(xyz[2]))
            elif xyz[2] != 0:
                raise ValueError("Long-finger joint changes the Z separation")
        tip = xml.find(f"joint/child[@link='{side}_{finger}_tip']/..")
        if tip is None or tip.attrib.get("type") != "fixed":
            raise ValueError("Dex3 fixed tip missing")
        parent = tip.find("parent")
        if parent is None or parent.attrib["link"] != f"{side}_hand_{finger}_1_link":
            raise ValueError("Unexpected Dex3 tip chain")
        origin = tip.find("origin")
        if origin is None:
            raise ValueError("Dex3 fixed tip origin missing")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        if xyz[2] != 0:
            raise ValueError("Fixed tip changes the Z separation")
    return abs(base_z[0] - base_z[1])


def position_residuals(
    source_joints: np.ndarray,
    robot_sites: np.ndarray,
    targets: tuple[str, ...],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure scaled targets without inventing fingertip rotations."""
    if (
        len(targets) != 4
        or targets[0] != "wrist"
        or any(name not in JOINT_NAMES for name in targets)
    ):
        raise ValueError("Expected wrist and three named source joints")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive source-to-robot scale required")
    indices = [JOINT_NAMES.index(name) for name in targets]
    wrist = source_joints[:, :1]
    positions = wrist + (source_joints[:, indices] - wrist) * scale
    errors = np.linalg.norm(robot_sites - positions, axis=-1)
    return errors, positions


def observation_contract(source: dict, robot: dict) -> None:
    """Refuse coverage upgrades, holds, substituted frames and finite gaps."""
    for key in ("frame_indices", "timestamps_s"):
        if not np.array_equal(robot[key], source[key]):
            raise ValueError(f"Source {key} mismatch")
    for name, key in (
        ("robot_valid", "hand_valid"),
        ("source_hand_valid", "hand_valid"),
        ("source_hand_directly_observed", "hand_directly_observed"),
    ):
        if robot[name].dtype != np.bool_ or not np.array_equal(
            robot[name], source[key]
        ):
            raise ValueError(f"Source observation provenance mismatch: {name}")
    valid = source["hand_valid"]
    if (
        not np.isfinite(robot["robot_qpos"][valid]).all()
        or not np.isnan(robot["robot_qpos"][~valid]).all()
    ):
        raise ValueError("Robot missing poses must stay NaN")


def review(bundle: Path, retarget: Path, out: Path) -> dict:
    """Review a complete baseline artifact; experimental target changes are refused."""
    validation = validate_bundle(bundle)
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    manifest_path = retarget / "manifest.json"
    manifest_sha = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "v2d.track3.dex3.v1"
        or manifest.get("robot") != "dual_floating_dex3"
    ):
        raise ValueError("Unknown Dex3 trajectory schema")
    if any(key.startswith("development_") for key in manifest):
        raise ValueError(
            "Development target/cost overrides need their separate experimental audit"
        )
    if manifest["episode_bundle_manifest_sha256"] != validation["manifest_sha256"]:
        raise ValueError("Retarget uses a different source bundle")
    payload = retarget / manifest["payload"]["path"]
    if (
        not payload.resolve().is_relative_to(retarget.resolve())
        or digest(payload) != manifest["payload"]["sha256"]
    ):
        raise ValueError("Retarget payload integrity failure")
    assets = REPO / "source/robotic_grounding/robotic_grounding/assets"
    verify_robot_assets(manifest, assets)
    with np.load(bundle / "episode.npz", allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    with np.load(payload, allow_pickle=False) as archive:
        robot = {key: archive[key] for key in archive.files}
    observation_contract(source, robot)
    n = len(source["frame_indices"])
    if manifest["processed_frames"] != n or manifest["source_frames"] != n:
        raise ValueError("Incomplete Dex3 trajectory")
    errors = np.full((n, 2, 4), np.nan)
    reachability = np.full((n, 2), np.nan)
    reports = {}
    for si, side in enumerate(SIDES):
        valid = source["hand_valid"][:, si]
        names = manifest["joints"][side]["frame_names"]
        frames = robot[f"robot_{side}_frame_poses"]
        if frames.shape != (n, len(names), 7):
            raise ValueError("Saved FK frame shape mismatch")
        if not np.isfinite(frames[valid]).all() or not np.isnan(frames[~valid]).all():
            raise ValueError("Saved FK observation gaps must stay NaN")
        if not np.allclose(np.linalg.norm(frames[valid, :, 3:], axis=-1), 1, atol=1e-6):
            raise ValueError("Invalid saved FK quaternions")
        site_names = (
            f"{side}_hand_palm_link",
            f"{side}_thumb_tip",
            f"{side}_index_tip",
            f"{side}_middle_tip",
        )
        indices = [names.index(name) for name in site_names]
        sites = frames[valid][:, indices, :3]
        residual, targets = position_residuals(
            source["hand_joints_world"][valid, si],
            sites,
            DEFAULT_TARGETS,
            manifest["mano_to_robot_scale"],
        )
        separation = minimum_long_finger_separation(
            assets / f"urdfs/g1/dex3_{side}.urdf", side
        )
        robot_pair_distance = np.linalg.norm(sites[:, 2] - sites[:, 3], axis=-1)
        if np.any(robot_pair_distance < separation - 1e-9):
            raise ValueError(
                "Saved FK contradicts the Dex3 minimum fingertip separation"
            )
        pair_distance = np.linalg.norm(targets[:, 2] - targets[:, 3], axis=-1)
        bound = np.maximum(0, separation - pair_distance)
        errors[valid, si] = residual
        reachability[valid, si] = bound
        reports[side] = {
            "supported_frames": int(valid.sum()),
            "directly_observed_frames": int(
                source["hand_directly_observed"][:, si].sum()
            ),
            "source_joint_targets": list(DEFAULT_TARGETS),
            "per_site_world_distance_mm": {
                name: distribution(residual[:, i] * 1000)
                for i, name in enumerate(("wrist", "thumb", "index", "middle"))
            },
            "three_tip_world_distance_mm": distribution(residual[:, 1:] * 1000),
            "legacy_tip_translation_log_norm_mm": distribution(
                robot["task_position_errors_m"][valid, si, 1:] * 1000
            ),
            "minimum_robot_long_finger_tip_distance_m": separation,
            "combined_long_finger_error_lower_bound_mm": distribution(bound * 1000),
        }
    if (
        digest(manifest_path) != manifest_sha
        or digest(payload) != manifest["payload"]["sha256"]
    ):
        raise ValueError("Retarget artifact changed during review")
    verify_robot_assets(manifest, assets)
    if validate_bundle(bundle) != validation:
        raise ValueError("Source bundle changed during review")
    out.mkdir(parents=True)
    arrays = {
        "frame_indices": source["frame_indices"],
        "source_hand_valid": source["hand_valid"],
        "source_hand_directly_observed": source["hand_directly_observed"],
        "site_world_distance_m": errors,
        "combined_long_finger_error_lower_bound_m": reachability,
    }
    np.savez_compressed(out / "site-distances.npz", **arrays)
    report = {
        "schema": "v2d.dex3.target-review.v1",
        "source_sha256": digest(Path(__file__)),
        "bundle_manifest_sha256": validation["manifest_sha256"],
        "retarget_manifest_sha256": manifest_sha,
        "payload_sha256": digest(out / "site-distances.npz"),
        "array_content_sha256": array_content_digest(arrays),
        "per_hand": reports,
        "fk_recomputed": False,
        "source_observations_modified": False,
        "display_holds_used": False,
        "pose_accuracy_accepted": False,
        "physical_task_accepted": False,
        "scope": "saved FK versus frozen reconstructed joint targets; neither physical hand truth nor contact acceptance",
    }
    write_json(out / "report.json", report)
    return report


def main() -> None:
    """Run the same read-only review for any complete Track 3 baseline episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--retarget", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.bundle, args.retarget, args.out), indent=2))


if __name__ == "__main__":
    main()
