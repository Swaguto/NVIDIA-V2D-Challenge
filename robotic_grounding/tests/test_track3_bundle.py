# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controls for world gauges, hand identity, unknowns and artifact integrity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

FILE = (
    Path(__file__).resolve().parents[1]
    / "source/robotic_grounding/robotic_grounding/retarget/track3_bundle.py"
)
spec = importlib.util.spec_from_file_location("track3_bundle", FILE)
assert spec is not None and spec.loader is not None
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def fixture(tmp: Path) -> tuple[Path, Path, Path]:
    """Export a synthetic three-frame scene with a single observed pose per hand."""
    trial = tmp / "trial"
    trial.mkdir()
    camera = np.repeat(np.eye(4)[None], 3, axis=0)
    camera[:, :3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    camera[:, :3, 3] = [1, 2, 3]
    points = np.zeros((1, 21, 3))
    points[:] = [0.1, 0.2, 0.3]
    arrays = {
        "frame_indices": np.arange(3),
        "fps": np.array(20.0),
        "hand_sides": np.array(bundle.SIDES),
        "refined_camera_w2c": camera,
        "K": np.diag([500.0, 500.0, 1.0]),
        "image_size": np.array([1280, 800]),
        "object_names": np.array(["box"]),
        "refined_object_0_o2c": np.repeat(np.eye(4)[None], 3, axis=0),
        "object_0_verts": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "object_0_faces": np.array([[0, 1, 2]]),
    }
    for i, side in enumerate(bundle.SIDES):
        arrays[f"hand_{i}_frame_indices"] = np.array([1])
        arrays[f"refined_hand_{i}_joints"] = points.copy()
        arrays[f"refined_hand_{i}_verts"] = np.repeat(points[:, :1], 778, axis=1)
        arrays[f"hand_{i}_faces"] = np.array([[0, 1, 2]])
        path = trial / "hands" / side / "000001.json"
        path.parent.mkdir(parents=True)
        bundle.write_json(
            path,
            {
                "frame_idx": 1,
                "is_right": side == "right",
                "directly_observed": True,
                "diagnostics": {"metric_valid": True},
                "mano": {"global_orient": [0.1, 0.2, 0.3]},
            },
        )
    np.savez(trial / "scene.npz", **arrays)
    bundle.write_json(
        trial / "run-receipt.json", {"scene_sha256": bundle.digest(trial / "scene.npz")}
    )
    inputs = tmp / "inputs.json"
    bundle.write_json(inputs, {"frames": [0, 1, 2], "source_frames": [0, 1, 2]})
    out = tmp / "export"
    bundle.export_bundle(trial / "scene.npz", trial, inputs, out, "a" * 40, 2)
    return trial, inputs, out


def alter_payload(root: Path, change: Callable[[dict[str, np.ndarray]], None]) -> None:
    """Modify the synthetic payload and update its declared hash for schema controls."""
    with np.load(root / "episode.npz") as data:
        arrays = dict(data)
    change(arrays)
    np.savez_compressed(root / "episode.npz", **arrays)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["payload"]["sha256"] = bundle.digest(root / "episode.npz")
    bundle.write_json(root / "manifest.json", manifest)


def test_export_preserves_unknowns_and_changes_world_gauge(tmp_path: Path) -> None:
    """Verify export preserves unknowns and changes world gauge."""
    _, _, out = fixture(tmp_path)
    summary = bundle.validate_bundle(out)
    assert summary["hand_valid_frames"] == {"left": 1, "right": 1}
    with np.load(out / "episode.npz") as data:
        # Independently calculated inverse of the 90-degree Y camera transform.
        assert np.allclose(data["hand_joints_world"][1, 0, 0], [2.7, -1.8, -0.9])
        assert np.isnan(data["hand_joints_world"][[0, 2]]).all()
        assert not data["contact_known"].any()
        assert np.isnan(data["contact_probability"]).all()


def test_freeze_rejects_changed_source_scene(tmp_path: Path) -> None:
    """Verify freeze rejects changed source scene."""
    trial, inputs, _ = fixture(tmp_path)
    with (trial / "scene.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="completed native run"):
        bundle.export_bundle(
            trial / "scene.npz", trial, inputs, tmp_path / "other", "a" * 40, 2
        )


def test_freeze_never_overwrites_existing_export(tmp_path: Path) -> None:
    """Verify freeze never overwrites existing export."""
    trial, inputs, out = fixture(tmp_path)
    with pytest.raises(FileExistsError):
        bundle.export_bundle(trial / "scene.npz", trial, inputs, out, "a" * 40, 2)


def test_validator_rejects_modified_frozen_file(tmp_path: Path) -> None:
    """Verify validator rejects modified frozen file."""
    _, _, out = fixture(tmp_path)
    (out / "source/hands/left/000001.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        bundle.validate_bundle(out)


def test_validator_rejects_geometry_in_unknown_interval(tmp_path: Path) -> None:
    """Verify validator rejects geometry in unknown interval."""
    _, _, out = fixture(tmp_path)
    alter_payload(out, lambda d: d["hand_joints_world"].__setitem__((0, 0), 0))
    with pytest.raises(ValueError, match="unknown mask"):
        bundle.validate_bundle(out)


def test_validator_rejects_scaled_camera_transform(tmp_path: Path) -> None:
    """Verify validator rejects scaled camera transform."""
    _, _, out = fixture(tmp_path)
    alter_payload(out, lambda d: d["camera_c2w"].__setitem__((0, 0, 0), 5))
    with pytest.raises(ValueError, match="rigid"):
        bundle.validate_bundle(out)


def test_validator_rejects_fabricated_contact(tmp_path: Path) -> None:
    """Verify validator rejects fabricated contact."""
    _, _, out = fixture(tmp_path)
    alter_payload(out, lambda d: d["contact_probability"].__setitem__((1, 0, 0), 1))
    with pytest.raises(ValueError, match="manufacture"):
        bundle.validate_bundle(out)


def test_object_nonuniform_scale_is_rejected() -> None:
    """Verify object nonuniform scale is rejected."""
    transform = np.diag([1.0, 2.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="uniform"):
        bundle.rigid_pose(transform)


def test_left_mirror_rotation_remains_proper_and_geometrically_consistent() -> None:
    """Verify left mirror rotation remains proper and geometrically consistent."""
    record = {"mano": {"global_orient": [0.4, -0.2, 0.1]}}
    point = np.array([0.1, 0.2, 0.3])
    mirror = np.diag([-1.0, 1.0, 1.0])
    right = Rotation.from_rotvec(record["mano"]["global_orient"]).as_matrix()
    left = bundle.wrist_rotation(record, "left")
    assert np.allclose(left @ (mirror @ point), mirror @ (right @ point))
    assert np.linalg.det(left) == pytest.approx(1)
