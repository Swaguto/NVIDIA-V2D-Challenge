# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Geometric controls that detect empty models and midshaft penetration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "source/robotic_grounding")
)
from robotic_grounding.retarget.track3_penetration import (
    classify_checks,
    collision_capsules,
    fit_capsule,
    hand_hand_depth,
    hand_object_depth,
    point_segment_distances,
    segment_distance,
)


def test_segment_crossing_parallel_and_spheres() -> None:
    """Verify segment crossing parallel and spheres."""
    a = np.array([-1.0, 0, 0])
    b = -a
    assert segment_distance(
        a, b, np.array([0.0, -1, 0]), np.array([0.0, 1, 0])
    ) == pytest.approx(0)
    assert segment_distance(a, b, a + [0, 2, 0], b + [0, 2, 0]) == pytest.approx(2)
    assert segment_distance(a, a, b, b) == pytest.approx(2)
    assert segment_distance(np.zeros(3), np.zeros(3), a, b) == pytest.approx(0)


def test_hand_overlap_and_no_empty_pass() -> None:
    """Verify hand overlap and no empty pass."""
    zero = np.zeros(3)
    depth, pair = hand_hand_depth(
        [(zero, zero, 0.02)], [(zero + [0.03, 0, 0], zero + [0.03, 0, 0], 0.02)]
    )
    assert depth == pytest.approx(0.01) and pair == (0, 0)
    with pytest.raises(ValueError, match="nonempty"):
        hand_hand_depth([], [(zero, zero, 0.02)])


def test_midshaft_is_checked_and_world_scale_is_applied() -> None:
    """Verify midshaft is checked and world scale is applied."""
    box = trimesh.creation.box(extents=[1, 1, 1])
    a, b = np.array([-2.0, 0, 0]), np.array([2.0, 0, 0])
    # Both endpoints are outside. The middle lies 0.5 m inside the box.
    low, high = hand_object_depth([(a, b, 0.01)], box, np.eye(4), 1.0, 0.02)
    assert low == pytest.approx(0.51) and high == pytest.approx(0.52)
    pose = np.eye(4)
    pose[:3, 3] = [3, 2, 1]
    low, high = hand_object_depth(
        [(pose[:3, 3], pose[:3, 3], 0.01)], box, pose, 2.0, 0.02
    )
    assert low == high == pytest.approx(1.01)


def test_capsule_encloses_rotated_mesh_and_fk_wxyz() -> None:
    """Verify capsule encloses rotated mesh and fk wxyz."""
    points = trimesh.creation.box(extents=[0.02, 0.04, 0.1]).vertices
    points = points @ Rotation.from_euler("xyz", [0.2, 0.4, 0.8]).as_matrix().T + [
        0.03,
        -0.02,
        0.01,
    ]
    capsule = fit_capsule(points)
    assert (
        np.max(point_segment_distances(points, capsule.a, capsule.b)) <= capsule.radius
    )
    pose = np.array([1, 2, 3, 1, 0, 0, 0])
    a, b, r = capsule.transformed(pose)
    assert np.allclose(a, capsule.a + [1, 2, 3]) and np.allclose(
        b, capsule.b + [1, 2, 3]
    )


def test_missing_collision_model_and_lfs_pointer_fail(tmp_path: Path) -> None:
    """Verify missing collision model and lfs pointer fail."""
    urdf = tmp_path / "empty.urdf"
    urdf.write_text('<robot name="test"><link name="empty"/></robot>')
    with pytest.raises(ValueError, match="No collision"):
        collision_capsules(urdf, tmp_path)
    urdf.write_text(
        '<robot name="test"><link name="palm"><collision><geometry><mesh filename="package://hand.obj"/></geometry></collision></link></robot>'
    )
    (tmp_path / "hand.obj").write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="LFS pointer"):
        collision_capsules(urdf, tmp_path)


def test_unknowns_exclusions_and_supported_single_hand_rejection() -> None:
    """Verify unknowns exclusions and supported single hand rejection."""
    valid = np.array([[True, True], [True, False], [True, True], [True, True]])
    hh = np.array([0.0, np.nan, 0.021, 0.0])
    objects = np.zeros((4, 2, 1))
    objects[0] = np.nan  # Explicitly excluded; cannot pass full coverage.
    objects[1, 0] = 0.021  # Known single-hand violation still rejects the frame.
    objects[1, 1] = np.nan
    _, _, rejected, _, proxy_pass, full = classify_checks(valid, hh, objects, 0.02)
    assert rejected.tolist() == [False, True, True, False]
    assert proxy_pass.tolist() == [True, False, False, True]
    assert full.tolist() == [False, False, True, True]
    assert (full & ~rejected).tolist() == [False, False, False, True]
