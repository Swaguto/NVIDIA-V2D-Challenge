# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controls for whole-triangle coverage and conservative clearance certificates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "source/robotic_grounding")
)
from robotic_grounding.retarget.track3_collision_bounds import (
    axis_plane_envelope_upper,
    box_volume_capsule_cover,
    capsule_halfspace_upper,
    capsule_parent_expansion,
    enclosing_halfspaces,
    triangle_capsule_cover,
    verify_triangle_cover,
    world_halfspaces,
)
from robotic_grounding.retarget.track3_penetration import (
    Capsule,
    point_segment_distances,
)


def test_cover_retains_a_complete_large_triangle_above_target() -> None:
    """Verify cover retains a complete large triangle above target."""
    vertices = np.array([[0.0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
    faces = np.array([[0, 1, 2]])
    caps, assigned = triangle_capsule_cover(vertices, faces, "palm", 0.001)
    assert assigned.tolist() == [0] and len(caps) == 1
    assert caps[0].radius > 0.001 and caps[0].provenance["single_triangle_above_target"]
    assert (
        verify_triangle_cover(vertices, faces, caps, assigned)["triangles_checked"] == 1
    )


def test_vertex_coverage_by_different_capsules_does_not_cover_a_triangle() -> None:
    """Verify vertex coverage by different capsules does not cover a triangle."""
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    caps = [Capsule("palm", v, v, 0.01, {}) for v in vertices]
    with pytest.raises(ValueError, match="Unreferenced|outside"):
        verify_triangle_cover(vertices, np.array([[0, 1, 2]]), caps, np.array([0]))


def test_partition_is_deterministic_and_encloses_each_face() -> None:
    """Verify partition is deterministic and encloses each face."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    caps, a = triangle_capsule_cover(mesh.vertices, mesh.faces, "palm", 0.012)
    repeat, b = triangle_capsule_cover(mesh.vertices, mesh.faces, "palm", 0.012)
    assert len(caps) > 1 and np.array_equal(a, b)
    assert all(
        np.array_equal(x.a, y.a) and x.radius == y.radius
        for x, y in zip(caps, repeat, strict=True)
    )
    assert (
        verify_triangle_cover(mesh.vertices, mesh.faces, caps, a)[
            "maximum_assigned_vertex_outside_m"
        ]
        == 0
    )
    with pytest.raises(ValueError, match="nonempty"):
        triangle_capsule_cover(mesh.vertices, np.zeros((0, 3), dtype=int), "palm")


def test_halfspace_upper_covers_midshaft_and_is_conservative_outside() -> None:
    """Verify halfspace upper covers midshaft and is conservative outside."""
    box = trimesh.creation.box(extents=[1, 1, 1])
    n, d = enclosing_halfspaces(box.vertices, 0.0)
    upper = capsule_halfspace_upper(
        [(np.array([-2.0, 0, 0]), np.array([2.0, 0, 0]), 0.01)], n, d, 0.0
    )
    assert upper[0] == pytest.approx(
        0.51
    )  # Endpoints alone miss the interior crossing.
    p = np.array([0.6, 0.6, 0])
    upper = capsule_halfspace_upper([(p, p, 0.2)], n, d, 0.0)
    assert upper[0] == pytest.approx(0.1)
    assert upper[0] >= max(
        0.0, 0.2 - np.sqrt(2) * 0.1
    )  # Exact Euclidean value is smaller.


def test_world_planes_enclose_scaled_rotated_vertices_with_nonorthogonal_roundoff() -> (
    None
):
    """Verify world planes enclose scaled rotated vertices with nonorthogonal roundoff."""
    box = trimesh.creation.box(extents=[1, 1, 1])
    n, d = enclosing_halfspaces(box.vertices)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.4, 0.8]).as_matrix()
    pose[:3, :3][0] *= 1.0000001
    pose[:3, 3] = [3, 2, 1]
    wn, wd = world_halfspaces(n, d, pose, 2.0)
    vertices = box.vertices @ pose[:3, :3].T * 2 + pose[:3, 3]
    assert (vertices @ wn.T + wd).max() <= 0.0
    p = pose[:3, 3]
    upper = capsule_halfspace_upper([(p, p, 0.01)], wn, wd, 0.0)
    assert upper[0] == pytest.approx(1.01, abs=3e-7)


def test_parent_expansion_encloses_every_point_of_child_capsule() -> None:
    """Verify parent expansion encloses every point of child capsule."""
    parent = Capsule("palm", np.array([0.0, 0, 0]), np.array([0.1, 0, 0]), 0.02, {})
    child = Capsule(
        "palm", np.array([0.0, 0.02, 0]), np.array([0.08, 0.01, 0]), 0.012, {}
    )
    delta = capsule_parent_expansion(child, parent)
    rng = np.random.default_rng(17)
    directions = rng.normal(size=(200, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    centers = child.a + rng.random((200, 1)) * (child.b - child.a)
    points = centers + directions * child.radius
    assert np.all(
        point_segment_distances(points, parent.a, parent.b) <= parent.radius + delta
    )


def test_box_cells_cover_the_complete_interior_and_have_shared_boundaries() -> None:
    """Verify box cells cover the complete interior and have shared boundaries."""
    box = trimesh.creation.box(extents=[0.0867, 0.085, 0.026])
    rotation = Rotation.from_euler("xyz", [0.3, -0.1, 0.8]).as_matrix()
    vertices = box.vertices @ rotation.T + [0.02, 0.03, -0.01]
    caps, proof = box_volume_capsule_cover(vertices, "palm", 0.012)
    basis = np.array(proof["basis_rows"])
    lo = np.array(proof["lo_box_m"])
    hi = np.array(proof["hi_box_m"])
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    rng = np.random.default_rng(71)
    points = (lo + rng.random((1500, 3)) * (hi - lo)) @ basis
    covered = np.zeros(len(points), dtype=bool)
    for cap, cell in zip(caps, proof["cells"], strict=True):
        corners = (
            np.array(
                list(
                    __import__("itertools").product(
                        *zip(cell["lo_box_m"], cell["hi_box_m"], strict=True)
                    )
                )
            )
            @ basis
        )
        assert np.all(point_segment_distances(corners, cap.a, cap.b) <= cap.radius)
        covered |= point_segment_distances(points, cap.a, cap.b) <= cap.radius
    assert covered.all() and len(caps) == np.prod(proof["grid_counts"])


def test_axis_envelope_finds_diagonal_midshaft_and_matches_independent_lp() -> None:
    """Verify axis envelope finds diagonal midshaft and matches independent lp."""
    box = trimesh.creation.box(extents=[1, 1, 1])
    n, d = enclosing_halfspaces(box.vertices, 0.0)
    a, b = np.array([-2.0, -2, -2]), np.array([2.0, 2, 2])
    upper, _ = axis_plane_envelope_upper(a, b, n, d)
    assert upper == pytest.approx(0.5)
    rng = np.random.default_rng(31)
    for _ in range(24):
        a, b = rng.normal(size=(2, 3))
        upper, proof = axis_plane_envelope_upper(a, b, n, d)
        intercept = -n @ a - d
        slopes = -n @ (b - a)
        oracle = linprog(
            [0.0, -1.0],
            A_ub=np.column_stack((-slopes, np.ones(len(d)))),
            b_ub=intercept,
            bounds=[(0, 1), (None, None)],
            method="highs",
        )
        assert oracle.success
        expected = -oracle.fun
        assert expected - 1e-9 <= upper <= expected + 1.1e-7
        assert proof["lower"] <= upper + 1e-9
