# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import trimesh
from robotic_grounding.retarget.track3_collision_voxels import (
    raster_solid,
    solid_depth_bounds,
    triangle_cells,
)
from scipy.ndimage import distance_transform_edt


def test_triangle_raster_covers_face_interior_and_rejects_box_corner() -> None:
    """Verify triangle raster covers face interior and rejects box corner."""
    tri = np.array([[0, 0, 0], [0.012, 0, 0], [0, 0.012, 0]], float)
    p = 0.001
    origin = np.full(3, -0.003)
    indices = triangle_cells(tri, origin, p)
    keys = set(map(tuple, indices))
    rng = np.random.default_rng(1)
    b = rng.dirichlet([1, 1, 1], 2000)
    points = b @ tri
    assert all(tuple(ix) in keys for ix in np.rint((points - origin) / p).astype(int))
    assert (
        tuple(np.rint((np.array([0.011, 0.011, 0]) - origin) / p).astype(int))
        not in keys
    )


def test_depth_bounds_against_independent_empty_cube_distance() -> None:
    """Verify depth bounds against independent empty cube distance."""
    rng = np.random.default_rng(8)
    solid = np.zeros((8, 8, 8), bool)
    solid[2:6, 2:6, 2:6] = rng.random((4, 4, 4)) > 0.3

    edt = distance_transform_edt(solid)
    origin = np.zeros(3)
    p = 1.0
    cells = np.argwhere(solid)
    empty = np.argwhere(~solid)
    points = cells[rng.integers(len(cells), size=200)] + rng.uniform(
        -0.49, 0.49, (200, 3)
    )
    low, up, inside = solid_depth_bounds(points, solid, edt, origin, p)
    assert inside.all()
    exact = np.array(
        [
            np.linalg.norm(np.maximum(np.abs(empty - x) - 0.5, 0), axis=1).min()
            for x in points
        ]
    )
    assert np.all(low <= exact + 1e-12) and np.all(exact <= up + 1e-12)
    _, up, inside = solid_depth_bounds([[20, 0, 0], [0, 0, 0]], solid, edt, origin, p)
    assert not inside.any() and np.array_equal(up, [0, 0])


def test_solid_raster_fills_box_material_and_preserves_torus_air() -> None:
    """Verify solid raster fills box material and preserves torus air."""
    box = trimesh.creation.box(extents=[0.012, 0.014, 0.016])
    origin, surface, solid, depth = raster_solid(box.vertices, box.faces, 0.001)
    assert solid.sum() > surface.sum()
    assert solid_depth_bounds([[0, 0, 0]], solid, depth, origin, 0.001)[2].all()
    ring = trimesh.creation.torus(
        major_radius=0.025, minor_radius=0.003, major_sections=24, minor_sections=12
    )
    origin, _, solid, depth = raster_solid(ring.vertices, ring.faces, 0.001)
    assert not solid_depth_bounds(
        [[0, 0, 0], [0, 0, 0.01]], solid, depth, origin, 0.001
    )[2].any()
    assert solid_depth_bounds([[0.025, 0, 0]], solid, depth, origin, 0.001)[2].all()


def test_midshaft_material_is_found_when_both_endpoints_are_outside() -> None:
    """Verify midshaft material is found when both endpoints are outside."""
    box = trimesh.creation.box(extents=[0.06, 0.06, 0.06])
    pitch = 0.001
    origin, _, solid, depth = raster_solid(box.vertices, box.faces, pitch)
    axis = np.c_[np.linspace(-0.1, 0.1, 101), np.zeros(101), np.zeros(101)]
    low, up, inside = solid_depth_bounds(axis, solid, depth, origin, pitch)
    assert not inside[0] and not inside[-1] and inside[50]
    assert low[50] > 0.03 and up[50] > 0.03
