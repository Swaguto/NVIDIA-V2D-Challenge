# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit closed-cell derived object solids and conservative depth bounds.

Occupied cells denote a union of closed axis-aligned cubes. Surface raster
uses triangle/box separating-axis tests, then a documented one-cell material
envelope and enclosed-void fill. No incomplete-mesh inside test is needed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt


def triangle_cells(triangle: ArrayLike, origin: ArrayLike, pitch: float) -> np.ndarray:
    """Rasterize every closed cube intersecting a triangle using separating axes."""
    triangle = np.asarray(triangle, dtype=np.float64)
    low = np.ceil((triangle.min(axis=0) - origin) / pitch - 0.5 - 1e-9).astype(int)
    high = np.floor((triangle.max(axis=0) - origin) / pitch + 0.5 + 1e-9).astype(int)
    ix = np.stack(
        np.meshgrid(
            *[np.arange(a, b + 1) for a, b in zip(low, high, strict=True)],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    centers = origin + ix * pitch
    edges = np.roll(triangle, -1, axis=0) - triangle
    axes = np.vstack(
        [
            np.cross(edges[:, None, :], np.eye(3)[None, :, :]).reshape(-1, 3),
            np.cross(edges[0], edges[1]),
        ]
    )
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-30] / lengths[lengths > 1e-30, None]
    # Box coordinate axes were already tested by the candidate index ranges.
    projected = triangle @ axes.T
    middle = centers @ axes.T
    radius = 0.5 * pitch * np.abs(axes).sum(axis=1)
    keep = np.all(
        (middle + radius >= projected.min(axis=0) - 1e-12)
        & (middle - radius <= projected.max(axis=0) + 1e-12),
        axis=1,
    )
    return ix[keep]


def raster_solid(
    vertices: ArrayLike,
    faces: ArrayLike,
    pitch: float = 0.001,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the declared material-cell union and empty-centre distance witnesses."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if pitch <= 0 or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Invalid geometry/pitch")
    origin = np.floor(vertices.min(axis=0) / pitch) * pitch - 3 * pitch
    shape = np.ceil((vertices.max(axis=0) - origin) / pitch).astype(int) + 4
    surface = np.zeros(tuple(shape), dtype=bool)
    for i, face in enumerate(faces):
        cells = triangle_cells(vertices[face], origin, pitch)
        assert len(cells) > 0 and np.all(cells >= 0) and np.all(cells < shape)
        surface[tuple(cells.T)] = True
        if progress and (i + 1) % 10000 == 0:
            progress(i + 1, len(faces))
    envelope = binary_dilation(
        surface, structure=np.ones((3, 3, 3), dtype=bool), iterations=1
    )
    solid = binary_fill_holes(envelope)
    assert not (
        solid[[0, -1], :, :].any()
        or solid[:, [0, -1], :].any()
        or solid[:, :, [0, -1]].any()
    )
    # Float64 distances to an empty cell centre; centres of empty cells
    # are verified outside the cube union. These are boundary witnesses.
    depth = distance_transform_edt(solid, sampling=pitch)
    return origin, surface, solid, depth


def solid_depth_bounds(
    points: ArrayLike,
    solid: np.ndarray,
    empty_center_distance: np.ndarray,
    origin: ArrayLike,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bound signed depth, positive inside; outside upper is safely zero."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Invalid query points")
    index = np.rint((points - origin) / pitch).astype(int)
    in_grid = np.all((index >= 0) & (index < np.array(solid.shape)), axis=1)
    inside = np.zeros(len(points), dtype=bool)
    rows = np.flatnonzero(in_grid)
    inside[rows] = solid[tuple(index[rows].T)]
    rows = np.flatnonzero(inside)
    upper = np.zeros(len(points))
    lower = np.full(len(points), -np.inf)
    if len(rows):
        delta = np.linalg.norm(points[rows] - (origin + index[rows] * pitch), axis=1)
        distance = empty_center_distance[tuple(index[rows].T)]
        upper[rows] = distance + delta + 1e-12
        # Every empty cube is within half its diagonal of its centre.
        lower[rows] = np.maximum(0.0, distance - delta - np.sqrt(3) * pitch / 2 - 1e-12)
    return lower, upper, inside
