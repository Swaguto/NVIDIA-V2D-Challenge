# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triangle-enclosing capsule covers and conservative solid-clearance bounds.

An enclosing polytope can certify a small penetration upper bound even when
the original concave mesh has no trustworthy signed distance. A large upper
bound is inconclusive; it is never a measured rejection.
"""
from __future__ import annotations

import itertools
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull

from .track3_penetration import Capsule, fit_capsule, point_segment_distances


def triangle_capsule_cover(
    vertices: ArrayLike, faces: ArrayLike, link: str, target_radius_m: float = 0.012
) -> tuple[list[Capsule], np.ndarray]:
    """Partition whole triangles; each triangle has one enclosing capsule.

    A single large triangle may exceed the requested target. It is retained
    at its required enclosing radius, never truncated or omitted.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0
        or faces.max() >= len(vertices)
        or not np.isfinite(target_radius_m)
        or target_radius_m <= 0
    ):
        raise ValueError(
            "Require finite vertices, nonempty valid triangles and a positive target radius"
        )
    centroids = vertices[faces].mean(axis=1)
    pending = [np.arange(len(faces))]
    capsules: list[Capsule] = []
    assignment = np.full(len(faces), -1, dtype=np.int64)
    while pending:
        ids = pending.pop()
        points = vertices[np.unique(faces[ids])]
        # The existing fitter requires four rows; repeating a vertex changes
        # no geometry and permits a retained single-triangle leaf.
        fitted_points = points if len(points) >= 4 else np.vstack((points, points[:1]))
        capsule = fit_capsule(fitted_points, link)
        if capsule.radius > target_radius_m and len(ids) > 1:
            centers = centroids[ids]
            axis = int(np.argmax(np.ptp(centers, axis=0)))
            ordered = ids[np.argsort(centers[:, axis], kind="stable")]
            middle = len(ordered) // 2
            pending.extend((ordered[middle:], ordered[:middle]))
            continue
        assignment[ids] = len(capsules)
        capsules.append(
            Capsule(
                link,
                capsule.a,
                capsule.b,
                capsule.radius,
                {
                    "triangles": len(ids),
                    "target_radius_m": target_radius_m,
                    "single_triangle_above_target": bool(
                        capsule.radius > target_radius_m
                    ),
                    "fit": "each assigned complete triangle is enclosed in this convex capsule",
                },
            )
        )
    verify_triangle_cover(vertices, faces, capsules, assignment)
    return capsules, assignment


def verify_triangle_cover(
    vertices: ArrayLike,
    faces: ArrayLike,
    capsules: Sequence[Any],
    assignment: ArrayLike,
) -> dict[str, Any]:
    """Verify whole-triangle enclosure by the assigned convex capsules."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    assignment = np.asarray(assignment)
    if (
        not capsules
        or assignment.shape != (len(faces),)
        or not len(faces)
        or not np.issubdtype(assignment.dtype, np.integer)
        or assignment.min() < 0
        or assignment.max() >= len(capsules)
    ):
        raise ValueError(
            "Every complete triangle must have exactly one valid enclosing capsule"
        )
    maximum_outside = -np.inf
    for i, capsule in enumerate(capsules):
        selected = faces[assignment == i]
        if not len(selected):
            raise ValueError("Unreferenced capsule in triangle cover")
        outside = (
            point_segment_distances(vertices[selected.ravel()], capsule.a, capsule.b)
            - capsule.radius
        )
        maximum_outside = max(maximum_outside, float(outside.max()))
    if maximum_outside > 1e-10:
        raise ValueError("A triangle vertex lies outside its assigned convex capsule")
    # Convexity then encloses every point of each triangle, not only vertices
    # scattered among different members of a nonconvex union.
    return {
        "triangles_checked": len(faces),
        "capsules": len(capsules),
        "maximum_assigned_vertex_outside_m": max(0.0, maximum_outside),
    }


def enclosing_halfspaces(
    vertices: ArrayLike, margin_m: float = 1e-7
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized convex halfspaces n.x+d<=0, expanded to enclose all vertices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) < 4
        or not np.isfinite(vertices).all()
        or margin_m < 0
        or not np.isfinite(margin_m)
    ):
        raise ValueError(
            "Require finite volume vertices and a nonnegative numerical margin"
        )
    equations = ConvexHull(vertices).equations
    length = np.linalg.norm(equations[:, :3], axis=1)
    normals, offsets = equations[:, :3] / length[:, None], equations[:, 3] / length
    violation = np.zeros(len(offsets))
    for start in range(0, len(vertices), 4096):
        violation = np.maximum(
            violation,
            (vertices[start : start + 4096] @ normals.T + offsets).max(axis=0),
        )
    offsets = offsets - violation - margin_m
    return normals, offsets


def world_halfspaces(
    normals: ArrayLike, offsets: ArrayLike, pose: ArrayLike, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Transform plane normals using the inverse, including float32 SE(3) error."""
    pose = np.asarray(pose, dtype=np.float64)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Require a finite object transform and positive scale")
    world_normals = np.asarray(normals) @ np.linalg.inv(pose[:3, :3] * scale)
    world_offsets = np.asarray(offsets) - world_normals @ pose[:3, 3]
    length = np.linalg.norm(world_normals, axis=1)
    return world_normals / length[:, None], world_offsets / length


def capsule_halfspace_upper(
    capsules: Sequence[Any],
    normals: ArrayLike,
    offsets: ArrayLike,
    numerical_margin_m: float = 1e-7,
) -> np.ndarray:
    """Upper-bound signed penetration of every point in each complete capsule.

    Each normalized supporting plane upper-bounds signed distance to the
    enclosing solid. Maximize that plane over the segment and its radius,
    then take the minimum among planes. This also covers the midshaft.
    """
    if not capsules:
        raise ValueError("An empty capsule model cannot certify clearance")
    normals, offsets = np.asarray(normals), np.asarray(offsets)
    if (
        normals.ndim != 2
        or normals.shape[1] != 3
        or offsets.shape != (len(normals),)
        or not len(normals)
        or not np.isfinite(normals).all()
        or not np.isfinite(offsets).all()
        or not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-10, rtol=0)
    ):
        raise ValueError("Require finite normalized supporting planes")
    a, b, radii = (np.asarray(v) for v in zip(*capsules, strict=True))
    if (
        a.shape != b.shape
        or a.shape != (len(capsules), 3)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.isfinite(radii).all()
        or np.any(radii <= 0)
        or numerical_margin_m < 0
        or not np.isfinite(numerical_margin_m)
    ):
        raise ValueError(
            "Require finite capsule axes, positive radii and nonnegative margin"
        )
    plane_maximum = np.maximum(-a @ normals.T - offsets, -b @ normals.T - offsets)
    signed_upper = plane_maximum.min(axis=1) + radii + numerical_margin_m
    return np.maximum(0.0, signed_upper)


def capsule_parent_expansion(capsule: Capsule, parent: Capsule) -> float:
    """Radius increment that provably encloses a child capsule in its parent."""
    endpoints = np.array([capsule.a, capsule.b])
    distance = point_segment_distances(endpoints, parent.a, parent.b).max()
    return max(0.0, float(distance + capsule.radius - parent.radius)) + 1e-9


def box_volume_capsule_cover(
    vertices: ArrayLike, link: str, target_radius_m: float = 0.012
) -> tuple[list[Capsule], dict[str, Any]]:
    """Cover an entire oriented bounding box by rectangular cells in capsules.

    Each cell's eight corners lie in one convex capsule. The cells tile the
    complete box without gaps, covering all mesh facets and all possible
    interiors within their convex hull, including nonwatertight source meshes.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) < 4
        or not np.isfinite(vertices).all()
        or target_radius_m <= 0
        or not np.isfinite(target_radius_m)
    ):
        raise ValueError("Require finite mesh vertices and a positive radius")
    _, _, pca = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)
    candidates = []
    for basis in (np.eye(3), pca):
        coords = vertices @ basis.T
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        for axis in range(3):
            other = [j for j in range(3) if j != axis]
            counts = [
                max(1, int(np.ceil((hi[j] - lo[j]) / (np.sqrt(2) * target_radius_m))))
                for j in other
            ]
            widths = [(hi[j] - lo[j]) / n for j, n in zip(other, counts, strict=True)]
            radius = float(np.linalg.norm(widths) / 2) + 1e-9
            count = int(np.prod(counts))
            volume = count * (
                np.pi * radius**2 * (hi[axis] - lo[axis]) + 4 * np.pi * radius**3 / 3
            )
            candidates.append(
                (volume, basis.copy(), lo, hi, axis, other, counts, radius)
            )
    _, basis, lo, hi, axis, other, counts, radius = min(candidates, key=lambda x: x[0])
    edges = [
        np.linspace(lo[j], hi[j], count + 1)
        for j, count in zip(other, counts, strict=True)
    ]
    capsules, cells = [], []
    for i, j in itertools.product(range(counts[0]), range(counts[1])):
        cell_lo, cell_hi = lo.copy(), hi.copy()
        for k, idx in enumerate((i, j)):
            cell_lo[other[k]], cell_hi[other[k]] = edges[k][idx], edges[k][idx + 1]
        a = (cell_lo + cell_hi) / 2
        b = a.copy()
        a[axis], b[axis] = lo[axis], hi[axis]
        capsules.append(
            Capsule(
                link,
                a @ basis,
                b @ basis,
                radius,
                {
                    "fit": "convex capsule encloses a complete cell in a gap-free bounding-volume tiling"
                },
            )
        )
        cells.append(
            {
                "grid_index": [i, j],
                "lo_box_m": cell_lo.tolist(),
                "hi_box_m": cell_hi.tolist(),
            }
        )
    proof = {
        "basis_rows": basis.tolist(),
        "lo_box_m": lo.tolist(),
        "hi_box_m": hi.tolist(),
        "full_axis": axis,
        "partition_axes": other,
        "grid_counts": counts,
        "cells": cells,
        "target_radius_m": target_radius_m,
        "volume_enclosure": True,
    }
    return capsules, proof


def axis_plane_envelope_upper(  # noqa: PLR0917 - Keep the complete axis/plane API together.
    a: ArrayLike,
    b: ArrayLike,
    normals: ArrayLike,
    offsets: ArrayLike,
    tolerance_m: float = 1e-7,
    max_iterations: int = 64,
) -> tuple[float, dict[str, Any]]:
    """Upper-bound max_t min_j supporting-plane margins on a closed axis.

    Every returned value is a valid envelope bound, including early exit.
    Active positive/negative-slope planes refine the bound at their crossing;
    a large remaining gap makes the result more conservative, never unsafe.
    """
    a, b = np.asarray(a), np.asarray(b)
    intercept = -np.asarray(normals) @ a - np.asarray(offsets)
    slopes = -np.asarray(normals) @ (b - a)
    left, right = int(np.argmin(intercept)), int(np.argmin(intercept + slopes))
    lower = max(float(intercept[left]), float(intercept[right] + slopes[right]))
    if slopes[left] <= 0:
        return float(intercept[left]), {
            "planes": [left],
            "lower": lower,
            "iterations": 0,
        }
    if slopes[right] >= 0:
        return float(intercept[right] + slopes[right]), {
            "planes": [right],
            "lower": lower,
            "iterations": 0,
        }
    upper = np.inf
    for iteration in range(max_iterations):
        evaluated_pair = [left, right]
        t = float(
            np.clip(
                (intercept[right] - intercept[left]) / (slopes[left] - slopes[right]),
                0,
                1,
            )
        )

        def pair(u: float, left_index: int = left, right_index: int = right) -> float:
            """Evaluate the two supporting planes selected for this iteration."""
            return min(
                intercept[left_index] + slopes[left_index] * u,
                intercept[right_index] + slopes[right_index] * u,
            )

        upper = float(max(pair(0.0), pair(1.0), pair(t)))
        values = intercept + slopes * t
        active = int(np.argmin(values))
        lower = max(lower, float(values[active]))
        if upper - lower <= tolerance_m:
            return upper, {
                "planes": [left, right],
                "lower": lower,
                "iterations": iteration + 1,
            }
        if slopes[active] == 0:
            return min(upper, float(intercept[active])), {
                "planes": [active],
                "lower": lower,
                "iterations": iteration + 1,
            }
        if slopes[active] > 0:
            left = active
        else:
            right = active
    # Keep the pair associated with the returned bound, rather than the
    # unexamined update at the final iteration.
    return upper, {
        "planes": evaluated_pair,
        "lower": lower,
        "iterations": max_iterations,
        "conservative_early_exit": True,
    }


def capsule_axis_envelope_upper(
    capsules: Sequence[Any],
    normals: ArrayLike,
    offsets: ArrayLike,
    numerical_margin_m: float = 1e-7,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    # Reuse validation without treating its looser endpoint-plane result as
    # the tighter axis maximum used below.
    """Bound complete capsule axes using a tighter supporting-plane envelope."""
    capsule_halfspace_upper(capsules, normals, offsets, numerical_margin_m)
    upper = []
    witnesses = []
    for a, b, radius in capsules:
        value, proof = axis_plane_envelope_upper(a, b, normals, offsets)
        upper.append(max(0.0, value + radius + numerical_margin_m))
        witnesses.append(proof)
    return np.array(upper), witnesses
