"""Geometric root rotations for a fixed local point and an image ray.

Ray parameters and rotations are hypotheses. Neither a source mask nor this
construction establishes measured depth, contact, joint feasibility or motion.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RayRootBranch:
    parameter: float
    rotation: np.ndarray


@dataclass(frozen=True)
class RayRootHypotheses:
    branches: tuple[RayRootBranch, ...]
    antiparallel_alignment_count: int


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def ray_bound_root_rotations(
    point_local,
    root_translation,
    ray_origin,
    ray_direction,
    reference_rotation,
    parameter_bounds=(0.0, math.inf),
):
    """Intersect the point's root-centered sphere with a declared ray.

    For each intersection, choose the least geodesic change from the supplied
    reference rotation that maps the local point onto it. This fixes the
    otherwise free twist by an explicit geometric convention, not by contact
    outcome. The ray direction is not normalized: its parameter retains the
    caller's convention. An antiparallel alignment has an undetermined minimum
    swing axis and is reported separately instead of claiming no solution.

    The result enumerates at most two sphere intersections under this twist
    convention. It does not enumerate all root rotations or certify a source
    orientation, original component bounds, other joints or a trajectory.
    """
    local = _vector(point_local, "point_local")
    root = _vector(root_translation, "root_translation")
    origin = _vector(ray_origin, "ray_origin")
    direction = _vector(ray_direction, "ray_direction")
    reference = np.asarray(reference_rotation, dtype=float)
    if (
        reference.shape != (3, 3)
        or not np.isfinite(reference).all()
        or not np.allclose(reference.T @ reference, np.eye(3), atol=1e-9, rtol=0)
        or not np.isclose(np.linalg.det(reference), 1.0, atol=1e-9, rtol=0)
    ):
        raise ValueError("reference_rotation must be a proper rotation")
    bounds = np.asarray(parameter_bounds, dtype=float)
    if (
        bounds.shape != (2,)
        or not np.isfinite(bounds[0])
        or bounds[0] < 0
        or math.isnan(bounds[1])
        or bounds[1] <= bounds[0]
    ):
        raise ValueError("Ray bounds must be ordered and nonnegative")
    radius = float(np.linalg.norm(local))
    a = float(np.dot(direction, direction))
    if radius <= 1e-12 or a <= 1e-24:
        raise ValueError("Local point and ray direction must be nonzero")
    relative = origin - root
    b = 2 * float(np.dot(direction, relative))
    c = float(np.dot(relative, relative)) - radius * radius
    discriminant = b * b - 4 * a * c
    tolerance = 1e-12 * max(b * b, abs(4 * a * c), a * radius * radius)
    if discriminant < -tolerance:
        return RayRootHypotheses((), 0)
    discriminant = max(0.0, discriminant)
    q = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
    parameters = [-b / (2 * a)] if q == 0 else [q / a, c / q]
    unique = []
    for parameter in sorted(parameters):
        if parameter < bounds[0] or parameter > bounds[1]:
            continue
        if not unique or abs(parameter - unique[-1]) > 1e-12:
            unique.append(parameter)
    initial = np.einsum("ij,j->i", reference, local, optimize=False) / radius
    branches = []
    antiparallel = 0
    for parameter in unique:
        desired = relative + parameter * direction
        target = desired / np.linalg.norm(desired)
        cosine = float(np.clip(np.dot(initial, target), -1, 1))
        if cosine < -1 + 1e-12:
            antiparallel += 1
            continue
        x, y, z = np.cross(initial, target)
        cross_matrix = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        swing = np.eye(3) + cross_matrix
        swing += np.einsum("ij,jk->ik", cross_matrix, cross_matrix, optimize=False) / (1 + cosine)
        rotation = np.einsum("ij,jk->ik", swing, reference, optimize=False)
        result = np.einsum("ij,j->i", rotation, local, optimize=False)
        if (
            not np.allclose(result, desired, atol=1e-9, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9, rtol=0)
        ):
            raise ValueError("Ray/root construction failed geometric closure")
        rotation.setflags(write=False)
        branches.append(RayRootBranch(float(parameter), rotation))
    return RayRootHypotheses(tuple(branches), antiparallel)
