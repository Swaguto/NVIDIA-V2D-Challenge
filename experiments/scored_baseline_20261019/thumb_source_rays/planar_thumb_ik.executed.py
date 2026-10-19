"""Analytic point IK for an explicitly bound Y/Z/Z thumb chain.

This enumerates finite kinematic branches for a requested point. It does not
establish source depth, collision clearance, a motor-feasible grasp or dynamics.
An axial singularity is reported separately: an empty finite branch list there
is not an infeasibility certificate.
"""

from dataclasses import dataclass
import math

import numpy as np


def _finite_vector(value, size, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return vector


@dataclass(frozen=True)
class PlanarThumbChain:
    """Origin, planar offset and two positive lengths, expressed in root axes.

    The first axis is +Y at ``origin``. After that rotation, ``offset_xy``
    locates the first +Z hinge. Each planar link extends along local +Y;
    the second +Z hinge lies at the end of the proximal link. A point at the
    end of the distal link is the IK witness, not an observed contact site.
    """

    origin: tuple[float, float, float]
    offset_xy: tuple[float, float]
    proximal_length: float
    distal_length: float

    def __post_init__(self):
        _finite_vector(self.origin, 3, "origin")
        _finite_vector(self.offset_xy, 2, "offset_xy")
        for length in [self.proximal_length, self.distal_length]:
            if not np.isfinite(length) or length <= 0:
                raise ValueError("Link lengths must be finite and positive")


@dataclass(frozen=True)
class PointIKBranches:
    joints_rad: tuple[tuple[float, float, float], ...]
    axial_singular: bool


def thumb_point(chain: PlanarThumbChain, joints_rad):
    a, b, c = _finite_vector(joints_rad, 3, "joints_rad")
    x = chain.offset_xy[0] - chain.proximal_length * math.sin(b)
    x -= chain.distal_length * math.sin(b + c)
    y = chain.offset_xy[1] + chain.proximal_length * math.cos(b)
    y += chain.distal_length * math.cos(b + c)
    return np.asarray(chain.origin) + [math.cos(a) * x, y, -math.sin(a) * x]


def _equivalent_angles(angle, lower, upper):
    period = 2 * math.pi
    first = math.ceil((lower - angle) / period)
    last = math.floor((upper - angle) / period)
    return [angle + period * k for k in range(first, last + 1)]


def thumb_point_branches(chain: PlanarThumbChain, point_root, joint_bounds):
    """Enumerate all nonsingular point branches inside the declared bounds.

    ``joint_bounds`` is finite (3,2), ordered, and spans at most one turn per
    joint. Neither limits nor the desired point are optimized or enlarged.
    Tangent branches are deduplicated. Out-of-workspace points return no branch;
    points on the first hinge axis set ``axial_singular`` instead of claiming
    infeasibility. The caller must independently bind this chain to its URDF.
    """
    point = _finite_vector(point_root, 3, "point_root")
    bounds = np.asarray(joint_bounds, dtype=float)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
        raise ValueError("joint_bounds must be finite with shape (3,2)")
    widths = bounds[:, 1] - bounds[:, 0]
    if (widths < 0).any() or (widths > 2 * math.pi).any():
        raise ValueError("Each joint interval must be ordered and at most one turn")
    relative = point - np.asarray(chain.origin)
    radius = math.hypot(relative[0], relative[2])
    if radius <= 1e-12:
        return PointIKBranches((), True)
    result = []
    for sign in [1, -1]:
        planar_x = sign * radius
        a = math.atan2(-relative[2] / planar_x, relative[0] / planar_x)
        u = planar_x - chain.offset_xy[0]
        v = relative[1] - chain.offset_xy[1]
        l1, l2 = chain.proximal_length, chain.distal_length
        cosine = (u * u + v * v - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        if cosine < -1 - 1e-12 or cosine > 1 + 1e-12:
            continue
        elbow = math.acos(max(-1., min(1., cosine)))
        for c in [elbow, -elbow]:
            b = math.atan2(v, u) - math.atan2(l1 + l2 * math.cos(c), -l2 * math.sin(c))
            aa = _equivalent_angles(a, *bounds[0])
            bb = _equivalent_angles(b, *bounds[1])
            cc = _equivalent_angles(c, *bounds[2])
            for av in aa:
                for bv in bb:
                    for cv in cc:
                        angles = (av, bv, cv)
                        if np.linalg.norm(thumb_point(chain, angles) - point) > 1e-10:
                            raise ValueError("Analytic branch failed independent forward closure")
                        if not any(np.max(np.abs(np.asarray(old) - angles)) < 1e-10 for old in result):
                            result.append(angles)
    return PointIKBranches(tuple(sorted(result)), False)
