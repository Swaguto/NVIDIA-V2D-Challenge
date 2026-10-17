"""Point-contact load capacity under measured normal-force budgets.

This is an offline linear friction-pyramid diagnostic, not a force-closure or
dynamic-success certificate. Supports are excluded. Motor feasibility, hand
gravity, contact compliance and torsional friction are not modeled.
"""

import numpy as np
from scipy.optimize import linprog


def capacity(points, inward_normals, normal_budgets, com, load, friction, edges=8):
    points = np.asarray(points, float).reshape(-1, 3)
    normals = np.asarray(inward_normals, float).reshape(-1, 3)
    budgets = np.asarray(normal_budgets, float)
    com, load = np.asarray(com, float), np.asarray(load, float)
    if (
        normals.shape != points.shape
        or budgets.shape != (len(points),)
        or com.shape != (3,)
        or load.shape != (6,)
        or edges < 4
        or not np.isfinite(friction)
        or friction < 0
        or not all(np.isfinite(x).all() for x in (points, normals, budgets, com, load))
        or (budgets < 0).any()
        or np.linalg.norm(load) < 1e-8
    ):
        raise ValueError("Need finite contact geometry, nonnegative budgets and a load")
    if not len(points):
        return dict(load_multiplier=0.0, success=True, contacts=0)
    lengths = np.linalg.norm(normals, axis=1)
    if (lengths < 0.99).any() or (lengths > 1.01).any():
        raise ValueError("Contact normals must be unit vectors")
    normals = normals / lengths[:, None]
    wrench = []
    for point, normal in zip(points, normals, strict=True):
        helper = np.eye(3)[np.argmin(abs(normal))]
        tangent = np.cross(normal, helper)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        for angle in np.arange(edges) * (2 * np.pi / edges):
            force = normal + friction * (
                np.cos(angle) * tangent + np.sin(angle) * bitangent
            )
            wrench.append(np.r_[force, np.cross(point - com, force)])
    wrench = np.asarray(wrench).T
    # Scale moment equations by a declared 10 cm length for conditioning only.
    scale = np.r_[np.ones(3), np.full(3, 10.0)]
    equal = np.c_[wrench, -load] * scale[:, None]
    bounds = np.zeros((len(points), len(wrench.T) + 1))
    for i in range(len(points)):
        bounds[i, i * edges : (i + 1) * edges] = 1
    objective = np.zeros(equal.shape[1])
    objective[-1] = -1
    solved = linprog(
        objective,
        A_ub=bounds,
        b_ub=budgets,
        A_eq=equal,
        b_eq=np.zeros(6),
        bounds=(0, None),
        method="highs",
    )
    if not solved.success:
        raise RuntimeError("Contact load diagnostic failed: " + solved.message)
    return dict(
        load_multiplier=float(solved.x[-1]),
        success=True,
        contacts=len(points),
        chosen_normal_forces_N=solved.x[:-1].reshape(-1, edges).sum(1).tolist(),
    )
