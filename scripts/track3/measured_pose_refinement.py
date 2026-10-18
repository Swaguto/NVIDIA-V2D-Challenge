"""Experimental translation refinement from measured foreground and finite support.

Rotation, scale and the raw body origin remain fixed. A support constraint is
applicable only when the complete projected body lies within the measured convex
footprint. A candidate must improve depth fit without worsening its silhouette.
This is an inference experiment, not a contact or hidden-pose certificate.
"""

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, cKDTree
from select_pose_hypothesis import observed_fit


def support_constraints(vertices, pose, plane, footprint, clearance=0.002):
    """Return linear translation constraints only for a fully supported footprint.

    The footprint is finite. An overhanging or off-table body receives no plane
    constraint, including when its minimum height is negative.
    """
    plane, footprint = np.asarray(plane, float), np.asarray(footprint, float)
    if plane.shape != (4,) or footprint.ndim != 2 or footprint.shape[1] != 3:
        raise ValueError("Need a finite plane and measured polygon")
    if (
        not np.isfinite(plane).all()
        or not np.isfinite(footprint).all()
        or not np.isclose(np.linalg.norm(plane[:3]), 1, atol=1e-6)
        or len(footprint) < 3
        or np.max(abs(np.einsum("ni,i->n", footprint, plane[:3]) + plane[3])) > 0.001
    ):
        raise ValueError("Support footprint must lie on its measured unit plane")
    normal = plane[:3]
    axis = np.eye(3)[np.argmin(abs(normal))]
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    basis = np.stack([u, np.cross(normal, u)], axis=1)
    hull = ConvexHull(np.einsum("ni,ij->nj", footprint, basis))
    camera = np.einsum("ni,ji->nj", vertices, pose[:3, :3]) + pose[:3, 3]
    # Linear extrema over mesh vertices also bound every projected triangle.
    side = np.einsum(
        "ni,ki->nk", np.einsum("ni,ij->nj", camera, basis), hull.equations[:, :2]
    )
    side = side.max(axis=0) + hull.equations[:, 2]
    if np.max(side) > -0.005:
        return None
    footprint_normals = hull.equations[:, :2] @ basis.T
    height = np.min(np.einsum("ni,i->n", camera, normal) + plane[3])
    # A @ delta >= b maintains height and a 1 mm inset in the original footprint.
    return (
        np.vstack([normal, -footprint_normals]),
        np.r_[clearance - height, side + 0.001],
    )


def refine_translation(
    vertices, faces, pose, depth, valid, mask, K, plane=None, footprint=None
):
    vertices, faces, pose = np.asarray(vertices), np.asarray(faces), np.asarray(pose)
    depth, valid, mask, K = map(np.asarray, (depth, valid, mask, K))
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1])
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)
        or depth.ndim != 2
        or valid.shape != depth.shape
        or mask.shape != depth.shape
        or K.shape != (3, 3)
        or not np.isfinite(K).all()
        or K[0, 0] <= 0
        or K[1, 1] <= 0
    ):
        raise ValueError(
            "Need finite rigid body geometry and calibrated measured pixels"
        )
    valid, mask = valid.astype(bool), mask.astype(bool)
    tree = cKDTree(vertices)
    before = observed_fit(vertices, faces, tree, pose, depth, valid, mask, K)
    report = {
        "accepted": False,
        "before": before,
        "support_constraint_applied": False,
        "body_rotation_changed": False,
        "body_scale_changed": False,
        "missing_observations_filled": False,
    }
    if before["measured_pixels"] < 100:
        return pose.copy(), dict(report, reason="insufficient measured foreground")
    y, x = np.nonzero(valid & mask & np.isfinite(depth) & (depth > 0.1) & (depth < 2.5))
    ix = np.linspace(0, len(x) - 1, min(2048, len(x)), dtype=int)
    x, y = x[ix], y[ix]
    z = depth[y, x]
    points = np.c_[(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z]
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    sampled = vertices[
        np.linspace(0, len(vertices) - 1, min(2048, len(vertices)), dtype=int)
    ]
    rotated = np.einsum("ni,ji->nj", sampled, pose[:3, :3])
    constraints = []
    start = np.zeros(3)
    if (plane is None) != (footprint is None):
        raise ValueError("Plane and finite footprint must be supplied together")
    support = (
        None if plane is None else support_constraints(vertices, pose, plane, footprint)
    )
    if support is not None:
        A, b = support
        constraints.append({"type": "ineq", "fun": lambda delta: A @ delta - b})
        start = A[0] * max(0, b[0])
        report["support_constraint_applied"] = True
    # Bound displacement independently of numerical optimizer behavior.
    constraints.append({"type": "ineq", "fun": lambda d: 0.08**2 - d @ d})

    def objective(delta):
        t = pose[:3, 3] + delta
        local = np.einsum("ni,ij->nj", points - t, pose[:3, :3])
        camera = rotated + t
        if not np.isfinite(local).all() or (camera[:, 2] <= 0.05).any():
            return 1000.0
        distances = np.sort(tree.query(local, workers=1)[0])[: int(0.8 * len(points))]
        robust = np.where(
            distances < 0.01, 0.5 * distances**2, 0.01 * (distances - 0.005)
        ).mean()
        u = K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2]
        v = K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2]
        x = np.clip(np.rint(u).astype(int), 0, mask.shape[1] - 1)
        y = np.clip(np.rint(v).astype(int), 0, mask.shape[0] - 1)
        over = np.maximum(0, outside[y, x] - 5) * camera[:, 2] / K[0, 0]
        over[(u < 0) | (u >= mask.shape[1]) | (v < 0) | (v >= mask.shape[0])] = 0.1
        return float(
            robust
            + 0.5 * np.minimum(over, 0.1).dot(np.minimum(over, 0.1)) / len(over)
            + (0.002 * np.linalg.norm(delta) / 0.05) ** 2
        )

    fit = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(-0.08, 0.08)] * 3,
        constraints=constraints,
        options={"maxiter": 100, "ftol": 1e-11},
    )
    candidate = pose.copy()
    candidate[:3, 3] += fit.x
    after = observed_fit(vertices, faces, tree, candidate, depth, valid, mask, K)
    report.update(
        after=after,
        optimizer_success=bool(fit.success),
        translation_delta_m=fit.x.tolist(),
        function_evaluations=int(fit.nfev),
    )
    if (
        not fit.success
        or not np.isfinite(fit.x).all()
        or np.linalg.norm(fit.x) > 0.08 + 1e-7
        or (support is not None and np.min(A @ fit.x - b) < -1e-7)
        or after["quality"] <= before["quality"]
        or after.get("silhouette_iou", -1) < before.get("silhouette_iou", 0)
        or after.get("median_measured_vertex_distance_m", np.inf)
        >= before.get("median_measured_vertex_distance_m", 0)
    ):
        return pose.copy(), dict(
            report,
            reason="candidate did not satisfy measured-fit and finite-support checks",
        )
    return candidate, dict(report, accepted=True)
