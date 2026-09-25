"""Stages 1-3: solve the per-camera ``world -> camera`` rigid transform.

Pipeline per camera:

    1. ``solve_pnp_ransac``  - RANSAC initialization from many 3D<->2D pairs.
    2. ``refine_reprojection`` - robust (Huber) numpy Levenberg-Marquardt over
       SE(3), minimizing multi-frame pixel reprojection error, honoring
       distortion.  A numpy optimizer (no scipy) keeps the module light, with
       cv2's ``solvePnPRefineLM`` available as a cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

from v2d.world_calib.calibration.correspondences import (
    WorldCameraCorrespondences,
    quaternion_from_matrix,
)


def project_points(
    points_world: np.ndarray,
    rot: np.ndarray,
    trans: np.ndarray,
    intrinsics,
) -> np.ndarray:
    """Project ``(N,3)`` world points -> ``(N,2)`` pixels honoring distortion."""

    if not _HAS_CV2:
        raise RuntimeError("solve + refinement require opencv (cv2.projectPoints)")
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    pixels, _ = cv2.projectPoints(
        pts,
        cv2.Rodrigues(np.asarray(rot, dtype=np.float64))[0],
        np.asarray(trans, dtype=np.float64).reshape(3),
        intrinsics.matrix(),
        intrinsics.dist_coeffs,
    )
    return pixels.reshape(-1, 2)


def _residual_vector(
    correspondences: WorldCameraCorrespondences,
    rot: np.ndarray,
    trans: np.ndarray,
    intrinsics,
) -> np.ndarray:
    predicted = project_points(correspondences.points_world, rot, trans, intrinsics)
    return (correspondences.points_image - predicted).reshape(-1)


def solve_pnp_ransac(
    camera: str,
    correspondences: WorldCameraCorrespondences,
    intrinsics,
    min_inliers: int = 6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Robust RANSAC-PnP initialization.  Returns ``(rot, trans, inliers)``.

    ``cv2.solvePnPRansac`` happily returns a *degenerate* success (model with
    points projected behind the camera or at infinity), which a downstream
    local optimizer cannot escape.  So several solver flags are tried, each
    candidate is gated on geometric validity + inlier count + inlier RMS, and
    the best is returned.
    """

    if not _HAS_CV2:
        raise RuntimeError("PnP requires opencv")
    n = correspondences.count
    if n < min_inliers:
        raise ValueError(f"{camera}: need >= {min_inliers} correspondences, got {n}")
    object_pts = correspondences.points_world.astype(np.float64)
    image_pts = correspondences.points_image.astype(np.float64)

    flags_list = tuple(
        getattr(cv2, f"SOLVEPNP_{name}")
        for name in ("EPNP", "ITERATIVE", "P3P", "DLS", "AP3P", "IPPE")
        if hasattr(cv2, f"SOLVEPNP_{name}")
    )
    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for flags in flags_list:
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_pts,
                image_pts,
                intrinsics.matrix(),
                intrinsics.dist_coeffs,
                iterationsCount=300,
                reprojectionError=1.5,
                confidence=0.999,
                flags=flags,
            )
        except cv2.error:
            continue
        if not ok or tvec is None:
            continue
        rot, _ = cv2.Rodrigues(rvec)
        trans = tvec.reshape(3)
        # gate 1: object must be in front of the camera
        cam_z = (object_pts @ rot.T + trans)[:, 2]
        if np.median(cam_z) <= 0.0 or np.mean(cam_z > 0.0) < 0.9:
            continue
        # gate 2: inlier count + inlier RMS
        predicted = project_points(object_pts, rot, trans, intrinsics)
        errs = np.sqrt(np.sum((predicted - image_pts) ** 2, axis=-1))
        if inliers is None:
            inlier_mask = errs < 1.5
            inlier_count = int(inlier_mask.sum())
        else:
            inlier_mask = np.zeros(n, dtype=bool)
            inlier_mask[np.asarray(inliers, dtype=np.int64).reshape(-1)] = True
            inlier_count = int(inlier_mask.sum())
        inlier_rms = float(np.sqrt(np.mean(errs[inlier_mask] ** 2))) if inlier_count else float("inf")
        score = (float(-inlier_count), inlier_rms)  # most inliers, then tightest
        candidates.append((inlier_rms, inlier_count, rot, trans))

    if not candidates:
        raise RuntimeError(f"{camera}: PnP-RANSAC failed to converge on any solver")

    candidates.sort(key=lambda c: (-c[1], c[0]))
    _, inlier_count, rot, trans = candidates[0]
    if inlier_count < min_inliers:
        raise RuntimeError(f"{camera}: too few PnP inliers ({inlier_count} < {min_inliers})")
    return rot, trans, inlier_count


def refine_reprojection(
    camera: str,
    correspondences: WorldCameraCorrespondences,
    intrinsics,
    init_rot: np.ndarray,
    init_trans: np.ndarray,
    robust_scale: float = 1.5,
    max_iterations: int = 40,
    damping_init: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float, bool, int]:
    """Huber-weighted Levenberg-Marquardt refinement of ``(R, t)``.

    Minimizes ``sum_i rho(||u_i - pi(K, D, R p_i + t)||, robust_scale)`` with
    rho the Huber loss.  SE(3) is parameterized as (axis-angle, translation);
    the Jacobian is computed numerically (6 params x 2N residuals), so even
    heavily distorted cameras are handled without special-casing the model.

    Returns ``(rot, trans, final_rms_px, converged, iterations)``.
    """

    points_world = correspondences.points_world
    points_image = correspondences.points_image

    rot = np.asarray(init_rot, dtype=np.float64)
    trans = np.asarray(init_trans, dtype=np.float64).reshape(3)
    rvec, _ = cv2.Rodrigues(rot)
    x = np.concatenate([rvec.reshape(3), trans])
    c = float(robust_scale)

    def residuals(xx: np.ndarray) -> np.ndarray:
        r, _ = cv2.Rodrigues(xx[:3])
        return _residual_vector(correspondences, r, xx[3:], intrinsics)

    r = residuals(x)
    mu = damping_init
    converged = False
    iterations = 0
    last_cost = float("inf")

    for _ in range(max_iterations):
        iterations += 1
        n = r.shape[0]
        # numeric Jacobian (6 columns)
        h = 1e-6
        jac = np.zeros((n, 6), dtype=np.float64)
        x0 = x.copy()
        for k in range(6):
            xh = x0.copy()
            xh[k] += h
            jac[:, k] = (residuals(xh) - r) / h
        # Huber weights
        mag = np.abs(r).reshape(-1, 2)
        row_mag = np.sqrt(np.sum(mag * mag, axis=-1))
        w = np.minimum(1.0, c / np.maximum(row_mag, 1e-12))  # per 2D point
        weights = np.repeat(w, 2)
        rw = r * weights
        jw = jac * weights[:, None]
        jtj = jw.T @ jw + mu * np.eye(6)
        try:
            dx = np.linalg.solve(jtj, -jw.T @ rw)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(jtj, -jw.T @ rw, rcond=None)[0]
        cost_now = float(np.sum(row_mag * w + (row_mag * w) * np.abs(w) * 0.0))

        trial = x + dx
        r_trial = residuals(trial)
        trial_mag = np.sqrt(np.sum(r_trial.reshape(-1, 2) ** 2, axis=-1))
        trial_cost = float(np.sum(
            np.where(trial_mag <= c, 0.5 * trial_mag**2, c * (trial_mag - 0.5 * c))
        ))
        if trial_cost < cost_now:
            x = trial
            r = r_trial
            mu = max(mu * 0.3, 1e-8)
            if float(np.linalg.norm(dx)) < 1e-9 or (cost_now - trial_cost) < 1e-10:
                converged = True
        else:
            mu *= 10.0
        if abs(last_cost - cost_now) < 1e-12 and cost_now < 1e6:
            converged = True
        last_cost = cost_now
        if converged:
            break

    rot, _ = cv2.Rodrigues(x[:3])
    trans = x[3:]
    rms = float(np.sqrt(np.mean(np.sum(r.reshape(-1, 2) ** 2, axis=-1))))
    return rot, trans, rms, converged, iterations


@dataclass
class CameraCalibrationSolve:
    """Per-camera result of the world<->camera solve."""

    camera: str
    rot: np.ndarray  # (3, 3) R_CW
    trans: np.ndarray  # (3,) t_CW
    status: str
    samples: int
    inliers: int
    fit_rms_px: float
    refine_rms_px: float
    iterations: int

    @property
    def quat_xyzw(self) -> np.ndarray:
        return quaternion_from_matrix(self.rot)

    def as_matrix(self) -> np.ndarray:
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = self.rot
        m[:3, 3] = self.trans
        return m


def solve_camera(
    camera: str,
    correspondences: WorldCameraCorrespondences,
    intrinsics,
    min_inliers: int = 6,
) -> CameraCalibrationSolve:
    """Full pipeline for one camera: RANSAC-PnP init -> Huber refinement."""

    rot0, trans0, inliers = solve_pnp_ransac(camera, correspondences, intrinsics, min_inliers)
    fit_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    _residual_vector(correspondences, rot0, trans0, intrinsics).reshape(-1, 2) ** 2,
                    axis=-1,
                )
            )
        )
    )
    rot, trans, refine_rms, _converged, iterations = refine_reprojection(
        camera, correspondences, intrinsics, rot0, trans0
    )
    return CameraCalibrationSolve(
        camera=camera,
        rot=rot,
        trans=trans,
        status="ok",
        samples=correspondences.count,
        inliers=inliers,
        fit_rms_px=fit_rms,
        refine_rms_px=refine_rms,
        iterations=iterations,
    )