"""Verification: transfer error, spatial error map, cam-to-cam consistency.

This is the "Lollypop" equivalent for the challenge: the transform recovered
from calibration frames is applied to *held-out* frames/episodes and the object
projection is compared to the observed image points.  A single global RMS hides
distortion problems, so errors are also reported spatially and cross-checked
against the provided cam<->cam extrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from v2d.world_calib.calibration.solve import project_points

_PLACES = (0, 11, 23, 31, 39)


@dataclass(frozen=True)
class RigConsistencyRow:
    """Estimated vs provided cam<->cam transform for one pair."""

    source: str
    target: str
    rot_delta_deg: float
    trans_delta_m: float
    status: str


def correspondence_reprojection_errors(
    correspondences,
    rot: np.ndarray,
    trans: np.ndarray,
    intrinsics,
) -> np.ndarray:
    """Per-point pixel residuals of the object projection (``(N,)`` px)."""

    predicted = project_points(correspondences.points_world, rot, trans, intrinsics)
    diff = correspondences.points_image - predicted
    return np.sqrt(np.sum(diff * diff, axis=-1))


def error_stats_pixels(errors: np.ndarray) -> dict[str, float]:
    """``{mean, median, rmse, p95, max}`` pixel error summary."""

    e = np.asarray(errors, dtype=np.float64)
    if e.size == 0:
        return {k: float("nan") for k in ("mean", "median", "rmse", "p95", "max")}
    return {
        "mean": float(e.mean()),
        "median": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "p95": float(np.percentile(e, 95)),
        "max": float(e.max()),
    }


def pixel_error_grid_map(
    correspondences,
    errors: np.ndarray,
    width: int,
    height: int,
    grid: tuple[int, int] = (4, 4),
) -> np.ndarray:
    """Mean pixel error per image region; ``(ROWS, COLS)``.

    Reveals spatially-consistent calibration/distortion problems that a single
    global statistic hides.
    """

    rows, cols = grid
    accum = np.zeros((rows, cols), dtype=np.float64)
    counts = np.zeros((rows, cols), dtype=np.int64)
    u = correspondences.points_image[:, 0]
    v = correspondences.points_image[:, 1]
    if width <= 0 or height <= 0:
        width = int(max(u.max(), 1))
        height = int(max(v.max(), 1))
    ru = np.clip((u / width * cols).astype(int), 0, cols - 1)
    rv = np.clip((v / height * rows).astype(int), 0, rows - 1)
    np.add.at(accum, (rv, ru), errors)
    np.add.at(counts, (rv, ru), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(accum, counts, out=np.full_like(accum, np.nan), where=counts > 0)
    return out


def _invert_rigid(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


def rig_consistency_rows(
    rig,
    solves: Mapping[str, "CameraCalibrationSolve"],
) -> list[RigConsistencyRow]:
    """Compare estimated ``T_camA->camB`` against the provided extrinsics.

    Provided ``extrinsics_to_stereo_left`` is interpreted as the pose of that
    camera expressed in the stereo-left frame, i.e. ``T_left->camera``.  The
    estimate is ``T_WC @ inv(T_WL)`` from the two world solves.
    """

    rows: list[RigConsistencyRow] = []
    for pair in rig.stereo_pairs:
        left = pair.get("left")
        right = pair.get("right")
        if not left or not right or left not in solves or right not in solves:
            continue
        t_wl = solves[left].as_matrix()
        t_wc = solves[right].as_matrix()
        est = t_wc @ _invert_rigid(t_wl)
        provided = rig.extrinsic_matrix(right)
        if provided is None:
            rows.append(
                RigConsistencyRow(left, right, float("nan"), float("nan"), "no-provided-extrinsic")
            )
            continue
        provided_44 = np.eye(4, dtype=np.float64)
        provided_44[:3] = provided[:3]
        rel = est[:3, :3] @ provided_44[:3, :3].T
        angle = float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))))
        trans_delta = float(np.linalg.norm(est[:3, 3] - provided_44[:3, 3]))
        status = "consistent" if (angle < 1.0 and trans_delta < 1e-3) else "MISMATCH"
        rows.append(RigConsistencyRow(left, right, angle, trans_delta, status))
    return rows


def proxy_episode_guard(fit_episodes: list[int], test_episodes: list[int]) -> list[str]:
    """Warn if the first stage would tune on the held-out proxy episodes."""

    warnings = []
    for ep in _PLACES:
        if ep in fit_episodes:
            warnings.append(
                f"episode {ep} is in the proxy (held-out score) set; using it for calibration "
                "is allowed only as a fit sample, never for tuning."
            )
    return warnings