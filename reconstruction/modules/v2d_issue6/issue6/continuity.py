"""Sequence depth corrections for a fitted hand prior, with explicit uncertainty.

The optimizer's pose is the prior on *every* frame. Missing depth measurements
do not remove that pose. Corrections are regularized in the fitted world basis,
so camera movement does not become a smoothing penalty on real hand motion.
"""

import numpy as np


def depth_occlusion(explicit_mask, object_depth, measured, foreground_margin=0.005):
    """A projected CAD silhouette must not hide a measured foreground hand.

    Explicit semantic object masks remain exclusions. CAD is an occluder unless
    valid stereo puts the visible surface clearly in front of its geometry.
    """
    explicit = np.asarray(explicit_mask, bool)
    objects = np.asarray(object_depth, float)
    measured = np.asarray(measured, float)
    if not (explicit.shape == objects.shape == measured.shape):
        raise ValueError("Occlusion inputs must share the color-camera grid")
    foreground = np.isfinite(measured) & (measured > 0) & (measured < objects - foreground_margin)
    return explicit | ((objects > 0) & ~foreground)


def sequence_corrections(
    times, offsets_camera, weights, camera_rotations, smooth_seconds=0.15, prior_weight=0.05, huber_m=0.015
):
    """Robustly fit a smooth correction, retaining zero correction as a prior.

    camera_rotations maps fitted world vectors into the source camera. Depth
    rejection has weight zero; it is never converted into an observed pose.
    A second derivative penalizes jitter without penalizing constant velocity.
    """
    times = np.asarray(times, float)
    offsets = np.asarray(offsets_camera, float)
    weights = np.asarray(weights, float)
    rotations = np.asarray(camera_rotations, float)
    n = len(times)
    if (
        offsets.shape != (n, 3)
        or weights.shape != (n,)
        or rotations.shape != (n, 3, 3)
        or n < 1
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(offsets).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isfinite(rotations).all()
        or prior_weight <= 0
        or smooth_seconds < 0
        or huber_m <= 0
    ):
        raise ValueError("Invalid sequence alignment inputs")
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4):
        raise ValueError("Expected world-to-camera rotations")
    world = np.einsum("nji,nj->ni", rotations, offsets)
    # Scale finite differences by actual timestamps, rather than assuming 20 Hz.
    d2 = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        a, b = times[i + 1] - times[i], times[i + 2] - times[i + 1]
        d2[i, i : i + 3] = [2 / (a * (a + b)), -2 / (a * b), 2 / (b * (a + b))]
    penalty = smooth_seconds**4 * (d2.T @ d2) + prior_weight * np.eye(n)
    effective = weights.copy()
    for _ in range(8):
        fitted = np.linalg.solve(penalty + np.diag(effective), effective[:, None] * world)
        residual = np.linalg.norm(fitted - world, axis=1)
        effective = weights * np.minimum(1, huber_m / np.maximum(residual, 1e-12))
    return np.einsum("nij,nj->ni", rotations, fitted)


def visibility_summary(frame_ids):
    ids = sorted(set(frame_ids))
    return {"frames": len(ids), "internal_gaps": sum(b > a + 1 for a, b in zip(ids, ids[1:]))}


def bridge_visibility(times, observed, projected, maximum_gap_seconds=1.5):
    """Retain a fitted pose in bounded gaps with visible observations at both ends.

    This bridges visibility only, never geometry or metric validity. An existing
    finite projected pose is required on every gap frame. No end extrapolation.
    """
    times = np.asarray(times, float)
    observed = np.asarray(observed, bool)
    projected = np.asarray(projected, bool)
    if times.shape != observed.shape or times.shape != projected.shape or np.any(np.diff(times) <= 0):
        raise ValueError("Visibility must share a strictly increasing frame axis")
    result = observed & projected
    endpoints = np.flatnonzero(result)
    for a, b in zip(endpoints, endpoints[1:]):
        if b > a + 1 and times[b] - times[a] <= maximum_gap_seconds and projected[a : b + 1].all():
            result[a : b + 1] = True
    return result
