"""Measured visible-surface translation can initialize a moving object's tracker.

The surface median is not the object's body origin. Only its displacement is
used as a refinement hypothesis; no pose observation or contact is certified.
"""

import numpy as np


def measured_surface(depth, valid, mask, K):
    depth, valid, mask, K = map(np.asarray, (depth, valid, mask, K))
    if (
        depth.ndim != 2
        or valid.shape != depth.shape
        or mask.shape != depth.shape
        or K.shape != (3, 3)
    ):
        raise ValueError("Need matching calibrated depth, validity and foreground")
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Need finite positive focal lengths")
    selected = (
        valid.astype(bool)
        & (mask > 127)
        & np.isfinite(depth)
        & (depth > 0.1)
        & (depth < 2.5)
    )
    y, x = np.nonzero(selected)
    if len(x) < 100:
        return None
    z = depth[y, x]
    points = np.c_[(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z]
    return np.median(points, axis=0)


def translated_prior(pose, previous_surface, current_surface, camera_transport=None):
    """Return a new initialization; never mutate the original acceptance prior."""
    pose, previous_surface, current_surface = [
        np.asarray(a, float) for a in (pose, previous_surface, current_surface)
    ]
    if (
        pose.shape != (4, 4)
        or previous_surface.shape != (3,)
        or current_surface.shape != (3,)
        or not all(
            np.isfinite(a).all() for a in (pose, previous_surface, current_surface)
        )
    ):
        raise ValueError("Need finite pose and measured surface points")
    if camera_transport is not None:
        camera_transport = np.asarray(camera_transport)
        if camera_transport.shape != (4, 4) or not np.isfinite(camera_transport).all():
            raise ValueError("Invalid independent camera transport")
        previous_surface = (
            camera_transport[:3, :3] @ previous_surface + camera_transport[:3, 3]
        )
    shifted = pose.copy()
    shifted[:3, 3] += current_surface - previous_surface
    return shifted
