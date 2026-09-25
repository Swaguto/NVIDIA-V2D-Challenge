"""Extract four reliability signals from explicit, aligned diagnostics.

Masks must be boolean and share a pixel grid. Keypoints must already be in
the same image coordinates. Positions are metric world-space hand/object
relative positions. No ground truth is needed by this module.
"""
import numpy as np


def visibility(valid_keypoints):
    valid = np.asarray(valid_keypoints)
    if valid.dtype != bool or valid.ndim < 1 or valid.shape[-1] == 0:
        raise ValueError("visibility requires nonempty boolean keypoint validity")
    return valid.mean(axis=-1)


def mask_iou(observed, rendered):
    observed, rendered = np.asarray(observed), np.asarray(rendered)
    if observed.shape != rendered.shape or observed.ndim < 2:
        raise ValueError("mask grids must match")
    if observed.dtype != bool or rendered.dtype != bool:
        raise ValueError("masks must be boolean")
    union = (observed | rendered).sum(axis=(-2, -1))
    intersection = (observed & rendered).sum(axis=(-2, -1))
    return np.divide(intersection, union, out=np.full(union.shape, np.nan), where=union > 0)


def reprojection_error(observed, projected, valid, image_size):
    observed, projected = np.asarray(observed, float), np.asarray(projected, float)
    valid = np.asarray(valid, bool)
    if observed.shape != projected.shape or observed.shape[-1] != 2 or valid.shape != observed.shape[:-1]:
        raise ValueError("keypoint arrays must align")
    if len(image_size) != 2 or not all(np.isfinite(x) and x > 0 for x in image_size):
        raise ValueError("positive image width and height required")
    valid = valid & np.isfinite(observed).all(-1) & np.isfinite(projected).all(-1)
    errors = np.linalg.norm(observed - projected, axis=-1) / np.linalg.norm(image_size)
    count = valid.sum(-1)
    return np.divide(np.where(valid, errors, 0).sum(-1), count,
                     out=np.full(count.shape, np.nan), where=count > 0)


def temporal_acceleration(relative_positions, timestamps):
    positions, times = np.asarray(relative_positions, float), np.asarray(timestamps, float)
    if positions.ndim < 2 or positions.shape[0] != len(times) or positions.shape[-1] != 3:
        raise ValueError("positions must be time-major xyz")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    result = np.full(positions.shape[:-1], np.nan)
    if len(times) >= 3:
        dt = np.diff(times).reshape((-1,) + (1,) * (positions.ndim - 1))
        velocity = np.diff(positions, axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / ((dt[1:] + dt[:-1]) / 2)
        result[1:-1] = np.linalg.norm(acceleration, axis=-1)
    return result


def extract(visibility_mask, observed_masks, rendered_masks, observed_keypoints,
            projected_keypoints, relative_positions, timestamps, image_size):
    """Inputs have leading [T,H,B] axes, followed by keypoints/pixels/xyz."""
    values = np.stack((visibility(visibility_mask), mask_iou(observed_masks, rendered_masks),
                       reprojection_error(observed_keypoints, projected_keypoints,
                                          visibility_mask, image_size),
                       temporal_acceleration(relative_positions, timestamps)), axis=-1)
    valid = np.isfinite(values)
    return np.where(valid, values, 0), valid
