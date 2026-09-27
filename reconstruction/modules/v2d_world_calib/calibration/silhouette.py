"""Option-B verification: silhouette (mask) alignment quality for a candidate transform.

The keypoint path in :mod:`solve` needs paired 2D keypoints.  When only 2D
object masks are available, alignment can instead be judged by rendering the GT
mesh under a candidate ``world -> camera`` transform and measuring mask overlap
(IoU) against the observed object mask.

Rendering here is deliberately dependency-light: a single ``cv2.fillPoly``
triangle rasterizer over the projected mesh faces (falling back to the convex
hull when face topology is not provided).
"""

from __future__ import annotations

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

from v2d.world_calib.calibration.correspondences import quaternion_to_matrix


def silhouette_render_mask(
    points_object: np.ndarray,  # (V, 3) object-frame vertices
    pose_world_object: np.ndarray,  # (7,) xyzw
    rot: np.ndarray,
    trans: np.ndarray,
    intrinsics,
    width: int,
    height: int,
    faces: np.ndarray | None = None,  # (F, 3) vertex indices, optional
) -> np.ndarray:
    """Rasterize the mesh silhouette (bool ``(H, W)`` mask) under R, t."""

    if not _HAS_CV2:
        raise RuntimeError("silhouette rendering requires opencv")
    pts = np.asarray(points_object, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(pose_world_object, dtype=np.float64)
    qn = q[3:] / (np.linalg.norm(q[3:]) + 1e-12)
    local_rot = quaternion_to_matrix(qn)
    world = pts @ local_rot.T + q[:3]
    cam = world @ np.asarray(rot, dtype=np.float64).T + np.asarray(trans, dtype=np.float64).reshape(3)
    pixels, _ = cv2.projectPoints(
        world,
        cv2.Rodrigues(np.asarray(rot, dtype=np.float64))[0],
        np.asarray(trans, dtype=np.float64).reshape(3),
        intrinsics.matrix(),
        intrinsics.dist_coeffs,
    )
    pixels = pixels.reshape(-1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    if faces is not None and faces.shape[0] > 0:
        for tri in np.asarray(faces, dtype=np.int64):
            poly = pixels[tri].astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
    else:
        if pixels.shape[0] >= 3:
            hull = cv2.convexHull(pixels.astype(np.float32)).astype(np.int32)
            cv2.fillPoly(mask, [hull], 1)
    return mask.astype(bool)


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks."""

    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = float(np.count_nonzero(a & b))
    union = float(np.count_nonzero(a | b))
    return inter / union if union > 0 else float("nan")


def silhouette_overlap_verify(
    observation,
    rot: np.ndarray,
    trans: np.ndarray,
    intrinsics,
) -> float:
    """IoU of the rendered GT mesh vs the observed mask under ``R, t``."""

    h, w = observation.mask.shape[:2]
    rendered = silhouette_render_mask(
        observation.points_object,
        observation.pose_world_object,
        rot,
        trans,
        intrinsics,
        w,
        h,
    )
    return silhouette_iou(rendered, observation.mask)