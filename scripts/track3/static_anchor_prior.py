"""Source-only stillness evidence for an independently transported body anchor.

This experiment compares observed mask boundaries after camera compensation.
It does not modify a track, infer contact, replace missing masks, or bypass a
depth/pose acceptance gate. Pixel agreement cannot prove a body's full 3D pose.
"""

import cv2
import numpy as np


def mask_stillness(depth, valid, anchor_mask, current_mask, K, camera_delta):
    depth, valid, anchor_mask, current_mask = map(
        np.asarray, (depth, valid, anchor_mask, current_mask)
    )
    K, camera_delta = np.asarray(K), np.asarray(camera_delta)
    if (
        depth.ndim != 2
        or valid.shape != depth.shape
        or anchor_mask.shape != depth.shape
        or current_mask.shape != depth.shape
        or K.shape != (3, 3)
        or not np.isfinite(K).all()
        or min(K[0, 0], K[1, 1]) <= 0
        or camera_delta.shape != (4, 4)
        or not np.isfinite(camera_delta).all()
        or not np.allclose(camera_delta[3], [0, 0, 0, 1])
        or not np.allclose(
            camera_delta[:3, :3].T @ camera_delta[:3, :3], np.eye(3), atol=1e-5
        )
        or not np.isclose(np.linalg.det(camera_delta[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError(
            "Stillness needs calibrated common grids and a rigid independent camera delta"
        )
    anchor_mask, current_mask = anchor_mask.astype(bool), current_mask.astype(bool)
    good = (
        valid.astype(bool)
        & anchor_mask
        & np.isfinite(depth)
        & (depth > 0.1)
        & (depth < 2.5)
    )
    report = {
        "eligible": False,
        "measured_anchor_pixels": int(good.sum()),
        "current_mask_pixels": int(current_mask.sum()),
        "missing_observations_filled": False,
        "boundary_limit_px": 3.0,
        "boundary_quantile": 0.95,
        "source_labels_read": False,
    }
    if min(int(good.sum()), int(current_mask.sum())) < 200:
        return dict(report, reason="insufficient measured mask support")
    # Warp the observed image mask rather than turning stereo holes into false
    # silhouette boundaries. Unknown projection depths use an explicit median
    # foreground-depth hypothesis; source depth/validity arrays stay unchanged.
    median_depth = float(np.median(depth[good]))
    y, x = np.nonzero(anchor_mask)
    z = np.where(good[y, x], depth[y, x], median_depth)
    report.update(
        projection_depth_hypothesis="measured depth where valid, median foreground depth elsewhere",
        inferred_projection_depth_pixels=int((~good[y, x]).sum()),
        projection_is_not_measured_depth_completion=True,
    )
    points = np.c_[(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z]
    points = np.einsum("ni,ji->nj", points, camera_delta[:3, :3]) + camera_delta[:3, 3]
    good = np.isfinite(points).all(-1) & (points[:, 2] > 0.05)
    points = points[good]
    if len(points) < 200:
        return dict(report, reason="anchor is outside the observing camera")
    u = np.rint(K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2]).astype(int)
    v = np.rint(K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]).astype(int)
    inside = (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
    warped = np.zeros(depth.shape, np.uint8)
    warped[v[inside], u[inside]] = 1
    # This closes tiny sampling holes in the inferred projection only. Neither
    # observed mask nor measured depth is edited or relabelled.
    warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if warped.sum() < 200:
        return dict(report, reason="too little projected anchor remains visible")
    boundary_a = warped & (1 - cv2.erode(warped, np.ones((3, 3), np.uint8)))
    mask = current_mask.astype(np.uint8)
    boundary_b = mask & (1 - cv2.erode(mask, np.ones((3, 3), np.uint8)))
    if not boundary_a.any() or not boundary_b.any():
        return dict(report, reason="no supported foreground boundary")
    da = cv2.distanceTransform(1 - boundary_a, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    db = cv2.distanceTransform(1 - boundary_b, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance = max(
        float(np.quantile(da[boundary_b > 0], 0.95)),
        float(np.quantile(db[boundary_a > 0], 0.95)),
    )
    iou = float(
        ((warped > 0) & current_mask).sum() / ((warped > 0) | current_mask).sum()
    )
    return dict(
        report,
        eligible=distance <= 3 and iou >= 0.75,
        symmetric_boundary_p95_px=distance,
        camera_compensated_mask_iou=iou,
        projected_anchor_pixels=int(warped.sum()),
        scope="2D stillness hypothesis only; original tracking depth/jump gates still required",
    )
