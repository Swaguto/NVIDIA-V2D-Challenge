"""Compare camera-space CAD hypotheses with visible target pixels, without GT."""
import cv2
import numpy as np
from scipy.spatial import cKDTree


def projected_support(moved, K, mask, depth, valid, excluded_mask,
                      depth_margin=.03, distance_clip=20., pixel_tolerance=3.):
    """Return separate coverage and exterior costs for a sampled CAD hypothesis.

    Labelled other-object pixels are excluded when the hypothesized CAD is not
    clearly in front of their measured depth. This is an observation diagnostic:
    masks and noisy stereo are not independent visibility ground truth.
    """
    moved, K = np.asarray(moved), np.asarray(K)
    mask, valid, excluded_mask = [np.asarray(x, dtype=bool) for x in (mask, valid, excluded_mask)]
    depth = np.asarray(depth)
    if (moved.ndim != 2 or moved.shape[1] != 3 or K.shape != (3, 3)
            or not np.isfinite(moved).all() or not np.isfinite(K).all()
            or mask.ndim != 2 or any(x.shape != mask.shape for x in (depth, valid, excluded_mask))
            or not mask.any() or not np.isfinite(depth[valid]).all()
            or (depth[valid] <= 0).any()):
        raise ValueError('finite camera points/intrinsics and nonempty aligned mask/depth required')
    if (not np.isfinite([depth_margin, distance_clip, pixel_tolerance]).all()
            or depth_margin < 0 or distance_clip <= 0 or pixel_tolerance < 0):
        raise ValueError('invalid projected-support parameters')
    front = moved[:, 2] > .01
    xyz = moved[front]
    projection = xyz @ K.T
    uv = projection[:, :2] / projection[:, 2:]
    pixel = np.rint(uv).astype(int)
    h, w = mask.shape
    inside = ((pixel[:, 0] >= 0) & (pixel[:, 0] < w)
              & (pixel[:, 1] >= 0) & (pixel[:, 1] < h))
    uv, pixel, z = uv[inside], pixel[inside], xyz[inside, 2]
    px, py = pixel[:, 0], pixel[:, 1]
    measured, known = depth[py, px], valid[py, px]
    occluded = (known & (measured + depth_margin < z))
    occluded |= excluded_mask[py, px] & (~known | (z >= measured-depth_margin))
    visible = ~occluded
    distance_to_mask = cv2.distanceTransform((~mask).astype('uint8'), cv2.DIST_L2, 5)
    outside_dist = distance_to_mask[py[visible], px[visible]]
    outside = float(np.minimum(outside_dist, distance_clip).mean()/distance_clip) if len(outside_dist) else 1.
    yy, xx = np.nonzero(mask)
    target_uv = np.column_stack([xx, yy])
    support_uv = uv[visible]
    coverage_dist = (cKDTree(support_uv).query(target_uv)[0] if len(support_uv)
                     else np.full(len(target_uv), distance_clip))
    coverage = float(np.minimum(coverage_dist, distance_clip).mean()/distance_clip)
    return {'projected_support_score': outside+coverage,
            'projected_outside_mask_cost': outside, 'target_coverage_cost': coverage,
            'target_pixels_supported_within_3px_fraction': float((coverage_dist <= pixel_tolerance).mean()),
            'visible_projected_samples': int(visible.sum())}
