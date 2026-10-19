"""Visible-surface depth comparisons with explicit missing-ray semantics.

An estimated depth-map sample is a surface observation. It does not observe an
occluded anatomical joint, and this module never adds a skin-to-joint offset or
upgrades observation confidence. Inputs use a calibrated, undistorted CV camera
with z forward and depth in metres along z, rather than Euclidean ray range.
"""

from typing import NamedTuple

import numpy as np


class SurfaceComparison(NamedTuple):
    observed_valid: np.ndarray
    model_hit: np.ndarray
    predicted_depth_m: np.ndarray
    residual_m: np.ndarray


def front_surface_depth(vertices, faces, pixels, intrinsics):
    """Return the first positive ray/triangle z depth; misses remain infinity."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces)
    uv = np.asarray(pixels, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    if v.ndim != 2 or v.shape[1:] != (3,) or f.ndim != 2 or f.shape[1:] != (3,):
        raise ValueError("Expected vertices[N,3] and triangle faces[M,3]")
    if not np.issubdtype(f.dtype, np.integer) or (f.size and (f.min() < 0 or f.max() >= len(v))):
        raise ValueError("Faces must contain valid integer vertex indices")
    if uv.ndim != 2 or uv.shape[1:] != (2,) or k.shape != (3, 3):
        raise ValueError("Expected pixels[N,2] and pinhole intrinsics[3,3]")
    if not all(np.isfinite(x).all() for x in (v, uv, k)):
        raise ValueError("Surface geometry and queried pixels must be finite")
    canonical = np.array([[k[0, 0], 0, k[0, 2]], [0, k[1, 1], k[1, 2]], [0, 0, 1.]])
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k, canonical, atol=1e-12, rtol=0):
        raise ValueError("Use calibrated undistorted pinhole pixels")
    result = np.full(len(uv), np.inf)
    if not len(f) or not len(uv):
        return result
    triangle = v[f]
    a, edge1, edge2 = triangle[:, 0], triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    origin_to_a = -a
    cross = np.cross(origin_to_a, edge1)
    # Bound pairwise temporary arrays; no dependency on a renderer or GPU.
    chunk = min(128, max(1, 250_000 // len(f)))
    for start in range(0, len(uv), chunk):
        p = uv[start:start + chunk]
        rays = np.c_[(p[:, 0] - k[0, 2]) / k[0, 0], (p[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(p))]
        h = np.cross(rays[:, None, :], edge2[None])
        det = np.einsum("fi,rfi->rf", edge1, h)
        nonzero = abs(det) > 1e-12
        inverse = np.divide(1, det, out=np.zeros_like(det), where=nonzero)
        u = np.einsum("fi,rfi->rf", origin_to_a, h) * inverse
        w = np.einsum("ri,fi->rf", rays, cross) * inverse
        z = np.einsum("fi,fi->f", edge2, cross)[None] * inverse
        hit = nonzero & (u >= -1e-8) & (w >= -1e-8) & (u + w <= 1 + 1e-8) & (z > 0)
        result[start:start + len(p)] = np.min(np.where(hit, z, np.inf), axis=1)
    return result


def compare_estimated_surface(vertices, faces, pixels, intrinsics, depth_m, valid, *, evidence_kind):
    """Compare the correct surface operator; preserve every expected missing hit.

    A valid observation with no model intersection has an infinite residual.
    Unknown observations remain NaN and have ``observed_valid=False``. Callers
    must require all expected model hits when checking coverage. Numeric source
    and native reference-contact acceptance thresholds belong to their existing
    gates and are not changed here.
    """
    if evidence_kind != "estimated_visible_surface_depth":
        raise ValueError("Surface comparisons require explicitly typed estimated visible-surface evidence")
    uv = np.asarray(pixels, dtype=float)
    depth = np.asarray(depth_m, dtype=float)
    observed = np.asarray(valid)
    if uv.ndim != 2 or uv.shape[1:] != (2,) or depth.shape != (len(uv),) or observed.shape != depth.shape:
        raise ValueError("Surface observations require matching pixel, depth and validity axes")
    if observed.dtype != np.bool_:
        raise ValueError("Surface validity must be an explicit boolean mask")
    if not np.isfinite(depth[observed]).all() or (depth[observed] <= 0).any():
        raise ValueError("Valid surface depths must be finite and positive")
    predicted = np.full(len(uv), np.nan)
    predicted[observed] = front_surface_depth(vertices, faces, uv[observed], intrinsics)
    model_hit = observed & np.isfinite(predicted)
    residual = predicted - depth
    residual[~observed] = np.nan
    return SurfaceComparison(observed.copy(), model_hit, predicted, residual)
