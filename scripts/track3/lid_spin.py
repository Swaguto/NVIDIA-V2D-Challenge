"""Fixed-axis appearance diagnostics for the scanned pot lid.

The planar texture is an approximation to the marker-bearing CAD face. These
helpers do not estimate translation, camera motion, or a tracking trajectory.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class LidTemplate:
    gray: np.ndarray
    roi: np.ndarray
    axis: np.ndarray
    tangent1: np.ndarray
    tangent2: np.ndarray
    origin: np.ndarray
    scale: float


def build_template(vertices, faces, colors):
    """Rasterize the CAD +X face about its measured minimum-variance axis."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    colors = np.asarray(colors)[:, :3]
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3
            or not np.isfinite(vertices).all() or colors.shape != vertices.shape
            or faces.ndim != 2 or faces.shape[1] != 3):
        raise ValueError('finite textured triangular lid mesh required')
    values, vectors = np.linalg.eigh(np.cov(vertices.T))
    axis = vectors[:, 0]
    if axis[0] < 0:
        axis = -axis
    tangent1 = np.array([0., 1., 0.])
    tangent1 -= np.dot(tangent1, axis) * axis
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(axis, tangent1)
    planar = np.column_stack([vertices @ tangent1, vertices @ tangent2])
    height = vertices @ axis
    size, border = 512, 24
    scale = (size - 2 * border) / np.ptp(planar, axis=0).max()
    offset = planar.min(0) - border / scale
    uv = np.rint((planar - offset) * scale).astype('int32')
    texture = np.zeros((size, size, 3), np.uint8)
    hit = np.zeros((size, size), np.uint8)
    height_image = np.full((size, size), np.nan, np.float32)
    for face in faces[np.argsort(height[faces].mean(1))]:
        rgb = tuple(int(x) for x in np.mean(colors[face], axis=0))
        cv2.fillConvexPoly(texture, uv[face], rgb)
        cv2.fillConvexPoly(hit, uv[face], 1)
        cv2.fillConvexPoly(height_image, uv[face], float(height[face].mean()))
    yy, xx = np.indices(hit.shape)
    physical = np.stack([xx / scale + offset[0], yy / scale + offset[1]], axis=-1)
    radius = np.linalg.norm(physical, axis=-1)
    roi = (hit > 0) & (radius > .035) & (radius < .086)
    if not roi.any():
        raise ValueError('lid marker annulus is empty')
    plane_height = float(np.nanmedian(height_image[roi]))
    origin = plane_height * axis + offset[0] * tangent1 + offset[1] * tangent2
    gray = cv2.cvtColor(texture, cv2.COLOR_RGB2GRAY).astype('float32')
    return (LidTemplate(gray, roi, axis, tangent1, tangent2, origin, scale),
            texture, {'axis_in_cad': axis.tolist(),
                      'covariance_eigenvalues': values.tolist(),
                      'plane_height_m': plane_height,
                      'texture_roi_radius_m': [.035, .086]})


def spin_rotation(rotation, axis, degrees):
    """Right-compose CAD-axis spin, preserving the camera-space axis tilt."""
    return rotation @ Rotation.from_rotvec(axis * np.deg2rad(degrees)).as_matrix()


def appearance_feature(gray, method):
    gray = np.asarray(gray, dtype=np.float32)
    if method == 'gradient_ncc':
        smooth = cv2.GaussianBlur(gray, (0, 0), 1)
        return cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0),
                             cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    if method == 'highpass_ncc':
        return gray - cv2.GaussianBlur(gray, (0, 0), 3)
    raise ValueError('unknown appearance feature')


def sweep_appearance(template, image, mask, K, rotation, translation, *,
                     method='gradient_ncc', minimum_gap=.1, maximum_cost=.75):
    """Return coarse/fine curves and a guarded proposal without reading GT."""
    if minimum_gap < 0 or not np.isfinite(minimum_gap):
        raise ValueError('finite nonnegative separated-peak gap required')
    if not 0 <= maximum_cost <= 2:
        raise ValueError('appearance cost limit must be in [0, 2]')
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError('empty target mask')
    x0, x1 = max(int(xx.min()) - 10, 0), min(int(xx.max()) + 11, image.shape[1])
    y0, y1 = max(int(yy.min()) - 10, 0), min(int(yy.max()) + 11, image.shape[0])
    crop = image[y0:y1, x0:x1]
    observed = appearance_feature(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), method)
    target = mask[y0:y1, x0:x1]
    cropped_K = K.copy()
    cropped_K[0, 2] -= x0
    cropped_K[1, 2] -= y0
    def render(degrees):
        R = spin_rotation(rotation, template.axis, degrees)
        H = cropped_K @ np.column_stack([
            R @ template.tangent1 / template.scale,
            R @ template.tangent2 / template.scale,
            R @ template.origin + translation])
        warped = cv2.warpPerspective(template.gray, H, (x1 - x0, y1 - y0))
        eligible = (cv2.warpPerspective(template.roi.astype('float32'), H,
                                       (x1 - x0, y1 - y0)) > .99) & target
        return R, warped, eligible
    def cost(degrees):
        _, warped, eligible = render(degrees)
        a = observed[eligible]
        b = appearance_feature(warped, method)[eligible]
        result = {'angle_deg': float(degrees), 'cost': 2., 'pixels': len(a)}
        if len(a) >= 100:
            a, b = a - a.mean(), b - b.mean()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom > 1e-9:
                result['cost'] = 1 - float(np.clip(np.dot(a, b) / denom, -1, 1))
        return result
    coarse = [cost(a) for a in np.arange(0, 360, 5)]
    seed = min(coarse, key=lambda c: c['cost'])['angle_deg']
    fine = [cost(a) for a in np.arange(seed - 5, seed + 5.01, .5)]
    best = min(fine, key=lambda c: c['cost'])
    alternative = min((c for c in coarse
                       if abs((c['angle_deg'] - best['angle_deg'] + 180) % 360 - 180) >= 30),
                      key=lambda c: c['cost'])
    gap = alternative['cost'] - best['cost']
    center = rotation @ template.origin + translation
    face_visible = np.dot(rotation @ template.axis, -center) > 0
    reason = ('marker_face_not_visible' if not face_visible else
              'appearance_cost_above_limit' if best['cost'] > maximum_cost else
              'ambiguous_separated_peaks' if gap < minimum_gap else None)
    reliable = reason is None
    proposed_R, proposed_image, _ = render(best['angle_deg'])
    applied_R = proposed_R if reliable else rotation.copy()
    _, applied_image, _ = render(best['angle_deg'] if reliable else 0.)
    return {'coarse_curve': coarse, 'fine_curve': fine, 'selected': best,
            'alternative_at_least30deg': alternative, 'separated_peak_cost_gap': gap,
            'baseline_cost': coarse[0]['cost'], 'translation_fixed': True,
            'axis_tilt_fixed': True, 'update_applied': reliable,
            'uncertainty_reason': reason}, proposed_R, applied_R, (crop, proposed_image, applied_image)


def geometric_residual_function(cad, cloud, translation):
    """Recompute the native symmetric, 60%-trimmed fine-fit distance in metres."""
    cad, cloud, translation = np.asarray(cad), np.asarray(cloud), np.asarray(translation)
    if not len(cad) or not len(cloud) or not all(np.isfinite(a).all() for a in (cad, cloud, translation)):
        raise ValueError('nonempty finite clouds and translation required')
    tree = cKDTree(cloud)
    def residual(rotation):
        moved = cad @ rotation.T + translation
        forward = tree.query(moved)[0]
        backward = cKDTree(moved).query(cloud)[0]
        kf, kb = max(1, int(.6 * len(forward))), max(1, int(.6 * len(backward)))
        return float(np.partition(forward, kf - 1)[:kf].mean()
                     + np.partition(backward, kb - 1)[:kb].mean())
    return residual
