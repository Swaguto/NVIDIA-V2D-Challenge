"""Appearance polarity, ambiguous spin and exported-pose residual controls."""
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/track3'))
from lid_spin import LidTemplate, geometric_residual_function, spin_rotation, sweep_appearance


def scene(*, symmetric=False, constant=False, invert=False):
    gray = np.full((128, 128), 180., np.float32)
    if not constant:
        markers = ([(40, 64, 7), (64, 40, 7), (88, 64, 7), (64, 88, 7)] if symmetric else
                   [(23, 46, 8), (48, 20, 5), (94, 35, 7), (93, 93, 9), (41, 103, 4)])
        for x, y, radius in markers:
            cv2.circle(gray, (x, y), radius, 20., -1)
            cv2.circle(gray, (x, y), radius - 2, 245., -1)
    yy, xx = np.indices(gray.shape)
    radius = np.hypot(xx - 64, yy - 64)
    template = LidTemplate(gray, (radius > 12) & (radius < 54), np.array([0., 0., -1.]),
                           np.array([1., 0., 0.]), np.array([0., -1., 0.]),
                           np.array([-.064, .064, 0.]), 1000.)
    K = np.array([[1000., 0., 110.], [0., 1000., 100.], [0., 0., 1.]])
    translation = np.array([0., 0., 1.])
    angle = 0. if symmetric or constant else 35.
    R = spin_rotation(np.eye(3), template.axis, angle)
    H = K @ np.column_stack([R @ template.tangent1 / template.scale,
                             R @ template.tangent2 / template.scale,
                             R @ template.origin + translation])
    observed = np.uint8(cv2.warpPerspective(gray, H, (220, 200)))
    if invert:
        observed = 255 - observed
    mask = cv2.warpPerspective(template.roi.astype('float32'), H, (220, 200)) > .99
    image = cv2.cvtColor(observed, cv2.COLOR_GRAY2BGR)
    return template, image, mask, K, np.eye(3), translation


def test_inverted_marker_brightness_retains_gradient_spin_but_breaks_gray_matching():
    inputs = scene(invert=True)
    gradient, proposed, applied, _ = sweep_appearance(*inputs, method='gradient_ncc')
    grayscale, *_ = sweep_appearance(*inputs, method='highpass_ncc')
    assert gradient['update_applied']
    assert abs(gradient['selected']['angle_deg'] - 35) <= .5
    assert gradient['selected']['cost'] < .01
    wrong_polarity_at_truth = next(x for x in grayscale['coarse_curve'] if x['angle_deg'] == 35)
    assert wrong_polarity_at_truth['cost'] > 1.9
    assert np.allclose(proposed, applied)
    assert np.allclose(applied @ inputs[0].axis, inputs[4] @ inputs[0].axis)


def test_repeated_markers_and_untextured_lid_do_not_claim_observable_spin():
    for inputs in [scene(symmetric=True), scene(constant=True)]:
        diagnostic, _, applied, _ = sweep_appearance(*inputs)
        assert not diagnostic['update_applied']
        assert diagnostic['uncertainty_reason'] is not None
        assert diagnostic['separated_peak_cost_gap'] < .1
        assert np.array_equal(applied, inputs[4])


def test_marker_face_on_backside_cannot_authorize_an_appearance_update():
    inputs = list(scene())
    template = inputs[0]
    # Same homography/texture match, but the designated CAD face points away.
    inputs[0] = LidTemplate(template.gray, template.roi, -template.axis,
                            template.tangent1, template.tangent2, template.origin, template.scale)
    diagnostic, _, applied, _ = sweep_appearance(*inputs)
    assert not diagnostic['update_applied']
    assert diagnostic['uncertainty_reason'] == 'marker_face_not_visible'
    assert np.array_equal(applied, inputs[4])


def test_tilted_axis_spin_requires_fresh_geometric_residual_of_actual_pose():
    axis = np.array([.98, .1, -.15]); axis /= np.linalg.norm(axis)
    baseline = Rotation.from_euler('xyz', [20, -35, 13], degrees=True).as_matrix()
    spun = spin_rotation(baseline, axis, 80)
    assert np.allclose(spun @ axis, baseline @ axis, atol=1e-15)
    assert np.allclose(spun.T @ spun, np.eye(3), atol=1e-15)
    rng = np.random.default_rng(4)
    cad = rng.normal(size=(100, 3)) * [.015, .03, .04] + [0., .15, .05]
    translation = np.array([.1, -.2, .7])
    cloud = cad @ baseline.T + translation
    residual = geometric_residual_function(cad, cloud, translation)
    assert residual(baseline) < 1e-12
    assert residual(spun) > .05
