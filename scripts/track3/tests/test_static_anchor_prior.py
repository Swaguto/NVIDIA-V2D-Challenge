import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from static_anchor_prior import mask_stillness


def observations():
    mask = np.zeros((100, 100), bool)
    mask[30:70, 30:70] = True
    depth = np.full(mask.shape, 0.5)
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1.0]])
    return depth, np.ones_like(mask), mask, K


def test_independent_camera_motion_is_not_mistaken_for_object_motion():
    d, v, mask, K = observations()
    current = np.roll(mask, 8, axis=1)
    delta = np.eye(4)
    delta[0, 3] = 0.04
    assert mask_stillness(d, v, mask, current, K, delta)["eligible"]
    assert not mask_stillness(d, v, mask, current, K, np.eye(4))["eligible"]


def test_missing_depth_and_large_occlusion_do_not_establish_stillness():
    d, v, mask, K = observations()
    r = mask_stillness(d, v & False, mask, mask, K, np.eye(4))
    assert not r["eligible"] and not r["missing_observations_filled"]
    cropped = mask.copy()
    cropped[:, :50] = False
    assert not mask_stillness(d, v, mask, cropped, K, np.eye(4))["eligible"]
    bad = np.eye(4)
    bad[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        mask_stillness(d, v, mask, mask, K, bad)


def test_stereo_holes_are_not_interpreted_as_new_silhouette_edges():
    d, v, mask, K = observations()
    v[40:60, 40:60] = False
    original = v.copy()
    r = mask_stillness(d, v, mask, mask, K, np.eye(4))
    assert r["eligible"] and r["inferred_projection_depth_pixels"] == 400
    assert r["projection_is_not_measured_depth_completion"]
    np.testing.assert_array_equal(v, original)
