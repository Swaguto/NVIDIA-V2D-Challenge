"""Preserve observable rigid-object features while rejecting isolated depth noise."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/track3"))
import register_poses as registration


def bowl_with_handle_and_noise():
    xy = np.mgrid[-25:26, -25:81].reshape(2, -1).T * .004
    bowl = (xy ** 2).sum(1) <= .1 ** 2
    handle = (abs(xy[:, 0]) <= .016) & (xy[:, 1] >= .08)
    surface = np.column_stack([xy[bowl | handle], np.zeros((bowl | handle).sum())])
    noise = np.array([[.8 + i * .002, .8, .8] for i in range(5)])
    return np.concatenate([surface, noise]), len(surface)


def test_filter_preserves_connected_handle_and_rejects_flying_points():
    points, surface_count = bowl_with_handle_and_noise()
    output = registration.trim_cloud(points)
    # Far from the bowl center is not evidence that its handle is an outlier.
    assert output[:, 1].max() >= .30
    assert np.count_nonzero(output[:, 1] > .14) >= .9 * np.count_nonzero(points[:surface_count, 1] > .14)
    assert np.all(output[:, 2] == 0)
    assert len(output) >= .9 * surface_count


def test_cloud_filter_is_rigid_transform_invariant():
    points, _ = bowl_with_handle_and_noise()
    rotation = Rotation.from_euler("xyz", [37, -61, 19], degrees=True).as_matrix()
    translation = np.array([.17, -.3, .8])
    expected = registration.trim_cloud(points) @ rotation.T + translation
    actual = registration.trim_cloud(points @ rotation.T + translation)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_small_cloud_is_preserved_without_mutation():
    points = np.arange(48, dtype=float).reshape(16, 3)
    before = points.copy()
    np.testing.assert_array_equal(registration.trim_cloud(points), before)
    np.testing.assert_array_equal(points, before)
