"""Visible depth shifts initialize refinement without changing the acceptance pose."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from foreground_prior import measured_surface, translated_prior


def test_surface_is_measured_foreground_only_and_missing_depth_stays_unknown():
    depth = np.ones((20, 20)) * 0.5
    depth[:, :10] = 2.0
    mask = np.zeros(depth.shape, np.uint8)
    mask[:, 10:] = 255
    valid = np.ones_like(mask, bool)
    K = np.array([[100, 0, 10], [0, 100, 10], [0, 0, 1]])
    center = measured_surface(depth, valid, mask, K)
    np.testing.assert_allclose(center, [0.0225, -0.0025, 0.5])
    valid[:, 10:] = False
    assert measured_surface(depth, valid, mask, K) is None


def test_initialization_preserves_body_gauge_and_original_jump_gate_reference():
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    original = pose.copy()
    candidate = translated_prior(pose, [2, 3, 4], [2.03, 3.01, 3.95])
    np.testing.assert_allclose(candidate[:3, 3], [1.03, 2.01, 2.95])
    np.testing.assert_array_equal(pose, original)
    np.testing.assert_array_equal(candidate[:3, :3], pose[:3, :3])


def test_static_world_surface_does_not_add_camera_motion_twice():
    transport = np.eye(4)
    transport[:3, 3] = [0.1, -0.2, 0.05]
    pose = transport.copy()
    previous = np.array([0.2, 0.3, 0.4])
    current = previous + transport[:3, 3]
    candidate = translated_prior(pose, previous, current, transport)
    np.testing.assert_allclose(candidate, pose, atol=1e-15)
