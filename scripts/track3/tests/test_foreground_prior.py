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


def test_recovery_candidate_uses_the_original_jump_reference():
    from foundationpose_track import pose_rejection

    previous = np.eye(4)
    previous[2, 3] = 0.5
    candidate = previous.copy()
    candidate[0, 3] = 0.2
    args = (None, None, None, None, None, False, previous)
    rejection, _ = pose_rejection(candidate, *args)
    assert rejection["error"] == "implausible pose jump"
    candidate[0, 3] = 0.1
    rejection, _ = pose_rejection(candidate, *args)
    assert rejection is None


def test_depth_support_is_rechecked_after_recovery():
    from foundationpose_track import pose_rejection

    depth = np.ones((30, 30)) * 0.5
    valid = np.ones_like(depth, bool)
    mask = np.ones(depth.shape, np.uint8) * 255
    K = np.array([[100.0, 0, 15], [0, 100.0, 15], [0, 0, 1]])
    bounds = np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]])
    candidate = np.eye(4)
    candidate[2, 3] = 0.5
    rejection, support = pose_rejection(
        candidate, depth, valid, mask, K, bounds, True, None
    )
    assert rejection is None and support["accepted"]
    candidate[0, 3] = 0.5
    rejection, support = pose_rejection(
        candidate, depth, valid, mask, K, bounds, True, None
    )
    assert rejection["error"] == "insufficient observed depth support"
    assert not support["accepted"]
