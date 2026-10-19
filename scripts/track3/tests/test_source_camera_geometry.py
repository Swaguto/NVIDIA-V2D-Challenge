"""Nonzero wrist pivots, chirality and camera round trips without private MANO assets."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from source_camera_geometry import transform_mano_camera


def mesh(pose, translation, wrist, local, scale, right):
    vertices = np.einsum("ni,ji->nj", local - wrist, Rotation.from_rotvec(pose[:3]).as_matrix()) + wrist
    vertices = (vertices - vertices.mean(0)) * scale + vertices.mean(0)
    vertices *= [1, 1, 1] if right else [-1, 1, 1]
    return vertices + translation


@pytest.mark.parametrize("right", [True, False])
def test_camera_conversion_closes_all_points_with_scaled_nonzero_rest_wrist(right):
    rng = np.random.default_rng(42)
    pose = rng.normal(0, .3, 48)
    translation = np.array([.12, -.08, .45])
    wrist = np.array([.025, -.014, .006])
    local = rng.normal(0, .04, (778, 3))
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_rotvec([.4, -.3, .2]).as_matrix()
    frame[:3, 3] = [.07, .01, -.02]
    original_pose = pose.copy()
    original_frame = frame.copy()
    converted, t = transform_mano_camera(pose, translation, wrist, frame, is_right=right)
    original_vertices = mesh(pose, translation, wrist, local, .83, right)
    expected = np.einsum("ni,ji->nj", original_vertices, frame[:3, :3]) + frame[:3, 3]
    actual = mesh(converted, t, wrist, local, .83, right)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(converted[3:], pose[3:])
    np.testing.assert_array_equal(pose, original_pose)
    np.testing.assert_array_equal(frame, original_frame)
    # A translation-only conversion rotates about the wrong origin.
    wrong_t = np.einsum("ij,j->i", frame[:3, :3], translation) + frame[:3, 3]
    assert np.max(abs(mesh(converted, wrong_t, wrist, local, .83, right) - expected)) > .001
    back, back_t = transform_mano_camera(converted, t, wrist, np.linalg.inv(frame), is_right=right)
    np.testing.assert_allclose(mesh(back, back_t, wrist, local, .83, right), original_vertices, atol=1e-12, rtol=0)


@pytest.mark.parametrize("change", ["reflection", "scale", "nonfinite", "nonhomogeneous", "wrong_pose", "implicit_side"])
def test_rejects_invalid_camera_or_pose_authority(change):
    p = np.zeros(48)
    t = np.zeros(3)
    root = np.zeros(3)
    frame = np.eye(4)
    right = True
    if change == "reflection": frame[0, 0] = -1
    elif change == "scale": frame[0, 0] = 1.1
    elif change == "nonfinite": t[0] = np.nan
    elif change == "nonhomogeneous": frame[3, 0] = .1
    elif change == "wrong_pose": p = p[:3]
    elif change == "implicit_side": right = 1
    with pytest.raises(ValueError):
        transform_mano_camera(p, t, root, frame, is_right=right)
