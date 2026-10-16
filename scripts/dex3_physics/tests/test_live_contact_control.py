import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from live_contact_control import contact_patches, motor_effort  # noqa: E402


def test_live_patches_select_the_primary_object_and_ignore_unused_capacity():
    forces = np.array([np.nan, 2, 4, 999, 3, np.nan])[:, None]
    points = np.array(
        [[99, 99, 99], [1, 0, 0], [3, 0, 0], [99, 99, 99], [0, 2, 0], [99, 99, 99]]
    )
    normals = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0]]
    )
    data = [
        forces,
        points,
        normals,
        np.zeros((6, 1)),
        np.array([[1, 2], [0, 1]]),
        np.array([[3, 1], [99, 4]]),
    ]
    centers, inward = contact_patches(data, [0, 1], 1)
    np.testing.assert_allclose(centers, [[7 / 3, 0, 0], [0, 2, 0]])
    np.testing.assert_allclose(inward, [[-1, 0, 0], [0, -1, 0]])
    shifted = list(data)
    shifted[1] = points + [4, -2, 3]
    moved, directions = contact_patches(shifted, [0, 1], 1)
    np.testing.assert_allclose(moved, centers + [4, -2, 3])
    np.testing.assert_allclose(directions, inward)
    missing = list(data)
    missing[-2] = np.zeros((2, 2), dtype=int)
    with pytest.raises(ValueError, match="missing"):
        contact_patches(missing, [0, 1], 1)


def test_live_effort_adds_finger_gravity_with_correct_sign_and_preserves_real_limit():
    points = np.array([[-0.02, 0, 0], [0.02, 0, 0]])
    inward = np.array([[1, 0, 0], [-1, 0, 0]])
    jac = np.zeros((2, 3, 7))
    jac[0, :, :3] = np.eye(3) * 0.01
    jac[1, :, 3:6] = np.eye(3) * 0.01
    before, _ = motor_effort(points, inward, jac, np.zeros(3), np.zeros(7))
    gravity = np.linspace(-0.05, 0.05, 7)
    after, receipt = motor_effort(points, inward, jac, np.zeros(3), gravity)
    np.testing.assert_allclose(after - before, gravity)
    np.testing.assert_allclose(receipt["joint_gravity_compensation_Nm"], gravity)
    assert np.max(abs(after)) < 0.76
    with pytest.raises(ValueError, match="original motor limit"):
        motor_effort(points, inward, jac, np.zeros(3), np.ones(7))
