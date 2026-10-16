import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from live_contact_control import (  # noqa: E402
    contact_patches,
    motor_effort,
    patch_readback,
)


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
    # Reversing both signed force and normal preserves the physical force.
    # Dropping negative scalars would falsely reject a fully loaded patch.
    reversed_data = list(data)
    reversed_data[0] = -forces
    reversed_data[2] = -normals
    reversed_centers, reversed_inward = contact_patches(reversed_data, [0, 1], 1)
    np.testing.assert_allclose(reversed_centers, centers)
    np.testing.assert_allclose(reversed_inward, inward)
    evidence = patch_readback(reversed_data, [0, 1], 1)
    assert evidence[0]["signed_scalar_min_N"] == -4
    assert evidence[0]["scalar_magnitude_sum_N"] == 6
    np.testing.assert_allclose(evidence[0]["resultant_force_on_hand_N"], [6, 0, 0])
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


def test_accelerating_payload_receives_force_and_moment_without_changing_motor_caps():
    from grip_force import solve_gravity_contacts

    points = np.array([[-0.02, 0, 0], [0.02, 0, 0]])
    inward = np.array([[1, 0, 0], [-1, 0, 0]])
    jac = np.zeros((2, 3, 7))
    jac[0, :, :3] = np.eye(3) * 0.01
    jac[1, :, 3:6] = np.eye(3) * 0.01
    desired = np.array([0.3, 0.1, 3.543, 0, 0.01, 0.005])
    forces, torque, plan = solve_gravity_contacts(
        points, inward, jac, np.zeros(3), 0.3, desired_wrench_stage=desired
    )
    np.testing.assert_allclose(
        np.r_[forces.sum(0), np.cross(points, forces).sum(0)], desired, atol=1e-9
    )
    assert max(abs(torque)) <= 0.55
    np.testing.assert_allclose(plan["desired_object_wrench_stage"], desired)
    with pytest.raises(ValueError, match="six world"):
        solve_gravity_contacts(
            points, inward, jac, np.zeros(3), 0.3, desired_wrench_stage=[1, 2]
        )


def test_weak_contact_geometry_can_remain_usable_below_payload_support_threshold():
    data = [
        np.array([[-0.1], [-0.2]]),
        np.array([[0.0, 0, 0], [1.0, 0, 0]]),
        np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
        np.zeros((2, 1)),
        np.ones((2, 1), int),
        np.array([[0], [1]]),
    ]
    with pytest.raises(ValueError, match="insufficient"):
        contact_patches(data, [0, 1], 0)
    points, inward = contact_patches(data, [0, 1], 0, minimum_force_N=0.05)
    np.testing.assert_allclose(points, data[1])
    np.testing.assert_allclose(inward, [[1, 0, 0], [-1, 0, 0]])


def test_hybrid_posture_keeps_spring_without_opposing_contact_normal_torque():
    from live_contact_control import normal_nullspace_posture

    jac = (
        np.array(
            [[1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0.0]]
        )
        * 0.02
    )
    current = np.zeros(7)
    reference = np.array([0.2, -0.1, 0.1, 0.2, -0.1, 0.2, 0.3])
    kp = np.arange(1, 8, dtype=float)
    target, info = normal_nullspace_posture(
        reference, current, jac, kp, np.tile([-1.0, 1], (7, 1))
    )
    assert np.linalg.norm(target - current) > 0.1  # retains posture unlike pure torque
    assert (
        np.linalg.norm(target - reference) > 0.01
    )  # releases normal-axis position opposition
    np.testing.assert_allclose(jac @ (kp * (target - current)), 0, atol=1e-12)
    np.testing.assert_allclose(info["normal_torque_projection_residual"], 0, atol=1e-12)
    with pytest.raises(ValueError, match="rank deficient"):
        normal_nullspace_posture(
            reference, current, jac * 0, kp, np.tile([-1.0, 1], (7, 1))
        )
    with pytest.raises(ValueError, match="joint limits"):
        normal_nullspace_posture(
            reference, current, jac, kp, np.tile([-0.001, 0.001], (7, 1))
        )
