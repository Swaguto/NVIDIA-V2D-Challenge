"""Regression tests for noisy-reference pad loss and closure reference snapping."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from _dex3_task_grip import (  # noqa: E402
    TaskGripReference,
    compose_pose,
    denoise_robot_reference,
    relative_pose,
)


def test_bounded_denoising_removes_jitter_without_erasing_linear_motion_or_editing_input():
    times = np.linspace(0, 2, 81)
    trend = np.c_[0.03 * times, times * 0, times * 0]
    jitter = 0.001 * np.cos(2 * np.pi * 10 * times)
    poses = np.c_[
        trend + jitter[:, None] * [0, 1, 0],
        Rotation.from_rotvec(jitter[:, None] * [0, 0, 8]).as_quat(),
    ]
    original = poses.copy()
    filtered = denoise_robot_reference(times, poses)
    np.testing.assert_allclose(filtered[:, 0], trend[:, 0], atol=1e-9)
    assert np.std(filtered[8:-8, 1]) < np.std(poses[8:-8, 1]) / 10
    assert np.std(Rotation.from_quat(filtered[8:-8, 3:]).as_rotvec()[:, 2]) < 0.001
    np.testing.assert_array_equal(poses, original)
    assert np.linalg.norm(filtered[:, :3] - poses[:, :3], axis=1).max() <= 0.003
    # A larger outlier must hit the displacement bound rather than be erased.
    outlier = poses.copy()
    outlier[len(times) // 2, :3] += 0.02
    bounded = denoise_robot_reference(times, outlier)
    assert np.linalg.norm(bounded[:, :3] - outlier[:, :3], axis=1).max() <= 0.003001
    angles = (
        Rotation.from_quat(filtered[:, 3:]) * Rotation.from_quat(poses[:, 3:]).inv()
    ).magnitude()
    assert angles.max() <= np.deg2rad(2)
    signed = poses.copy()
    signed[::2, 3:] *= -1
    equivalent = denoise_robot_reference(times, signed)
    assert (
        Rotation.from_quat(equivalent[:, 3:])
        * Rotation.from_quat(filtered[:, 3:]).inv()
    ).magnitude().max() < 1e-9


def reference():
    phase = dict(
        sample="grasp",
        object=0,
        approach_start_s=1,
        close_s=2,
        release_start_s=4,
        hand_in_object_xyzw=[0.1, 0, 0, 0, 0, 0, 1],
        open_rad=[0] * 7,
        closed_rad=[0.2] * 7,
        effort_times_s=[2, 4],
        effort_Nm=[[0.3] * 7] * 2,
        required_digits=["thumb", "index"],
    )
    return TaskGripReference(
        dict(
            schema="v2d.dex3-task-grips.v1",
            physical_task_accepted=False,
            park_height_m=0.12,
            phases={"left": [phase]},
            experimental_bounded_reacquisition=True,
            experimental_register_settled_grasp=True,
        )
    )


def test_registration_anchors_once_preserves_relative_motion_and_prediction_is_read_only():
    ref = reference()
    original = np.array([0.2, -0.1, 0, 0, 0, 0, 1.0])
    settled = np.r_[
        original[:3] + [0.004, 0, -0.006],
        Rotation.from_euler("x", 3, degrees=True).as_quat(),
    ]
    ref.command("left", 2, original, settled, commit=False)
    assert ref.grasp_registrations == {}
    closing = ref.command("left", 2, original, settled)
    np.testing.assert_allclose(
        closing["pose"], compose_pose(settled, [0.1, 0, 0, 0, 0, 0, 1]), atol=1e-12
    )
    # The settled pose never changes the source clock or subsequent path shape.
    moved = original.copy()
    moved[:3] += [0.04, 0.02, 0.03]
    aligned = compose_pose(settled, relative_pose(moved, original))
    lost = settled.copy()
    lost[2] -= 10
    carry = ref.command("left", 3, moved, lost)
    np.testing.assert_allclose(
        carry["pose"], compose_pose(aligned, [0.1, 0, 0, 0, 0, 0, 1]), atol=1e-12
    )
    assert carry["object_feedback_weight"] == 0
    assert carry["settled_grasp_registration"] == closing["settled_grasp_registration"]
    np.testing.assert_array_equal(original, [0.2, -0.1, 0, 0, 0, 0, 1])


def test_registration_cannot_capture_a_late_or_fallen_object_or_revive_aborted_phase():
    pose = [0, 0, 0, 0, 0, 0, 1]
    ref = reference()
    ref.command("left", 3, pose, pose)
    assert ref.grasp_registrations == {}
    ref = reference()
    ref.command("left", 1.5, pose, [0, 0, -1, 0, 0, 0, 1])
    command = ref.command("left", 2, pose, pose)
    assert command["reacquisition_failure"] is not None
    assert ref.grasp_registrations == {}


def test_compliant_carry_attenuates_pose_mismatch_but_aborts_instead_of_chasing_a_drop():
    ref = reference()
    ref.spec["experimental_carry_compliance"] = True
    obj = np.array([0, 0, 0, 0, 0, 0, 1.0])
    ref.command("left", 2, obj, obj)
    target = obj.copy()
    target[0] = 0.01
    command = ref.command("left", 3, target, obj)
    np.testing.assert_allclose(command["pose"][:3], [0.101, 0, 0], atol=1e-12)
    assert command["object_feedback_weight"] == 0.9
    # The full source error remains available to the failure gate, even though
    # the robot's compliant command applies only part of it across the grasp.
    dropped = obj.copy()
    dropped[2] = -1000
    lost = ref.command("left", 3.1, target, dropped)
    assert lost["reacquisition_failure"] is not None
    assert lost["object_feedback_weight"] == 0
    assert abs(lost["pose"][2]) < 1
    after = ref.command("left", 3.4, target, obj)
    assert after["reacquisition_failure"] is not None
    np.testing.assert_allclose(after["effort"], 0)


def test_measurement_prediction_uses_world_actor_twist_and_real_time():
    from _dex3_task_grip import predict_measured_pose

    pose = np.r_[[1, 2, 3], Rotation.from_euler("x", 0.7).as_quat()]
    predicted = predict_measured_pose(pose, [0.2, -0.1, 0, 0, 0, 2], 0.0025)
    np.testing.assert_allclose(predicted[:3], [1.0005, 1.99975, 3])
    delta = Rotation.from_quat(predicted[3:]) * Rotation.from_quat(pose[3:]).inv()
    np.testing.assert_allclose(delta.as_rotvec(), [0, 0, 0.005], atol=1e-12)
    np.testing.assert_array_equal(pose[:3], [1, 2, 3])
