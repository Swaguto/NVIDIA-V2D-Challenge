import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from _dex3_task_grip import SmoothObjectReference  # noqa: E402


def test_reference_preserves_knots_and_has_no_interior_velocity_jump():
    times = np.array([0, 0.1, 0.2, 0.3, 0.4])
    xyz = np.column_stack((0.04 * np.sin(times * 10), times * 0, times * 0))
    angles = np.column_stack((times * 0.2, times**2, np.sin(times) * 0.3))
    poses = np.column_stack((xyz, Rotation.from_euler("xyz", angles).as_quat()))
    before = poses.copy()
    ref = SmoothObjectReference(times, poses)
    for t, pose in zip(times, poses, strict=True):
        np.testing.assert_allclose(ref.pose(t)[:3], pose[:3], atol=1e-12)
        assert (
            Rotation.from_quat(ref.pose(t)[3:]) * Rotation.from_quat(pose[3:]).inv()
        ).magnitude() < 1e-12
    for t in times[1:-1]:
        np.testing.assert_allclose(
            ref.position(t - 1e-8, 1), ref.position(t + 1e-8, 1), atol=1e-6
        )
        np.testing.assert_allclose(
            ref.rotation(t - 1e-8, 1), ref.rotation(t + 1e-8, 1), atol=1e-6
        )
        np.testing.assert_allclose(
            ref.load_wrench(t - 1e-8, 0.5), ref.load_wrench(t + 1e-8, 0.5), atol=1e-5
        )
    np.testing.assert_array_equal(poses, before)
    np.testing.assert_allclose(ref.pose(-1), ref.pose(0))
    np.testing.assert_allclose(ref.pose(1), ref.pose(0.4))


def test_angular_derivatives_are_rotated_from_body_to_world_and_speed_squared():
    t = np.linspace(0, 1, 11)
    r = Rotation.from_euler("xyz", np.column_stack((t**2, t * 0.4, t**2 * 0.5)))
    ref = SmoothObjectReference(
        t, np.column_stack((np.zeros((len(t), 3)), r.as_quat()))
    )
    q = 0.45
    h = 1e-5
    omega_world = (ref.rotation(q + h) * ref.rotation(q - h).inv()).as_rotvec() / (
        2 * h
    )
    np.testing.assert_allclose(
        omega_world, ref.rotation(q).apply(ref.rotation(q, 1)), atol=1e-8
    )
    inertia = np.diag([0.001, 0.002, 0.003])
    full = ref.load_wrench(q, 1, inertia_body=inertia)
    slow = ref.load_wrench(q, 0.5, inertia_body=inertia)
    np.testing.assert_allclose(slow[3:], full[3:] / 4, atol=1e-12)
    np.testing.assert_allclose(full[:3], [0, 0, 2.943], atol=1e-12)
    np.testing.assert_allclose(ref.load_wrench(2, 0.5), [0, 0, 2.943, 0, 0, 0])
    with pytest.raises(ValueError, match="mass/inertia"):
        ref.load_wrench(q, 0.5, inertia_body=-inertia)


def test_nominal_dynamic_force_obeys_newtons_law_with_physical_speed():
    times = np.linspace(0, 1, 11)
    # This cubic has zero endpoint velocity, so the clamped interpolant is exact.
    x = 0.1 * (3 * times**2 - 2 * times**3)
    poses = np.column_stack(
        (x, np.zeros((len(times), 2)), np.tile([0, 0, 0, 1], (len(times), 1)))
    )
    ref = SmoothObjectReference(times, poses)
    # x''(0.25) = 0.3 source m/s²; half speed reduces acceleration by four.
    np.testing.assert_allclose(
        ref.load_wrench(0.25, 0.5), [0.3 * 0.3 / 4, 0, 2.943, 0, 0, 0], atol=1e-10
    )


def test_lost_object_cannot_pull_reacquisition_hand_out_of_scene_or_close_empty_grasp():
    from _dex3_task_grip import TaskGripReference

    phase = dict(
        sample="first",
        object=0,
        approach_start_s=1,
        close_s=2,
        release_start_s=3,
        hand_in_object_xyzw=[0, 0, 0.1, 0, 0, 0, 1],
        open_rad=[0] * 7,
        closed_rad=[0.2] * 7,
        effort_times_s=[2, 3],
        effort_Nm=[[0.3] * 7] * 2,
        required_digits=["thumb", "index"],
    )
    second = dict(
        phase, sample="second", approach_start_s=4, close_s=5, release_start_s=6
    )
    ref = TaskGripReference(
        dict(
            schema="v2d.dex3-task-grips.v1",
            physical_task_accepted=False,
            park_height_m=0.12,
            phases={"left": [phase, second]},
            experimental_bounded_reacquisition=True,
        )
    )
    original = np.array([0, 0, 0, 0, 0, 0, 1.0], float)
    lost = original.copy()
    lost[2] = -1000
    preview = ref.command("left", 1.01, original, lost, commit=False)
    assert preview["reacquisition_failure"] is not None
    assert ref.aborted_phases == {}
    ref.command("left", 1.01, original, lost)
    closed = ref.command("left", 2, original, original)
    assert closed["reacquisition_failure"] is not None  # failed phase stays latched
    assert closed["object_feedback_weight"] == 0
    np.testing.assert_allclose(closed["pose"][:3], [0, 0, 0.22])
    np.testing.assert_allclose(closed["fingers"], 0)
    np.testing.assert_allclose(closed["effort"], 0)
    recovered = ref.command("left", 5, original, original)
    assert recovered["reacquisition_failure"] is None
    np.testing.assert_allclose(recovered["fingers"], 0.2)
    np.testing.assert_array_equal(original, [0, 0, 0, 0, 0, 0, 1])


def test_aborted_hand_holds_measured_pose_instead_of_following_source_or_fallen_object():
    from _dex3_task_grip import aborted_hand_hold

    anchor = np.array([0.1, 0.2, 0.3, 0, 0, 0, 1.0])
    initial = np.ones(7) * 0.2
    opened = np.zeros(7)
    pose, start = aborted_hand_hold(anchor, initial, opened, 0)
    np.testing.assert_array_equal(pose, anchor)
    np.testing.assert_array_equal(start, initial)
    pose, end = aborted_hand_hold(anchor, initial, opened, 1)
    np.testing.assert_array_equal(pose, anchor)
    np.testing.assert_array_equal(end, opened)
    np.testing.assert_array_equal(initial, np.ones(7) * 0.2)
