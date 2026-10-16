import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _dex3_grip_control import (
    lift_summary,
    lift_target,
    payload_gravity_wrench,
    payload_contact_hold,
    static_load_schedule,
)


def test_lift_reference_is_stationary_until_support_release_and_bounded():
    initial = np.array([1, 2, 3, 0, 0, 0, 1])
    poses = np.array([lift_target(initial, t) for t in np.linspace(0, 1.5, 151)])
    np.testing.assert_allclose(poses[:51], np.broadcast_to(initial, (51, 7)))
    assert poses[:, 2].max() == pytest.approx(3.05)
    assert poses[-1, 2] == pytest.approx(3.05)
    np.testing.assert_array_equal(initial, [1, 2, 3, 0, 0, 0, 1])


def test_payload_compensation_includes_the_correct_lever_arm_torque():
    wrench = payload_gravity_wrench([0, 0, 0], [0.1, 0.2, -0.05], 0.3)
    np.testing.assert_allclose(wrench, [0, 0, 2.943, 0.5886, -0.2943, 0])
    np.testing.assert_allclose(
        wrench, payload_gravity_wrench([3, 1, -2], [3.1, 1.2, -2.05], 0.3)
    )


def test_payload_compensation_uses_two_digit_contacts_and_expires_after_loss():
    paths = ["right_hand_thumb_2_link", "right_hand_index_1_link"]
    forces = [[1, 0, 0], [-1, 0, 0]]
    remaining = payload_contact_hold(forces, paths, 0, 0.01)
    assert remaining == 0.03
    for _ in range(4):
        remaining = payload_contact_hold([[1, 0, 0], [0, 0, 0]], paths, remaining, 0.01)
    assert remaining == 0


def test_three_pad_payload_compensation_requires_the_middle_contact_too():
    paths = [
        "left_hand_thumb_2_link",
        "left_hand_index_1_link",
        "left_hand_middle_1_link",
    ]
    assert (
        payload_contact_hold(
            [[1, 0, 0], [-1, 0, 0], [0, 0, 0]],
            paths,
            0,
            0.01,
            ("thumb", "index", "middle"),
        )
        == 0
    )
    assert (
        payload_contact_hold(
            [[1, 0, 0], [-1, 0, 0], [1, 0, 0]],
            paths,
            0,
            0.01,
            ("thumb", "index", "middle"),
        )
        == 0.03
    )


def test_lift_gate_requires_actual_lift_retention_and_wrist_tracking():
    rows = []
    for step in range(151):
        pose = lift_target([0, 0, 0.1, 0, 0, 0, 1], step / 100)
        obj = pose.copy()
        obj[2] -= 0.03
        rows.append(
            dict(
                step=step,
                actual_wrist_pose_stage_xyzw=pose.tolist(),
                target_wrist_pose_stage_xyzw=pose.tolist(),
                object_root_state_stage_xyzw=obj.tolist(),
            )
        )
    assert lift_summary(rows)["lifted_and_retained_diagnostic_only"]
    rows[-1]["object_root_state_stage_xyzw"][0] += 0.012
    assert not lift_summary(rows)["lifted_and_retained_diagnostic_only"]
    rows[-1]["object_root_state_stage_xyzw"][0] -= 0.012
    rows[-1]["actual_wrist_pose_stage_xyzw"][0] += 0.03
    assert not lift_summary(rows)["lifted_and_retained_diagnostic_only"]


@pytest.mark.parametrize("dt", [0.01, 0.0025])
def test_finer_physics_preserves_fixture_events_and_observation_clock(dt):
    schedule = static_load_schedule(dt, True)
    assert schedule["load_start_step"] * dt == pytest.approx(0.5)
    assert schedule["closure_steps"] * dt == pytest.approx(0.25)
    assert schedule["averaging_steps"] * dt == pytest.approx(0.2)
    times = np.arange(0, schedule["last_step"] + 1, schedule["render_steps"]) * dt
    np.testing.assert_allclose(times, np.linspace(0, 1.5, 31), atol=1e-12)
    rows = []
    for step in range(schedule["last_step"] + 1):
        t = step * dt
        pose = lift_target([0, 0, 0.1, 0, 0, 0, 1], t)
        obj = pose.copy()
        obj[2] -= 0.03
        if t < 0.5:
            obj[0] += 2.0  # The supported approach is outside the lift metric.
        rows.append(
            dict(
                step=step,
                time_s=t,
                actual_wrist_pose_stage_xyzw=pose.tolist(),
                target_wrist_pose_stage_xyzw=pose.tolist(),
                object_root_state_stage_xyzw=obj.tolist(),
            )
        )
    assert lift_summary(rows)["lifted_and_retained_diagnostic_only"]


def test_static_physics_rejects_timestep_that_moves_fixture_events():
    with pytest.raises(ValueError, match="event times"):
        static_load_schedule(0.003, True)
