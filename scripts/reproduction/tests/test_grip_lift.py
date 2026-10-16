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
