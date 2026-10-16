"""Payload compensation preserves clock, world/child frames and phase gating."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[2] / "reproduction"))
from v2d_dex3_flashchord.payload import payload_plan, child_moment


def test_static_payload_lift_is_robot_force_with_correct_lever_sign():
    times = np.linspace(0, 5, 101)
    poses = np.tile([0, 0, 0, 0, 0, 0, 1.0], (len(times), 1))
    phases = [dict(close_s=1.0, release_start_s=4.0)]
    wrench = payload_plan(times, poses, phases, 0.3, np.eye(3) * 0.001, [0, 0, 0])
    np.testing.assert_allclose(wrench[40], [0, 0, 2.943, 0, 0, 0])
    assert not wrench[[0, -1]].any()
    # A 90 degree wrist about Z rotates world X moment to negative child Y.
    q = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    moment = child_moment(
        wrench[40, :3], wrench[40, 3:], np.array([0, 0.2, 0]), np.zeros(3), q
    )
    np.testing.assert_allclose(moment, [0, -0.5886, 0], atol=1e-12)


def test_payload_acceleration_uses_full_source_speed_and_gain_is_bounded():
    times = np.linspace(0, 5, 101)
    poses = np.tile([0, 0, 0, 0, 0, 0, 1.0], (len(times), 1))
    poses[:, 0] = 0.05 * times**2
    phases = [dict(close_s=0, release_start_s=5)]
    args = (times, poses, phases, 0.3, np.eye(3) * 0.001, [0, 0, 0])
    full = payload_plan(*args)
    np.testing.assert_allclose(full[50, 0], 0.03, atol=1e-9)
    np.testing.assert_allclose(payload_plan(*args, gain=0.5), 0.5 * full)
    with pytest.raises(ValueError, match="gain"):
        payload_plan(*args, gain=1.01)
