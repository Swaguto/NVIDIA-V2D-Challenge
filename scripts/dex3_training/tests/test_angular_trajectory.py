from dataclasses import make_dataclass
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.angular_trajectory import reference_angular_velocity
from v2d_dex3_flashchord.phase_prior import control_signature


def test_angular_reference_uses_world_axes_and_is_quaternion_sign_invariant():
    times = np.arange(8) * .05
    initial = Rotation.from_euler('z', 90, degrees=True)
    q = (initial * Rotation.from_rotvec(np.c_[2 * times, times * 0, times * 0])).as_quat()
    omega = reference_angular_velocity(times, q)
    np.testing.assert_allclose(omega[:-1], np.tile([0, 2, 0], (7, 1)), atol=1e-12)
    q[::2] *= -1
    np.testing.assert_allclose(reference_angular_velocity(times, q), omega, atol=1e-12)
    np.testing.assert_array_equal(omega[-1], 0)


def test_invalid_clock_or_excess_speed_fails_without_silent_clipping():
    times = np.arange(5) * .05
    q = Rotation.from_euler('x', times * 4).as_quat()
    with pytest.raises(ValueError, match='exceeds'):
        reference_angular_velocity(times, q, maximum_speed=2)
    with pytest.raises(ValueError):
        reference_angular_velocity(times[::-1], q)


def test_disabled_servo_preserves_archived_controller_hashes():
    old = make_dataclass('ExampleAction', [('kp', float)])
    new = make_dataclass('ExampleAction', [('kp', float),
        ('wrist_angular_reference_velocity', bool, False),
        ('wrist_angular_kd', float, .4),
        ('maximum_reference_wrist_angular_speed_rad_s', float, 10.)])
    assert control_signature(old(12.)) == control_signature(new(12.))
    assert control_signature(old(12.)) != control_signature(new(12., True))
