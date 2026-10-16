"""Physical-clock noise readback must not depend on quaternion signs."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_wrist_jitter import motion_residual


def test_motion_residual_preserves_physical_frequency_and_quaternion_equivalence():
    t = np.arange(0, 4, 0.0025)
    noise = np.sin(2 * np.pi * 12 * t)
    poses = np.c_[
        0.02 * t,
        0.002 * noise,
        t * 0,
        Rotation.from_rotvec(np.c_[t * 0, t * 0, 0.03 * noise]).as_quat(),
    ]
    result = motion_residual(t, poses)
    assert result["position_residual_rms_mm"] == pytest.approx(np.sqrt(2), abs=0.03)
    assert result["rotation_residual_rms_deg"] == pytest.approx(
        np.rad2deg(0.03 / np.sqrt(2)), abs=0.03
    )
    signed = poses.copy()
    signed[::2, 3:] *= -1
    assert motion_residual(t, signed) == result
    smooth = np.c_[
        0.02 * t,
        t * 0,
        t * 0,
        Rotation.from_rotvec(np.c_[t * 0, t * 0, 0.03 * t]).as_quat(),
    ]
    assert motion_residual(t, smooth)["position_residual_rms_mm"] < 0.01
    assert motion_residual(t, smooth)["rotation_residual_rms_deg"] < 0.01


def test_motion_residual_rejects_gaps_instead_of_interpolating_them():
    t = np.arange(0, 1, 0.0025)
    pose = np.tile([0, 0, 0, 0, 0, 0, 1], (len(t), 1))
    t[100:] += 0.05
    with pytest.raises(ValueError, match="clock"):
        motion_residual(t, pose)
