"""Reject physically invalid apparent retention even with small relative drift."""

import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_native_task import phase_metrics


def fixture():
    hand = np.tile([0, 0, 0, 0, 0, 0, 1.0], (21, 1))
    hand[:, 2] = np.linspace(0, 0.05, len(hand))
    obj = hand.copy()
    obj[:, 0] += 0.1
    return hand, obj, obj.copy(), np.ones((len(hand), 3)) * 2


def test_supported_rigid_carry_passes():
    assert phase_metrics(*fixture())[
        "relative_retention_10mm_5deg_with_sampled_support"
    ]


def test_stationary_hand_and_object_cannot_fake_a_lift():
    hand, obj, ref, force = fixture()
    hand[:] = hand[0]
    obj[:] = obj[0]
    result = phase_metrics(hand, obj, ref, force)
    assert result["maximum_relative_position_drift_m"] == 0
    assert not result["relative_retention_10mm_5deg_with_sampled_support"]


def test_twisting_or_missing_pad_contact_is_rejected():
    hand, obj, ref, force = fixture()
    obj[:, 3:] = Rotation.from_euler(
        "z", np.linspace(0, 20, len(obj)), degrees=True
    ).as_quat()
    assert not phase_metrics(hand, obj, ref, force)[
        "relative_retention_10mm_5deg_with_sampled_support"
    ]
    hand, obj, ref, force = fixture()
    force[:, 1] = 0
    assert not phase_metrics(hand, obj, ref, force)[
        "relative_retention_10mm_5deg_with_sampled_support"
    ]
