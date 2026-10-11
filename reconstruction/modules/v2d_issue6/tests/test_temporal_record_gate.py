"""Regressions for invalid native estimates without hiding calibration errors."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "validation/issue6"))
from temporal_record_gate import rejection_reason

CALIBRATION = {
    "convention": "rectified_pinhole",
    "width": 1280,
    "height": 800,
    "fx": 565.5896606445312,
    "fy": 565.5896606445312,
    "cx": 612.2412719726562,
    "cy": 391.17578125,
}
RECORD = {
    "camera": {
        "scaled_focal_length": 565.5896606445312,
        "scaled_focal_length_y": 565.5896606445312,
        "principal_point": [612.2412719726562, 391.17578125],
        "pred_cam_t_full": [0.09005289236273593, -0.03468720638038432, -0.08633542116510377],
    }
}


def test_actual_episode43_behind_camera_estimate_cannot_be_a_measured_observation():
    original = copy.deepcopy(RECORD)
    assert rejection_reason(RECORD, True, CALIBRATION) == "invalid_virtual_camera"
    assert RECORD == original  # Preserve the raw optimizer result for review.
    supported = copy.deepcopy(RECORD)
    supported["camera"]["pred_cam_t_full"][2] = 0.45
    assert rejection_reason(supported, True, CALIBRATION) is None


@pytest.mark.parametrize("depth", [0.0, float("nan")])
def test_zero_or_nonfinite_depth_stays_unknown(depth):
    record = copy.deepcopy(RECORD)
    record["camera"]["pred_cam_t_full"][2] = depth
    assert rejection_reason(record, True, CALIBRATION) == "invalid_virtual_camera"


def test_recovered_camera_ray_does_not_invent_direct_observation():
    assert rejection_reason({}, False, CALIBRATION) == "not_directly_observed"


def test_invalid_calibration_is_still_a_fatal_configuration_error():
    with pytest.raises(ValueError, match="camera intrinsics"):
        rejection_reason(RECORD, True, {**CALIBRATION, "fx": 0.0})
