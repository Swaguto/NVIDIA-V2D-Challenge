"""Control interpolation must preserve source time, knots and unknown labels."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.sampling import resample_targets


def source():
    poses = np.zeros((3, 1, 7))
    poses[:, 0, 0] = [0, 1, 2]
    poses[:, 0, 3:7] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, -1]]
    return dict(
        frame_indices=np.arange(3),
        timestamps_s=np.array([0, 0.05, 0.1]),
        confidence_valid=np.array([True, False, True]),
        robot_qpos=poses,
        phase_index=np.array([[0], [1], [2]]),
        capture_settled_registration=np.array([[0], [1], [0]]),
        finger_feedforward_Nm=np.array([[0.1], [0.3], [0.5]]),
    )


def test_source_knots_and_duration_survive_upsampling():
    data = source()
    result = resample_targets(data, 11)
    for key, value in data.items():
        np.testing.assert_array_equal(result[key][::5], value)
    assert result["timestamps_s"][-1] == 0.1
    assert not result["confidence_valid"][1:-1].any()
    assert result["phase_index"][4, 0] == 0
    assert result["phase_index"][5, 0] == 1
    assert result["capture_settled_registration"][9, 0] == 1
    assert result["finger_feedforward_Nm"].max() == 0.5
    # Source arrays are evidence and are never changed in place.
    np.testing.assert_array_equal(data["robot_qpos"], source()["robot_qpos"])


def test_rotation_uses_shortest_arc_and_keeps_unit_quaternions():
    result = resample_targets(source(), 5)["robot_qpos"]
    np.testing.assert_allclose(result[1, 0, 3:7], [0, 0, 2**-0.5, 2**-0.5])
    np.testing.assert_allclose(np.linalg.norm(result[..., 3:7], axis=-1), 1)
    data = source()
    data["robot_qpos"][:, 0, 3:7] = [[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 1]]
    result = resample_targets(data, 11)["robot_qpos"]
    np.testing.assert_allclose(abs(result[:, 0, 6]), 1)


def test_downsampling_and_partial_intervals_are_rejected():
    for count in (1, 2, 4):
        with pytest.raises(ValueError, match="integer command upsampling"):
            resample_targets(source(), count)
