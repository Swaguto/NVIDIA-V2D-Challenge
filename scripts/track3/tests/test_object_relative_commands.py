"""An occluded hold follows object motion without forging source observations."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from object_relative_commands import infer_object_relative_gap


def fixture():
    times = np.linspace(0, 1, 7)
    objects = np.tile(np.eye(4), (7, 1, 1))
    objects[:, :3, :3] = Rotation.from_euler("z", times * np.pi / 2).as_matrix()
    objects[:, :3, 3] = np.c_[times, .2 * np.sin(np.pi * times), np.zeros(7)]
    local_offset = np.array([.08, .02, .03])
    desired = np.zeros((7, 14))
    desired[:, :3] = np.einsum("tij,j->ti", objects[:, :3, :3], local_offset) + objects[:, :3, 3]
    desired[:, 3:7] = Rotation.from_matrix(objects[:, :3, :3]).as_quat()
    desired[:, 7:] = np.arange(49).reshape(7, 7) / 100
    commands = desired.copy()
    commands[2:5, :3] = np.linspace(desired[1, :3], desired[5, :3], 5)[1:-1]
    commands[2:5, 3:7] = [0, 0, 0, 1]
    valid = np.array([True, True, False, False, False, True, True])
    return commands, objects, times, valid, desired


def call(q, objects, times, valid, start=2, stop=5, assumption="inferred_object_relative_hold"):
    return infer_object_relative_gap(q, objects, times, start, stop, source_valid=valid, contact_assumption=assumption)


def test_curved_object_motion_restores_fixed_hold_instead_of_world_chord():
    q, objects, times, valid, expected = fixture()
    assert np.linalg.norm(q[3, :3] - expected[3, :3]) > .05
    original = q.copy()
    result = call(q, objects, times, valid)
    np.testing.assert_allclose(result.commands[2:5, :3], expected[2:5, :3], atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(result.commands[2:5, 3:7]).as_matrix(), Rotation.from_quat(expected[2:5, 3:7]).as_matrix(), atol=1e-12)
    np.testing.assert_array_equal(result.commands[:, 7:], original[:, 7:])
    np.testing.assert_array_equal(result.commands[[0, 1, 5, 6]], original[[0, 1, 5, 6]])
    np.testing.assert_array_equal(q, original)
    np.testing.assert_array_equal(result.original_source_valid, valid)
    np.testing.assert_array_equal(result.inferred_command_mask, ~valid)
    assert not result.original_source_valid[result.inferred_command_mask].any()


def test_irregular_clock_uses_time_fraction_for_relative_wrist_translation():
    q, objects, _, valid, _ = fixture()
    objects[:] = np.eye(4)
    q[1, :3] = [0, 0, 0]
    q[5, :3] = [1, 0, 0]
    times = np.array([0, .1, .15, .3, .7, .9, 1.])
    result = call(q, objects, times, valid)
    np.testing.assert_allclose(result.commands[2:5, 0], (times[2:5] - .1) / .8)


@pytest.mark.parametrize("failure", ["valid_inside", "invalid_anchor", "implicit_flags", "nonrigid", "nonfinite", "clock", "quaternion", "unbracketed", "contact_truth"])
def test_rejects_unsupported_inference_contract(failure):
    q, objects, times, valid, _ = fixture()
    start, stop, assumption = 2, 5, "inferred_object_relative_hold"
    if failure == "valid_inside": valid[3] = True
    elif failure == "invalid_anchor": valid[1] = False
    elif failure == "implicit_flags": valid = valid.astype(int)
    elif failure == "nonrigid": objects[3, 0, 0] = 2
    elif failure == "nonfinite": q[3, 0] = np.nan
    elif failure == "clock": times[3] = times[2]
    elif failure == "quaternion": q[3, 3:7] = 0
    elif failure == "unbracketed": start = 0
    elif failure == "contact_truth": assumption = "measured_contact"
    with pytest.raises(ValueError):
        call(q, objects, times, valid, start, stop, assumption)
