import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from prepare_scored_case import command_anchors, complete_commands


def test_metric_anchor_trial_excludes_unmeasured_initial_depth_without_relabeling_it():
    q, valid, times, limits, parking = inputs()
    q[:] = [0, 0, -0.1, 0, 0, 0, 1, *([0.0] * 7)]
    valid[:] = True
    metric = np.zeros_like(valid)
    metric[[1, 3], 0] = True
    q[[1, 3], 0, 2] = 0.2
    original_valid, original_metric, original_q = valid.copy(), metric.copy(), q.copy()
    anchors = command_anchors(valid, metric, "metric")
    completed, interpolated, parked = complete_commands(
        q, anchors, times, limits, parking
    )
    assert completed[0, 0, 2] == 0.2 and interpolated[0, 0]
    assert parked[:, 1].all()  # A visible estimate alone is not a metric anchor.
    np.testing.assert_array_equal(valid, original_valid)
    np.testing.assert_array_equal(metric, original_metric)
    np.testing.assert_array_equal(q, original_q)
    np.testing.assert_array_equal(command_anchors(valid, metric, "visual"), valid)


def inputs():
    times = np.arange(5) * 0.05
    q = np.full((5, 2, 14), np.nan)
    valid = np.zeros((5, 2), bool)
    limits = {
        side: dict(lower_radians=[-0.5] * 7, upper_radians=[1.0] * 7)
        for side in ("left", "right")
    }
    parking = {side: [0, 1, 0.5, 0, 0, 0, 1] for side in limits}
    return q, valid, times, limits, parking


def test_full_commands_preserve_known_fits_and_keep_unknown_provenance():
    q, valid, times, limits, parking = inputs()
    q[1, 0] = [0, 0, 0, 0, 0, 0, 1, *([0.0] * 7)]
    q[3, 0] = [0.2, 0, 0, 0, 0, 0, -1, *([0.4] * 7)]
    valid[[1, 3], 0] = True
    old_valid = valid.copy()
    old_q = q.copy()
    result, interpolated, parked = complete_commands(q, valid, times, limits, parking)
    np.testing.assert_array_equal(result[valid], q[valid])
    np.testing.assert_allclose(result[2, 0, [0, 7]], [0.1, 0.2])
    np.testing.assert_allclose(
        Rotation.from_quat(result[:, 0, 3:7]).magnitude(), 0, atol=1e-12
    )
    assert interpolated[:, 0].tolist() == [True, False, True, False, True]
    assert parked[:, 1].all() and not parked[:, 0].any()
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(valid, old_valid)
    np.testing.assert_array_equal(q, old_q)


def test_invalid_supported_commands_are_rejected_instead_of_filled():
    q, valid, times, limits, parking = inputs()
    valid[2, 0] = True
    with pytest.raises(ValueError, match="Supported IK"):
        complete_commands(q, valid, times, limits, parking)
    q[2, 0] = [0, 0, 0, 0, 0, 0, 1, *([2.0] * 7)]
    with pytest.raises(ValueError, match="finger limits"):
        complete_commands(q, valid, times, limits, parking)
