"""Generic task diagnostics expose failures across all worlds and objects."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_scored_policy import tracking_errors


def test_one_world_dropping_one_object_is_not_averaged_out():
    reference = np.zeros((10, 2, 7))
    reference[..., 6] = 1
    actual = np.broadcast_to(reference[:, None], (10, 3, 2, 7)).copy()
    actual[5:, 2, 1, 2] -= 0.2
    rows = tracking_errors(actual, reference)
    assert rows[0]["maximum_position_error_m"] == 0
    assert rows[1]["maximum_position_error_m"] == pytest.approx(0.2)
    assert rows[1]["per_world_mean_position_error_m"] == pytest.approx([0, 0, 0.1])
    assert rows[1]["fraction_samples_over_10cm"] == pytest.approx(1 / 6)


def test_invalid_rollout_cannot_produce_a_clean_report():
    reference = np.zeros((2, 1, 7))
    reference[..., 6] = 1
    actual = reference[:, None].copy()
    actual[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        tracking_errors(actual, reference)
    with pytest.raises(ValueError, match="complete"):
        tracking_errors(reference, reference)
