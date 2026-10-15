"""Confidence selection never joins disjoint observations or invents a valid row."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_window import longest_window


def test_longest_interval_retains_internal_axis():
    valid = np.array([False, True, True, False, True, True, True, False])
    assert longest_window(valid) == (4, 7)
    assert valid[slice(*longest_window(valid))].all()


def test_equal_windows_choose_first_without_outcome_selection():
    assert longest_window(np.array([True, True, False, True, True])) == (0, 2)
    assert longest_window(np.ones(3, dtype=bool)) == (0, 3)


def test_missing_or_multihand_mask_cannot_be_silently_flattened():
    with pytest.raises(ValueError, match="No complete"):
        longest_window(np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="one-dimensional"):
        longest_window(np.ones((4, 2), dtype=bool))
    with pytest.raises(ValueError, match="one-dimensional"):
        longest_window(np.array([0, 1, 1]))
