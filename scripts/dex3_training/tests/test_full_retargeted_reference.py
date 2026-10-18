"""One-hand tasks may retain unknown source masks without inventing observations."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from prepare_window import select_rows


def commands(count):
    q = np.zeros((count, 2, 14))
    q[:, :, 6] = 1
    return q


def test_complete_commands_do_not_require_both_hands_to_be_observed():
    valid = np.zeros(50, dtype=bool)
    valid[5:10] = True
    original = valid.copy()
    assert select_rows(valid, commands(50), full_episode=True) == (0, 50)
    np.testing.assert_array_equal(valid, original)
    with pytest.raises(ValueError, match="one second"):
        select_rows(valid, commands(50))


@pytest.mark.parametrize("invalid", ["missing_frame", "nan", "quaternion"])
def test_full_reference_never_silently_fills_invalid_commands(invalid):
    q = commands(30)
    if invalid == "missing_frame":
        q = q[:-1]
    elif invalid == "nan":
        q[2, 0, 0] = np.nan
    else:
        q[1, 0, 6] = 0
    with pytest.raises(ValueError):
        select_rows(np.zeros(30, dtype=bool), q, full_episode=True)
