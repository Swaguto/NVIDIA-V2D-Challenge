from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.wrench_capacity import capacity


def test_opposed_contacts_can_lift_but_cannot_use_external_support():
    result = capacity(
        [[-0.05, 0, 0], [0.05, 0, 0]],
        [[1, 0, 0], [-1, 0, 0]],
        [4, 4],
        [0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        0.5,
    )
    assert result["load_multiplier"] == pytest.approx(2.0)
    empty = capacity([], [], [], [0, 0, 0], [0, 0, 2, 0, 0, 0], 0.5)
    assert empty["load_multiplier"] == 0


def test_offset_single_contact_cannot_balance_moment_even_with_large_force():
    result = capacity(
        [[0.1, 0, 0]], [[0, 0, 1]], [100], [0, 0, 0], [0, 0, 2, 0, 0, 0], 1.25
    )
    assert result["load_multiplier"] == pytest.approx(0)


def test_com_shift_and_wrong_normal_direction_do_not_create_free_wrench():
    args = ([[-0.05, 0, 0], [0.05, 0, 0]], [[1, 0, 0], [-1, 0, 0]], [4, 4])
    result = capacity(*args, [0, 0.1, 0], [0, 0, 2, 0, 0, 0], 0.5)
    assert result["load_multiplier"] == pytest.approx(0)
    with pytest.raises(ValueError):
        capacity([[0, 0, 0]], [[2, 0, 0]], [1], [0, 0, 0], np.ones(6), 0.5)
