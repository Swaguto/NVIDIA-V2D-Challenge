"""A stable initial hold cannot be exported as successful full transport."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.fit_curriculum import window_end, complete


def test_partial_hold_stays_unaccepted_until_full_carry_window():
    phase = dict(close_s=2.35, release_start_s=11.55)
    end = window_end(phase, 1.0)
    assert end == pytest.approx(3.4)
    assert not complete(phase, end)
    for _ in range(9):
        end = window_end(phase, 1.0, end)
    assert end == pytest.approx(11.5)
    assert complete(phase, end)


def test_legacy_search_uses_full_window_and_invalid_curriculum_is_refused():
    phase = dict(close_s=2.35, release_start_s=11.55)
    assert complete(phase, window_end(phase, 0))
    for bad in [-1, 6, float("nan")]:
        with pytest.raises(ValueError):
            window_end(phase, bad)
