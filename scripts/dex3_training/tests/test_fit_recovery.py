from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.fit_recovery import failed_worlds, retry_candidates, repeat_count


def test_failure_in_any_frame_or_native_motor_rejects_its_world():
    bodies = np.zeros((5, 8, 3, 7))
    bodies[2, 7, 0, 0] = np.nan
    cap_ok = np.ones(8, bool)
    cap_ok[3] = False
    np.testing.assert_array_equal(failed_worlds([bodies], cap_ok), [3, 7])
    assert np.isnan(bodies[2, 7, 0, 0])


def test_discard_both_repeats_of_the_bad_proposal_without_modifying_originals():
    candidates = np.arange(4 * 13, dtype=float).reshape(4, 13)
    original = candidates.copy()
    replay, pairs = retry_candidates(candidates, [6, 7], np.zeros(13))
    np.testing.assert_array_equal(pairs, [3])
    np.testing.assert_array_equal(replay[:3], candidates[:3])
    np.testing.assert_array_equal(replay[3], 0)
    np.testing.assert_array_equal(candidates, original)


def test_broad_or_repeated_failure_cannot_be_hidden_by_replacing_the_population():
    with pytest.raises(ValueError, match="Too many"):
        retry_candidates(np.zeros((4, 13)), [0, 2], np.zeros(13))
    with pytest.raises(ValueError):
        retry_candidates(np.zeros((4, 13)), [8], np.zeros(13))


def test_four_world_proposals_are_excluded_as_a_complete_group():
    candidates = np.arange(4 * 14, dtype=float).reshape(4, 14)
    replay, excluded = retry_candidates(candidates, [8, 11], np.zeros(14), 4)
    np.testing.assert_array_equal(excluded, [2])
    np.testing.assert_array_equal(replay[2], 0)
    np.testing.assert_array_equal(replay[[0, 1, 3]], candidates[[0, 1, 3]])
    assert repeat_count(4, 16) == 4


@pytest.mark.parametrize("repeats,worlds", [(True, 16), (3, 12), (4, 8), (8, 20)])
def test_invalid_cohort_layout_is_rejected(repeats, worlds):
    with pytest.raises(ValueError, match="worlds per candidate"):
        repeat_count(repeats, worlds)
