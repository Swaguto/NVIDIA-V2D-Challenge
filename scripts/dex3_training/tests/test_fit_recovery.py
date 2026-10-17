from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.fit_recovery import failed_worlds, retry_candidates


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
