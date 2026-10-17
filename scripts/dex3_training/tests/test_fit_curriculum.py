"""A stable initial hold cannot be exported as successful full transport."""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.fit_curriculum import window_end, complete, initial_sigma


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


def test_resumed_objective_still_includes_the_initial_grasp_prefix():
    phase = dict(close_s=2.35, release_start_s=11.55)
    end = window_end(phase, 1, requested=6.4)
    assert end == 6.4
    assert not complete(phase, end)
    with pytest.raises(ValueError):
        window_end(phase, 1, requested=20)


def test_finer_sampling_does_not_enlarge_physical_action_bounds():
    import numpy as np

    np.testing.assert_allclose(
        initial_sigma("grasp_refit", 0.25), 0.25 * initial_sigma("grasp_refit")
    )
    for bad in [0, -1, 2, float("nan")]:
        with pytest.raises(ValueError):
            initial_sigma("grasp_refit", bad)


def test_joint_pose_and_height_parameters_keep_independent_units_and_bounds():
    from v2d_dex3_flashchord.fit_curriculum import project_pose_and_height
    raw = np.ones((2, 14)) * 10
    raw[1, 13] = -.1
    original = raw.copy()
    lower, upper = np.full(13, -.4), np.full(13, .4)
    upper[7] = .02
    projected = project_pose_and_height(raw, lower, upper, 'grasp_refit')
    np.testing.assert_allclose(projected[:, 13], [.025, 0])
    assert projected[:, 7].max() <= .02
    assert np.linalg.norm(projected[:, :3], axis=1).max() <= .4000001
    np.testing.assert_array_equal(raw, original)
    with pytest.raises(ValueError):
        project_pose_and_height(np.zeros(13), lower, upper, 'grasp_refit')
