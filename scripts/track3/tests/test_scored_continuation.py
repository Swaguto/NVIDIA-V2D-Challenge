import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from continue_scored_policy import summarize


def report(auc, sr=0.0):
    return {
        "world_count": 8,
        "non_finite_worlds": [],
        "start_frame": 0,
        "step_count": 313,
        "metrics": {"add_auc": auc, "maniptrans_sr": sr},
    }


def test_repeated_tracking_improvement_does_not_promote_a_failed_grasp():
    result = summarize([report(0.1)] * 3, [report(0.2)] * 3, 313)
    assert result["auc_ranges_overlap"] is False
    assert result["candidate"]["mean_mp_sr"] == 0
    assert result["policy_promoted"] is False
    assert result["physical_task_accepted"] is False
    assert result["independent_training_repeat_still_required"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"world_count": 1},
        {"non_finite_worlds": [0]},
        {"start_frame": 10},
        {"step_count": 250},
    ],
)
def test_incomplete_or_unstable_rollouts_cannot_qualify_a_candidate(change):
    bad = dict(report(0.9), **change)
    with pytest.raises(ValueError, match="frame-zero"):
        summarize([report(0.1)] * 3, [bad] * 3, 313)


def test_one_good_rollout_cannot_replace_repeated_comparison():
    with pytest.raises(ValueError, match="three full"):
        summarize([report(0.1)] * 3, [report(0.9)], 313)
