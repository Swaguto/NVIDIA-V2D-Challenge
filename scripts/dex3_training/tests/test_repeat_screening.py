import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from analyze_controlled_experiments import screen, summarize


def trials(auc):
    return [
        {
            "add_auc": v,
            "chord_sr": 1.0,
            "spider_sr_uncentered": 1.0,
            "maniptrans_sr": 0.75,
            "relative_position_error_cm": 5.0,
            "mppe_cm": 2.5,
        }
        for v in auc
    ]


def test_small_gain_within_observed_control_variability_does_not_qualify():
    parent = summarize(trials([0.84, 0.86, 0.85]))
    control = summarize(trials([0.84, 0.86, 0.85]))
    candidate = summarize(trials([0.86, 0.86, 0.86]))
    result = screen(parent, control, candidate)
    assert not result["eligible_for_independent_confirmation"]
    assert result["empirical_auc_variability_guard"] > 0.019


def test_clear_repeated_improvement_qualifies_only_for_confirmation():
    parent = summarize(trials([0.84, 0.86, 0.85]))
    candidate = summarize(trials([0.89, 0.90, 0.90]))
    result = screen(parent, parent, candidate)
    assert result["eligible_for_independent_confirmation"]
    assert not result["automatic_long_continuation"]


def test_auc_cannot_compensate_for_success_rate_regression():
    parent = summarize(trials([0.84, 0.86, 0.85]))
    values = trials([0.90, 0.91, 0.90])
    for value in values:
        value["maniptrans_sr"] = 0.5
    assert not screen(parent, parent, summarize(values))[
        "eligible_for_independent_confirmation"
    ]


@pytest.mark.parametrize("values", [[0.8, 0.9], [0.8, float("nan"), 0.9]])
def test_missing_repeat_or_nonfinite_metrics_are_rejected(values):
    with pytest.raises(ValueError):
        summarize(trials(values))
