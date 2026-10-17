from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.fit_objective import constraint_cost


def candidate(**updates):
    values = dict(maximum_relative_position_drift_m=.008,
                  maximum_relative_rotation_drift_deg=4.,
                  all_required_pad_sampled_support_fraction=1.,
                  maximum_reference_lift_m=.099,
                  maximum_actual_lift_m=.10,
                  maximum_source_position_error_m=.025,
                  maximum_source_orientation_error_deg=10.)
    return dict(values, **updates)


def test_reducing_safe_rotation_does_not_buy_more_slip():
    near = constraint_cost(candidate(maximum_relative_position_drift_m=.0105), .02, .001)
    worse = constraint_cost(candidate(maximum_relative_position_drift_m=.0116,
                                     maximum_relative_rotation_drift_deg=1.), .02, .001)
    assert near < worse
    assert constraint_cost(candidate(), .02, .001) < near


@pytest.mark.parametrize("changes", [
    {"all_required_pad_sampled_support_fraction": .5},
    {"maximum_actual_lift_m": .03},
    {"maximum_source_position_error_m": .1},
    {"maximum_source_orientation_error_deg": 30.},
])
def test_lift_support_and_source_failures_are_not_hidden_by_good_retention(changes):
    assert constraint_cost(candidate(**changes), .02, .001) > 100


def test_native_limits_and_nonfinite_evidence_still_reject_a_candidate():
    good = constraint_cost(candidate(), .02, .001)
    assert constraint_cost(candidate(), -.001, .001) > good + 100
    assert constraint_cost(candidate(), .02, .003) > good + 100
    with pytest.raises(ValueError):
        constraint_cost(candidate(maximum_actual_lift_m=float("nan")), .02, .001)
