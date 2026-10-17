from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.reset_curriculum import validate


def fixture():
    weights = dict(
        object_keypoints=2.0, robot_pad_support=3.0, contact_wrench_support=0.0
    )
    config = dict(
        _target_="flash_chord.lifecycle.curriculum.FixedCurriculum",
        thresholds=[500, 1000, 2000],
        stages=[
            dict(
                voc_scale=0.0,
                reset_to_first_frame_probability=p,
                objective_weights=deepcopy(weights),
            )
            for p in [0.25, 0.6, 1.0]
        ],
    )
    return config, weights


def test_unassisted_practice_must_finish_at_frame_zero():
    config, weights = fixture()
    validate(config, weights)
    config["stages"][-1]["reset_to_first_frame_probability"] = 0.8
    with pytest.raises(ValueError, match="frame-zero"):
        validate(config, weights)


@pytest.mark.parametrize("change", ["assistance", "labels", "weight", "settling"])
def test_reset_schedule_cannot_change_rewards_or_add_assistance(change):
    config, weights = fixture()
    if change == "assistance":
        config["stages"][0]["voc_scale"] = 0.1
    if change == "labels":
        config["stages"][0]["objective_weights"]["contact_wrench_support"] = 1
    if change == "weight":
        config["stages"][0]["objective_weights"]["robot_pad_support"] = 99
    if change == "settling":
        config["stages"][0]["immediate_first_frame_probability"] = 0.5
    with pytest.raises(ValueError):
        validate(config, weights)
