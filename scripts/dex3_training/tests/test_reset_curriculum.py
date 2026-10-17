from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.reset_curriculum import validate, assistance_contract


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


def assisted_fixture():
    curriculum, weights = fixture()
    curriculum["stages"][0]["voc_scale"] = 0.75
    curriculum["stages"][1]["voc_scale"] = 0.2
    return dict(
        training_assistance="decaying_voc",
        logging=dict(mode="disabled"),
        env=dict(voc_scale=0.0),
        voc_scale_override=None,
        reset=dict(
            _target_="flash_chord.lifecycle.reset.ReferenceResetConfig",
            voc_decay_steps=20,
            voc_decay_mode="linear",
            first_frame_voc_threshold=1.1,
        ),
        curriculum=curriculum,
        training=dict(max_iterations=2000),
        sim=dict(fps=20),
    ), weights


def test_training_assistance_keeps_rewards_and_explicit_unassisted_evaluation():
    config, weights = assisted_fixture()
    receipt = assistance_contract(config)
    validate(config["curriculum"], weights, allow_training_assistance=True)
    assert receipt["target_zero_from_iteration"] == 1000
    assert receipt["evaluation_voc_scale"] == 0
    assert receipt["evaluation_settling_steps"] == 0
    assert receipt["reset_stabilization_seconds"] == 1
    config["curriculum"]["stages"][0]["objective_weights"]["contact_wrench_support"] = 1
    with pytest.raises(ValueError):
        validate(config["curriculum"], weights, allow_training_assistance=True)


@pytest.mark.parametrize("change", ["override", "increasing", "nonzero_final", "nan", "late_zero", "long_reset", "reset_mode", "probability_bypass"])
def test_assisted_training_rejects_hidden_or_nondecaying_assistance(change):
    config, _ = assisted_fixture()
    if change == "override":
        config["voc_scale_override"] = 1
    elif change == "increasing":
        config["curriculum"]["stages"][1]["voc_scale"] = 0.9
    elif change == "nonzero_final":
        config["curriculum"]["stages"][-1]["voc_scale"] = 0.1
    elif change == "nan":
        config["curriculum"]["stages"][0]["voc_scale"] = float("nan")
    elif change == "late_zero":
        config["training"]["max_iterations"] = 1200
    elif change == "long_reset":
        config["reset"]["voc_decay_steps"] = 21
    elif change == "reset_mode":
        config["reset"]["voc_decay_mode"] = "step"
    elif change == "probability_bypass":
        config["reset"]["first_frame_voc_threshold"] = 0.1
    with pytest.raises(ValueError):
        assistance_contract(config)


def test_default_mode_still_rejects_reset_assistance():
    config, _ = assisted_fixture()
    config.pop("training_assistance")
    config["voc_scale_override"] = 0
    with pytest.raises(ValueError, match="reset assistance"):
        assistance_contract(config)
