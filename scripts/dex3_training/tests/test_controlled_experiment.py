import json
from copy import deepcopy

import pytest
from test_resume_contract import fixture
from v2d_dex3_flashchord.resume_contract import sha256, validate


def controlled(tmp_path, kind="objective_weight"):
    args = fixture(tmp_path)
    old = json.loads(args[0]["resolved_config_json"])
    old["objective"] = {"contact_wrench_support": {"enabled": True, "weight": 10.0}}
    old["curriculum"] = {
        "stages": [{"objective_weights": {"contact_wrench_support": 10.0}}]
    }
    old["reset"] = {"voc_decay_steps": 20}
    old["env"] = {
        "objective": deepcopy(old["objective"]),
        "reset": deepcopy(old["reset"]),
    }
    old["training"]["curriculum"] = deepcopy(old["curriculum"])
    new = deepcopy(old)
    new["training"].update(resume=True, checkpoint="/parent/model", max_iterations=2100)
    new["output_dir"] = "/outputs/new"
    spec = {"schema": "v2d.single-factor-experiment.v1", "kind": kind}
    if kind == "objective_weight":
        spec.update(term="contact_wrench_support", multiplier=0.5)
        new["objective"]["contact_wrench_support"]["weight"] = 5
        new["env"]["objective"]["contact_wrench_support"]["weight"] = 5
        for c in (new["curriculum"], new["training"]["curriculum"]):
            c["stages"][0]["objective_weights"]["contact_wrench_support"] = 5
    elif kind == "reset_stabilization":
        spec["steps"] = 5
        new["reset"]["voc_decay_steps"] = 5
        new["env"]["reset"]["voc_decay_steps"] = 5
    new["controlled_experiment"] = spec
    args[0]["resolved_config_json"] = json.dumps(old)
    args[1]["resolved_config_json"] = json.dumps(new)
    context = json.loads(args[3].read_text())
    context["files"].update(
        {"extension/v2d_dex3_flashchord/controlled_experiment.py": "declared validator"}
    )
    args[3].write_text(json.dumps(context))
    args[1]["dex3_image_context_sha256"] = sha256(args[3])
    return args


@pytest.mark.parametrize("kind", ["control", "objective_weight", "reset_stabilization"])
def test_declared_factor_restores_same_learner_with_honest_lineage(tmp_path, kind):
    result = validate(*controlled(tmp_path, kind))
    assert result["controlled_experiment"]["kind"] == kind
    assert result["controlled_experiment"]["complete_learner_restored"]
    assert not result["unchanged_experiment"]
    assert result["parent_environment_steps"] == 16_384_000


@pytest.mark.parametrize(
    "extra", ["action", "worlds", "optimizer", "other_weight", "environment_alias"]
)
def test_declared_factor_cannot_hide_second_change(tmp_path, extra):
    args = controlled(tmp_path)
    c = json.loads(args[1]["resolved_config_json"])
    if extra == "action":
        c["action"]["finger_joint_pos_clip"] = 1.5
    if extra == "worlds":
        c["training"]["world_count"] = 128
    if extra == "optimizer":
        c["training"]["learning_rate"] = 1e-2
    if extra == "other_weight":
        c["objective"]["undeclared"] = {"weight": 5}
    if extra == "environment_alias":
        c["env"]["objective"]["contact_wrench_support"]["weight"] = 9
    args[1]["resolved_config_json"] = json.dumps(c)
    with pytest.raises(ValueError):
        validate(*args)


def test_control_cannot_hide_reward_change(tmp_path):
    args = controlled(tmp_path, "control")
    c = json.loads(args[1]["resolved_config_json"])
    c["objective"]["contact_wrench_support"]["weight"] = 5
    args[1]["resolved_config_json"] = json.dumps(c)
    with pytest.raises(ValueError, match="same resolved experiment"):
        validate(*args)


def test_factor_cannot_change_physics_payload(tmp_path):
    args = controlled(tmp_path)
    c = json.loads(args[3].read_text())
    c["files"]["extension/action.py"] = "changed physics"
    args[3].write_text(json.dumps(c))
    args[1]["dex3_image_context_sha256"] = sha256(args[3])
    with pytest.raises(ValueError, match="runtime payload changed"):
        validate(*args)
