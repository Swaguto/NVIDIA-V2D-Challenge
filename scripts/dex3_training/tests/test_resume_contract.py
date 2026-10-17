from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.resume_contract import sha256, validate


def fixture(tmp_path):
    old = dict(
        output_dir="/outputs/old",
        training=dict(
            resume=False,
            checkpoint=None,
            max_iterations=2000,
            save_interval=100,
            world_count=256,
            rollout_steps=32,
            seed=42,
        ),
        action=dict(finger_joint_pos_clip=0.5),
        observation=dict(maximum_observation_abs=100),
    )
    new = deepcopy(old)
    new["output_dir"] = "/outputs/continued"
    new["resume_context_manifest"] = "/parents/context-manifest.json"
    new["training"].update(
        resume=True, checkpoint="/parents/model_2000.safetensors", max_iterations=3000
    )
    context = dict(
        selected_source_dirty=False,
        private_episode_inputs_included=False,
        runtime_commit="pinned",
        files={"extension/train.py": "old", "extension/action.py": "a"},
    )
    previous, current = tmp_path / "parent.json", tmp_path / "current.json"
    previous.write_text(json.dumps(context))
    context["files"].update(
        {
            "extension/train.py": "new",
            "extension/v2d_dex3_flashchord/resume_contract.py": "validator",
        }
    )
    current.write_text(json.dumps(context))
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"validated by the upstream checkpoint reader")
    parent = dict(
        resolved_config_json=json.dumps(old),
        iteration="2000",
        environment_steps="16384000",
        dex3_image_context_sha256=sha256(previous),
        dex3_input_manifest_sha256="input",
        dex3_runtime_patches_sha256="patches",
        policy_schema_json="schema",
    )
    active = dict(
        parent,
        resolved_config_json=json.dumps(new),
        dex3_image_context_sha256=sha256(current),
    )
    return parent, active, previous, current, checkpoint


def test_same_experiment_continues_without_relabeling_initial_state(tmp_path):
    result = validate(*fixture(tmp_path))
    assert result["parent_environment_steps"] == 16_384_000
    assert result["continuation_resets_simulator"]
    assert not result["bit_exact_continuation_claimed"]


@pytest.mark.parametrize(
    "key",
    ["dex3_input_manifest_sha256", "dex3_runtime_patches_sha256", "policy_schema_json"],
)
def test_different_inputs_runtime_or_observations_cannot_resume(tmp_path, key):
    args = fixture(tmp_path)
    args[1][key] = "changed"
    with pytest.raises(ValueError, match="contract differs"):
        validate(*args)


def test_changed_physics_or_action_source_cannot_resume(tmp_path):
    args = fixture(tmp_path)
    c = json.loads(args[3].read_text())
    c["files"]["extension/action.py"] = "modified action law"
    args[3].write_text(json.dumps(c))
    args[1]["dex3_image_context_sha256"] = sha256(args[3])
    with pytest.raises(ValueError, match="runtime payload changed"):
        validate(*args)


def test_changed_action_bounds_cannot_resume(tmp_path):
    args = fixture(tmp_path)
    cfg = json.loads(args[1]["resolved_config_json"])
    cfg["action"]["finger_joint_pos_clip"] = 1.5
    args[1]["resolved_config_json"] = json.dumps(cfg)
    with pytest.raises(ValueError, match="same resolved experiment"):
        validate(*args)


def test_wrong_counters_and_parent_context_cannot_resume(tmp_path):
    args = fixture(tmp_path)
    args[0]["environment_steps"] = "1"
    with pytest.raises(ValueError, match="step count"):
        validate(*args)
    args[0]["environment_steps"] = "16384000"
    args[2].write_text("{}")
    with pytest.raises(ValueError, match="Parent build context"):
        validate(*args)


def test_numerical_failure_handling_change_is_declared_in_lineage(tmp_path):
    args = fixture(tmp_path)
    config = json.loads(args[1]["resolved_config_json"])
    config["numerical_failure_policy"] = "terminate"
    args[1]["resolved_config_json"] = json.dumps(config)
    lineage = validate(*args)
    assert lineage["parent_numerical_failure_policy"] == "abort"
    assert lineage["current_numerical_failure_policy"] == "terminate"
    assert not lineage["invalid_transition_handling_unchanged"]
