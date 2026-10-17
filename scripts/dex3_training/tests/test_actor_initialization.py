from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.actor_initialization import actor_parameters, validate


def parameters(value):
    return {"params": {**{
        f"Dense_{i}": {"kernel": np.full((2, 2), value + i, np.float32)} for i in range(4)
    }, "log_std": np.array([value], np.float32)}}


def test_transfer_preserves_fresh_critic_noise_and_inputs():
    fresh, parent = parameters(0), parameters(100)
    actual = actor_parameters(fresh, parent, 1)
    for i in range(4):
        np.testing.assert_array_equal(actual["params"][f"Dense_{i}"]["kernel"],
                                      (parent if i < 2 else fresh)["params"][f"Dense_{i}"]["kernel"])
    np.testing.assert_array_equal(actual["params"]["log_std"], fresh["params"]["log_std"])
    actual["params"]["Dense_0"]["kernel"][0, 0] = -9
    assert parent["params"]["Dense_0"]["kernel"][0, 0] == 100
    assert fresh["params"]["Dense_0"]["kernel"][0, 0] == 0


@pytest.mark.parametrize("bad", [np.array([1], np.float32), np.full((2, 2), np.nan, np.float32), np.ones((2, 2), np.float64)])
def test_transfer_rejects_incompatible_or_invalid_actor(bad):
    parent = parameters(10)
    parent["params"]["Dense_0"]["kernel"] = bad
    with pytest.raises(ValueError):
        actor_parameters(parameters(0), parent, 1)


@pytest.fixture
def contract(tmp_path):
    context = tmp_path / "context.json"
    context.write_text(json.dumps(dict(selected_source_dirty=False, private_episode_inputs_included=False, runtime_commit="pinned")))
    checkpoint = tmp_path / "parent.safetensors"
    checkpoint.write_bytes(b"parent")
    config = dict(
        **{k: {} for k in ("action", "observation", "embodiment", "task", "scene", "reset")},
        sim={"_target_": "flash_chord.runtime.sim.SimConfig", "substeps": 20},
        training=dict(world_count=2, rollout_steps=4, observation_normalization=True,
                      observation_normalization_epsilon=1e-5, network=dict(actor_hidden_dims=[2], initial_noise_std=0.3)),
        objective={"weight": 1},
    )
    parent = dict(dex3_input_manifest_sha256="input", dex3_runtime_patches_sha256="patches",
                  policy_schema_json="schema", resolved_config_json=json.dumps(config),
                  dex3_image_context_sha256=hashlib.sha256(context.read_bytes()).hexdigest(), iteration="2", environment_steps="16")
    config["sim"].update(_target_="v2d_dex3_flashchord.sim.NativeContactSimConfig", enable_multiccd=False)
    config["objective"]["weight"] = 8
    config["training"]["network"]["initial_noise_std"] = 0.1
    current = {**parent, "resolved_config_json": json.dumps(config)}
    return parent, current, context, context, checkpoint


def test_reward_and_noise_change_are_explicit_new_experiment(contract):
    result = validate(*contract)
    assert result["new_run_environment_steps"] == 0
    assert result["parent_environment_steps"] == 16
    assert not result["unchanged_resume"]


@pytest.mark.parametrize("field", ["action", "observation", "scene", "sim", "reset"])
def test_incompatible_behavior_or_physics_rejects_transfer(contract, field):
    parent, current, *paths = contract
    current = deepcopy(current)
    config = json.loads(current["resolved_config_json"])
    config[field]["changed"] = True
    current["resolved_config_json"] = json.dumps(config)
    with pytest.raises(ValueError):
        validate(parent, current, *paths)


def test_wrong_input_bytes_reject_transfer(contract):
    parent, current, *paths = contract
    with pytest.raises(ValueError):
        validate(parent, {**current, "dex3_input_manifest_sha256": "other"}, *paths)
