"""Check a real new-run model_0 against its parent and a fresh upstream learner."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf
from flash_chord.configuration import instantiate_typed
from flash_chord.training.checkpoint import load_checkpoint_for_inference, read_checkpoint_metadata
from flash_chord.training.ppo.config import TrainingConfig
from flash_chord.training.ppo.learner import create_learner
from v2d_dex3_flashchord.actor_initialization import validate


def check(parent_path, child_path, parent_context, current_context):
    old = read_checkpoint_metadata(parent_path)
    new = read_checkpoint_metadata(child_path)
    lineage = validate(old, new, parent_context, current_context, parent_path)
    assert json.loads(new["dex3_actor_initialization_json"]) == lineage
    config = json.loads(new["resolved_config_json"])
    training = instantiate_typed(OmegaConf.create(config["training"]), TrainingConfig)
    schema = json.loads(new["policy_schema_json"])

    class DummyEnv:
        world_count = training.world_count
        observation_dim = schema["observation"]["dimension"]
        action_dim = schema["action"]["dimension"]

        def reset(self):
            return np.zeros((self.world_count, self.observation_dim), np.float32)

    fresh = create_learner(DummyEnv(), training)
    parent, _ = load_checkpoint_for_inference(parent_path, fresh)
    child, _ = load_checkpoint_for_inference(child_path, fresh)
    for i in range(len(training.network.actor_hidden_dims) + 1):
        name = f"Dense_{i}"
        for key, array in parent.train_state.params["params"][name].items():
            np.testing.assert_array_equal(child.train_state.params["params"][name][key], array)
    for name, value in fresh.train_state.params["params"].items():
        if name.startswith("Dense_") and int(name[6:]) <= len(training.network.actor_hidden_dims):
            continue
        for actual, expected in zip(jax.tree_util.tree_leaves(child.train_state.params["params"][name]), jax.tree_util.tree_leaves(value), strict=True):
            np.testing.assert_array_equal(actual, expected)
    for attribute in ("opt_state", "step"):
        for actual, expected in zip(jax.tree_util.tree_leaves(getattr(child.train_state, attribute)), jax.tree_util.tree_leaves(getattr(fresh.train_state, attribute)), strict=True):
            np.testing.assert_array_equal(actual, expected)
    for attribute in ("key", "learning_rate", "iteration"):
        np.testing.assert_array_equal(getattr(child, attribute), getattr(fresh, attribute))
    for actual, expected in zip(jax.tree_util.tree_leaves(child.normalization), jax.tree_util.tree_leaves(parent.normalization), strict=True):
        np.testing.assert_array_equal(actual, expected)
    observation = np.random.default_rng(0).normal(size=(4, DummyEnv.observation_dim)).astype(np.float32)
    parent_policy, _ = parent.train_state.apply_fn(parent.train_state.params, observation)
    child_policy, _ = child.train_state.apply_fn(child.train_state.params, observation)
    np.testing.assert_allclose(parent_policy.mean(), child_policy.mean(), atol=2e-6, rtol=2e-6)
    assert int(child.iteration) == int(new["iteration"]) == int(new["environment_steps"]) == 0
    return dict(actor_mean_parameters_and_forward_outputs_preserved=True,
                observation_normalization_preserved=True, critic_optimizer_noise_RNG_learning_rate_fresh=True,
                new_run_steps_and_iteration_zero=True, unchanged_resume=False,
                parent_environment_steps=lineage["parent_environment_steps"], physical_task_accepted=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--parent-context", type=Path, required=True)
    parser.add_argument("--current-context", type=Path, default=Path("/opt/dex3-extension/image-context-manifest.json"))
    args = parser.parse_args()
    print(json.dumps(check(args.parent, args.child, args.parent_context, args.current_context)))
