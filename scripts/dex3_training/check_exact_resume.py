"""Verify the pinned runner restores learner state and refreshes observations.

Run this read-only source file in the original checkpoint's immutable image.
It executes zero optimizer or simulator steps. A real resumed PPO smoke follows.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf
from flash_chord.configuration import instantiate_typed
from flash_chord.training.checkpoint import load_checkpoint, read_checkpoint_metadata
from flash_chord.training.ppo.config import TrainingConfig
from flash_chord.training.ppo.learner import create_learner
from flash_chord.training.ppo.runner import run_training


def check(checkpoint):
    meta = read_checkpoint_metadata(checkpoint)
    config = json.loads(meta["resolved_config_json"])
    training = instantiate_typed(OmegaConf.create(config["training"]), TrainingConfig)
    schema = json.loads(meta["policy_schema_json"])

    class DummyEnv:
        world_count = training.world_count
        observation_dim = schema["observation"]["dimension"]
        action_dim = schema["action"]["dimension"]

        def reset(self):
            return np.full((self.world_count, self.observation_dim), .125, np.float32)

        def step(self, *args):
            raise AssertionError("The restoration check must execute zero simulation steps")

    env = DummyEnv()
    fresh = create_learner(env, training)
    expected, _ = load_checkpoint(checkpoint, fresh)
    actual = run_training(env, replace(training, resume=True, checkpoint=str(checkpoint),
                          max_iterations=int(meta["iteration"])), checkpoint_metadata=meta)
    fields = [(name, getattr(actual, name), getattr(expected, name))
              for name in ("normalization", "key", "learning_rate", "iteration")]
    fields += [("train_state." + name, getattr(actual.train_state, name), getattr(expected.train_state, name))
               for name in ("params", "opt_state", "step")]
    for name, a, b in fields:
        assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b), name
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    np.testing.assert_array_equal(np.asarray(actual.observation), env.reset())
    assert int(actual.iteration) == int(meta["iteration"]) > 0
    assert int(actual.train_state.step) == int(expected.train_state.step) > 0
    return dict(schema="v2d.exact-ppo-resume-state.v1",
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                iteration=int(actual.iteration), environment_steps=int(meta["environment_steps"]),
                optimizer_step=int(actual.train_state.step), learning_rate=float(actual.learning_rate),
                actor_critic_exploration_optimizer_RNG_normalization_counters_restored=True,
                fresh_environment_observations_replace_stale_checkpoint_observations=True,
                new_training_steps_executed=0, bit_exact_simulator_continuation_claimed=False,
                physical_task_accepted=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = check(args.checkpoint)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
