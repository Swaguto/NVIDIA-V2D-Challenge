"""Prepare short, fully resolved PPO forks from one retained checkpoint.

The control and each single-factor treatment restore the same entire learner.
No training or cloud action occurs. JSON files are valid Hydra YAML inputs.
"""

import argparse
import hashlib
import json
import struct
from copy import deepcopy
from pathlib import Path

from v2d_dex3_flashchord.controlled_experiment import normalize
from v2d_dex3_flashchord.resume_contract import experiment_contract


def prepare(checkpoint, out, iterations=100):
    if type(iterations) is not int or not 1 <= iterations <= 100 or iterations % 25:
        raise ValueError("Use at most 100 iterations in chunks of 25")
    if out.exists():
        raise FileExistsError(out)
    with checkpoint.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= size <= min(64 * 1024 * 1024, checkpoint.stat().st_size - 8):
            raise ValueError("Invalid checkpoint header")
        meta = json.loads(stream.read(size))["__metadata__"]
    old = json.loads(meta["user.resolved_config_json"])
    parent_iteration = int(meta["user.iteration"])
    expected_steps = (
        parent_iteration
        * old["training"]["world_count"]
        * old["training"]["rollout_steps"]
    )
    if expected_steps != int(meta["user.environment_steps"]):
        raise ValueError("Parent training counters differ")
    variants = {
        "control": {"kind": "control"},
        "half-cws": {
            "kind": "objective_weight",
            "term": "contact_wrench_support",
            "multiplier": 0.5,
        },
        "short-settling": {"kind": "reset_stabilization", "steps": 5},
    }
    configs = {}
    for name, factor in variants.items():
        config = deepcopy(old)
        config.update(
            output_dir="/outputs/ppo",
            resume_context_manifest="/initial-context/context-manifest.json",
            controlled_experiment=dict(
                schema="v2d.single-factor-experiment.v1", **factor
            ),
        )
        config["training"].update(
            resume=True,
            checkpoint="/initial-policy/ppo/" + checkpoint.name,
            max_iterations=parent_iteration + iterations,
            save_interval=25,
        )
        if name == "half-cws":
            for objective in (config["objective"], config["env"]["objective"]):
                objective["contact_wrench_support"]["weight"] *= 0.5
            for curriculum in (config["curriculum"], config["training"]["curriculum"]):
                for stage in curriculum["stages"]:
                    stage["objective_weights"]["contact_wrench_support"] *= 0.5
        elif name == "short-settling":
            config["reset"]["voc_decay_steps"] = 5
            config["env"]["reset"]["voc_decay_steps"] = 5
        normalized, _ = normalize(old, config)
        if experiment_contract(normalized) != experiment_contract(old):
            raise ValueError(
                "Prepared experiment changes more than its declared factor"
            )
        configs[name] = config
    out.mkdir(parents=True)
    for name, config in configs.items():
        (out / (name + ".yaml")).write_text(json.dumps(config, indent=2) + "\n")
    result = {
        "schema": "v2d.controlled-ppo-plan.v1",
        "parent_iteration": parent_iteration,
        "parent_environment_steps": expected_steps,
        "parent_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "new_iterations_per_trial": iterations,
        "new_environment_steps_per_trial": iterations
        * old["training"]["world_count"]
        * old["training"]["rollout_steps"],
        "variants": variants,
        "optimizer_actor_critic_rng_normalization_restored": True,
        "evaluations_required": "Full unassisted frame-zero repeat evaluations; compare with same-image control and original parent",
        "continuation_automatically_authorized": False,
        "gpu_executed": False,
        "physical_task_accepted": False,
        "files": {
            name + ".yaml": hashlib.sha256(
                (out / (name + ".yaml")).read_bytes()
            ).hexdigest()
            for name in configs
        },
    }
    (out / "experiment-plan.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(prepare(args.checkpoint, args.out, args.iterations), indent=2))
