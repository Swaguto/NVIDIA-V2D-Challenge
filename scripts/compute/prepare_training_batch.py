"""Prepare a short PPO continuation, evaluation gate and conditional longer chunk.

Preparation is offline. Use only with a baseline readback, the matching immutable
image and input pack. A plan is not authorization to start paid compute.
"""

import argparse
import json
from pathlib import Path

from evaluation_gate import metadata, sha256
from prepare_profile_batch import prepare
from run_batch import preflight


def prepare_training(
    inputs,
    checkpoint,
    context,
    baseline,
    outputs,
    image,
    experiment,
    short_iterations=100,
    longer_iterations=400,
):
    if (
        not 25 <= short_iterations <= 100
        or not 25 <= longer_iterations <= 500
        or short_iterations % 25
        or longer_iterations % 25
    ):
        raise ValueError(
            "Use multiples of 25, up to 100 short and 500 gated continuation iterations"
        )
    if not experiment.replace("_", "").isalnum():
        raise ValueError("Expected a named experiment")
    plan = prepare(inputs, checkpoint, context, outputs, image)
    baseline = Path(baseline).resolve()
    checkpoint, context = Path(checkpoint).resolve(), Path(context).resolve()
    saved = metadata(checkpoint)
    cfg = json.loads(saved["user.resolved_config_json"])
    parent = int(saved["user.iteration"])
    if parent < 0 or parent % 25:
        raise ValueError(
            "Parent iteration must align with the 25-iteration save interval"
        )
    steps_per_iteration = (
        cfg["training"]["world_count"] * cfg["training"]["rollout_steps"]
    )
    if int(saved["user.environment_steps"]) != parent * steps_per_iteration:
        raise ValueError("Parent training counters do not match the configuration")
    short_end, long_end = (
        parent + short_iterations,
        parent + short_iterations + longer_iterations,
    )
    plan["mounts"].append(
        {"source": str(baseline), "target": "/baseline", "read_only": True}
    )
    plan["required_files"].extend(
        {"path": "/baseline/" + name, "sha256": sha256(baseline / name)}
        for name in ("report.json", "evaluation-report.json")
    )
    initial_checkpoint = "/profile-checkpoint/" + checkpoint.name

    def train(name, parent_checkpoint, end, timeout):
        return {
            "name": name,
            "kind": "container",
            "purpose": "train",
            "timeout_seconds": timeout,
            "script": "/opt/dex3-extension/train.py",
            "args": [
                "--config-dir",
                "/opt/dex3-extension/configs",
                "experiment=" + experiment,
                "training.resume=true",
                "training.checkpoint=" + parent_checkpoint,
                "training.max_iterations=" + str(end),
                "training.save_interval=25",
                "+resume_context_manifest=/profile-context/" + context.name,
                "output_dir=/outputs/" + name + "/ppo",
            ],
            "training_metrics": "/outputs/" + name + "/ppo/metrics.jsonl",
            "failure_artifacts": [
                "/outputs/" + name + "/ppo/nonfinite-training-metrics.json"
            ],
        }

    def evaluate(name, parent_checkpoint):
        return {
            "name": name,
            "kind": "container",
            "purpose": "evaluate",
            "timeout_seconds": 150,
            "script": "/opt/dex3-extension/record_policy_readback.py",
            "args": [
                "--checkpoint",
                parent_checkpoint,
                "--out",
                "/outputs/" + name,
                "--world-count",
                "4",
            ],
        }

    def gate(name, left, right, left_checkpoint, right_checkpoint):
        return {
            "name": name,
            "kind": "gate",
            "baseline": left,
            "candidate": right,
            "baseline_checkpoint": left_checkpoint,
            "candidate_checkpoint": right_checkpoint,
            "min_auc_gain": 0.005,
        }

    short_checkpoint = f"/outputs/train-short/ppo/model_{short_end}.safetensors"
    long_checkpoint = f"/outputs/train-longer/ppo/model_{long_end}.safetensors"
    plan["stages"] = [
        train("train-short", initial_checkpoint, short_end, 240),
        evaluate("evaluate-short", short_checkpoint),
        gate(
            "gate-short",
            "/baseline",
            "/outputs/evaluate-short",
            initial_checkpoint,
            short_checkpoint,
        ),
        train("train-longer", short_checkpoint, long_end, 600),
        evaluate("evaluate-longer", long_checkpoint),
        gate(
            "gate-longer",
            "/outputs/evaluate-short",
            "/outputs/evaluate-longer",
            short_checkpoint,
            long_checkpoint,
        ),
    ]
    plan["max_wall_seconds"] = 1170
    plan.pop("new_training_steps")
    plan.update(
        planned_short_steps=short_iterations * steps_per_iteration,
        conditional_longer_steps=longer_iterations * steps_per_iteration,
        purpose="Bounded complete-state continuation; longer chunk runs only after a comparable full-episode metric improvement. Allocation thresholds are not leaderboard acceptance thresholds.",
    )
    preflight(plan)
    return plan


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "checkpoint", "context", "baseline", "outputs", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--short-iterations", type=int, default=100)
    p.add_argument("--longer-iterations", type=int, default=400)
    a = p.parse_args()
    plan = prepare_training(
        a.inputs,
        a.checkpoint,
        a.context,
        a.baseline,
        a.outputs,
        a.image,
        a.experiment,
        a.short_iterations,
        a.longer_iterations,
    )
    with a.out.open("x") as stream:
        stream.write(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(preflight(plan), indent=2))
