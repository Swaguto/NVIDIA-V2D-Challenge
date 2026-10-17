"""Bind resumed PPO to the same private inputs, runtime and experiment.

The upstream learner restores optimizer, RNG and normalization state, then resets
the physics environment. This is a continuation, not bit-exact simulator replay.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def experiment_contract(config):
    result = deepcopy(config)
    result.pop("output_dir", None)
    result.pop("resume_context_manifest", None)
    for name in ("resume", "checkpoint", "max_iterations", "save_interval"):
        result["training"].pop(name, None)
    return result


def validate(parent, current, parent_context_path, current_context_path, checkpoint):
    for key in (
        "dex3_input_manifest_sha256",
        "dex3_runtime_patches_sha256",
        "policy_schema_json",
    ):
        if not parent.get(key) or parent[key] != str(current.get(key)):
            raise ValueError(f"Resume contract differs: {key}")
    old = json.loads(parent["resolved_config_json"])
    new = json.loads(current["resolved_config_json"])
    if experiment_contract(old) != experiment_contract(new):
        raise ValueError("Resume requires the same resolved experiment")
    iteration = int(parent["iteration"])
    steps = int(parent["environment_steps"])
    expected = (
        iteration * new["training"]["world_count"] * new["training"]["rollout_steps"]
    )
    if (
        iteration < 0
        or steps != expected
        or new["training"]["max_iterations"] <= iteration
    ):
        raise ValueError("Invalid resume step count or continuation horizon")
    if sha256(parent_context_path) != parent.get("dex3_image_context_sha256"):
        raise ValueError("Parent build context does not match the checkpoint")
    if sha256(current_context_path) != current.get("dex3_image_context_sha256"):
        raise ValueError("Current build context does not match the image")
    previous = json.loads(Path(parent_context_path).read_text())
    active = json.loads(Path(current_context_path).read_text())
    for context in (previous, active):
        if (
            context["selected_source_dirty"]
            or context["private_episode_inputs_included"]
        ):
            raise ValueError("Resume requires clean code-only build contexts")
    if previous["runtime_commit"] != active["runtime_commit"]:
        raise ValueError("Pinned upstream runtime changed")
    # The training entry point adds lineage checks. Every existing runtime,
    # config, action, objective, observation and physics file stays identical.
    mutable = {"extension/train.py"}
    for name, digest in previous["files"].items():
        if (
            name not in mutable
            and not name.startswith("extension/tests/")
            and active["files"].get(name) != digest
        ):
            raise ValueError(f"Resume runtime payload changed: {name}")
    added = set(active["files"]) - set(previous["files"])
    unexpected = {
        name
        for name in added
        if not name.startswith("extension/tests/")
        and name != "extension/v2d_dex3_flashchord/resume_contract.py"
    }
    if unexpected:
        raise ValueError("Unexpected runtime files in the resume image")
    return dict(
        parent_checkpoint_sha256=sha256(checkpoint),
        parent_iteration=iteration,
        parent_environment_steps=steps,
        parent_image_context_sha256=sha256(parent_context_path),
        continuation_resets_simulator=True,
        bit_exact_continuation_claimed=False,
    )
