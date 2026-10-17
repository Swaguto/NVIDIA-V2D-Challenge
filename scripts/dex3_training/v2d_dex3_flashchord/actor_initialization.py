"""Initialize a new experiment from an actor, with a fresh critic and optimizer."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .resume_contract import sha256


def physics_config(config):
    result = deepcopy(config)
    target = result.pop("_target_")
    if target not in (
        "flash_chord.runtime.sim.SimConfig",
        "v2d_dex3_flashchord.sim.NativeContactSimConfig",
    ):
        raise ValueError("Unknown actor initialization physics config")
    result.setdefault("enable_multiccd", False)
    return result


def reset_transition(old, new):
    """Permit only an explicitly declared removal of reset VOC assistance."""
    mode = new.get("actor_initialization_reset_transition", "unchanged")
    if mode == "unchanged":
        if old["reset"] != new["reset"]:
            raise ValueError("Actor initialization changes reset")
        return dict(mode=mode)
    if mode != "remove_training_settling":
        raise ValueError("Unknown actor reset transition")
    before, after = deepcopy(old["reset"]), deepcopy(new["reset"])
    old_steps, new_steps = before.pop("voc_decay_steps", None), after.pop("voc_decay_steps", None)
    stages = new.get("curriculum", {}).get("stages", [])
    if (
        before != after
        or before.get("_target_") != "flash_chord.lifecycle.reset.ReferenceResetConfig"
        or type(old_steps) is not int or not 0 < old_steps <= 20
        or type(new_steps) is not int or new_steps != 0
        or old.get("training_assistance") != "decaying_voc"
        or new.get("training_assistance") != "disabled"
        or new.get("voc_scale_override") != 0
        or new.get("env", {}).get("voc_scale") != 0
        or not stages
        or any(s["voc_scale"] != 0 or s.get("reset_to_first_frame_probability") != 1.0
               or s.get("immediate_first_frame_probability") is not None for s in stages)
    ):
        raise ValueError("Reset transition may only remove settling for zero-assistance frame-zero practice")
    return dict(mode=mode, parent_settling_steps=old_steps, new_settling_steps=0,
                all_stages_zero_VOC=True, all_stages_frame_zero=True)


def validate(parent, current, parent_context, current_context, checkpoint):
    if "dex3_evaluation_physics_probe_json" in parent:
        raise ValueError("Evaluation-only probes cannot initialize training")
    for key in ("dex3_input_manifest_sha256", "policy_schema_json", "dex3_runtime_patches_sha256"):
        if not parent.get(key) or parent[key] != str(current.get(key)):
            raise ValueError(f"Actor initialization contract differs: {key}")
    old = json.loads(parent["resolved_config_json"])
    new = json.loads(current["resolved_config_json"])
    iteration, steps = int(parent["iteration"]), int(parent["environment_steps"])
    if iteration <= 0 or steps != iteration * old["training"]["world_count"] * old["training"]["rollout_steps"]:
        raise ValueError("Invalid actor parent training counters")
    reset_change = reset_transition(old, new)
    for key in ("action", "observation", "embodiment", "task"):
        if old[key] != new[key]:
            raise ValueError(f"Actor initialization changes {key}")
    scenes = [deepcopy(c["scene"]) for c in (old, new)]
    for scene in scenes:
        scene.pop("world_count", None)
    if scenes[0] != scenes[1]:
        raise ValueError("Actor initialization changes scene geometry or materials")
    if physics_config(old["sim"]) != physics_config(new["sim"]):
        raise ValueError("Actor initialization changes physics")
    for key in ("observation_normalization", "observation_normalization_epsilon"):
        if old["training"][key] != new["training"][key]:
            raise ValueError("Actor initialization changes observation normalization")
    networks = [deepcopy(c["training"]["network"]) for c in (old, new)]
    for network in networks:
        network.pop("initial_noise_std")
    if networks[0] != networks[1]:
        raise ValueError("Actor initialization changes network structure")
    contexts = []
    for meta, path in ((parent, parent_context), (current, current_context)):
        if sha256(path) != meta.get("dex3_image_context_sha256"):
            raise ValueError("Actor initialization context hash differs")
        context = json.loads(Path(path).read_text())
        if context["selected_source_dirty"] or context["private_episode_inputs_included"]:
            raise ValueError("Actor initialization requires clean code-only contexts")
        contexts.append(context)
    if contexts[0]["runtime_commit"] != contexts[1]["runtime_commit"]:
        raise ValueError("Actor initialization changes pinned runtime")
    return dict(
        parent_checkpoint_sha256=sha256(checkpoint),
        parent_iteration=int(parent["iteration"]),
        parent_environment_steps=int(parent["environment_steps"]),
        parent_world_count=old["training"]["world_count"],
        current_world_count=new["training"]["world_count"],
        parent_context_sha256=sha256(parent_context),
        current_context_sha256=sha256(current_context),
        transferred="Actor mean parameters and observation normalization only",
        fresh="Critic, optimizer, exploration noise, RNG, simulator, learning rate and counters",
        new_run_iteration=0,
        new_run_environment_steps=0,
        unchanged_resume=False,
        reset_transition=reset_change,
        physical_task_accepted=False,
    )


def actor_parameters(fresh, parent, actor_hidden_layers):
    """Keep fresh critic/log_std, replacing only the pinned network's actor trunk."""
    result = deepcopy(fresh)
    for index in range(actor_hidden_layers + 1):
        name = f"Dense_{index}"
        if name not in parent["params"] or name not in result["params"]:
            raise ValueError("Missing actor layer")
        if set(parent["params"][name]) != set(result["params"][name]):
            raise ValueError("Actor layer parameters differ")
        for key, array in parent["params"][name].items():
            old = np.asarray(array)
            new = np.asarray(result["params"][name][key])
            if old.shape != new.shape or old.dtype != new.dtype or not np.isfinite(old).all():
                raise ValueError("Invalid actor parameter shape, dtype or value")
        result["params"][name] = deepcopy(parent["params"][name])
    return result


def initialize(fresh, checkpoint, actor_hidden_layers):
    import jax
    from flax.core import FrozenDict, freeze, unfreeze
    from flash_chord.training.checkpoint import load_checkpoint_for_inference

    parent, _ = load_checkpoint_for_inference(checkpoint, fresh)
    selected = actor_parameters(
        unfreeze(fresh.train_state.params), unfreeze(parent.train_state.params), actor_hidden_layers
    )
    if isinstance(fresh.train_state.params, FrozenDict):
        selected = freeze(selected)
    for array in jax.tree_util.tree_leaves(parent.normalization):
        if not np.isfinite(np.asarray(array)).all():
            raise ValueError("Nonfinite parent observation normalization")
    return fresh.replace(
        train_state=fresh.train_state.replace(params=selected),
        normalization=parent.normalization,
    )
