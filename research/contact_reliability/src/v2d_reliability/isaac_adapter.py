"""Opt-in reward configuration adapter for pinned whole-body V2D commands.

This does not import Isaac Lab at module import time. Wrench terms are weighted
per body. Upstream Chamfer and force-closure terms collapse body identity first;
those receive the minimum body confidence per hand before hand aggregation.
"""
import numpy as np
import torch
from .core import aligned_weights
from .io import read_json, sha256
from .rewards import contact_terms

SUPPORTED = {"contact_wrench_support_reward", "missed_contact_penalty", "unintended_contact_penalty"}
HAND_TERMS = {"force_closure_reward", "motion_contact_tracking_gaussian_exp"}


def _weights(command, sidecar_path):
    cache = getattr(command, "_reliability_cache", None)
    if cache is None:
        md = command._motion_data
        sidecar = read_json(sidecar_path)
        # The pinned command consumes every parquet frame without interpolation.
        times = np.arange(command.num_timesteps, dtype=float) / float(md.fps)
        bodies = list(command.retargeted_object_body_names)
        hands = list(md.hand_sides)
        if set(hands) != {"left", "right"}:
            raise ValueError("whole-body adapter currently requires explicit left and right hand entries")
        data = aligned_weights(sidecar, md.sequence_id, times, hands, bodies)
        order = [hands.index("right"), hands.index("left")]
        cache = (sidecar_path, torch.tensor(data[:, order, :], dtype=torch.float32, device=command.device))
        command._reliability_cache = cache
    if cache[0] != sidecar_path:
        raise ValueError("cannot mix reliability sidecars in one command")
    return cache[1][command.timestep_counter.long()].permute(1, 0, 2)


def weighted_contact_reward(env, term_name, sidecar_path, command_name="motion", tolerance=.1, var=.1):
    command = env.command_manager.get_term(command_name)
    command.refresh_tensors()
    weights = _weights(command, sidecar_path)
    def stack(suffix):
        return torch.stack([getattr(command, f"_cached_{s}_wrench_{suffix}") for s in ("right", "left")])
    current = torch.stack([command.right_contact_wrench_supports, command.left_contact_wrench_supports])
    return contact_terms(stack("cmd_active"), stack("cur_active"), stack("cmd_supports"),
                         current, weights, tolerance, var)[term_name]


def weighted_hand_reward(env, term_name, sidecar_path, command_name="motion", std=.05,
                         mask_zero_contact=True, min_support=.01):
    command = env.command_manager.get_term(command_name)
    # These upstream objectives merge object bodies. Conservatively attenuate
    # each hand by its least reliable body, without changing the base objective.
    weights = _weights(command, sidecar_path).amin(dim=-1)
    result = torch.zeros(env.num_envs, device=env.device)
    active_hands = torch.zeros_like(result)
    for index, side in enumerate(("right", "left")):
        if term_name == "force_closure_reward":
            supports = getattr(command, f"{side}_hand_contact_wrench_supports")
            active = getattr(command, f"{side}_hand_contact_active_command") > .5
            fraction = (supports.amax(dim=1) > min_support).float().mean(-1)
            result += fraction * active.float() * weights[index]
            active_hands += active.float()
        else:
            from robotic_grounding.tasks.v2d.mdp.utils import chamfer_distance
            pos = getattr(command, f"{side}_hand_object_contact_positions_e")
            world = getattr(command, f"{side}_hand_object_contact_positions_w")
            target = getattr(command, f"{side}_hand_object_contact_command_positions_e")
            target_valid = getattr(command, f"retargeted_{side}_object_contact_is_valid")[command.timestep_counter]
            pos = pos.reshape(env.num_envs, -1, 3)
            valid = world.reshape(env.num_envs, -1, 3).norm(dim=-1) > 1e-5
            target = target.reshape(env.num_envs, -1, 3)
            target_valid = target_valid.reshape(env.num_envs, -1)
            distance = chamfer_distance(pos, target, valid, target_valid)
            reward = torch.exp(-distance.square() / std**2)
            if mask_zero_contact:
                reward = torch.where((valid.sum(-1) == 0) & (target_valid.sum(-1) == 0), 0, reward)
            result += reward * weights[index]
    return result / active_hands.clamp(min=1) if term_name == "force_closure_reward" else result


def configure(env_cfg, sidecar_path=None, enabled=False):
    """Call after scene configuration and before gym.make. Disabled is a no-op."""
    if not enabled:
        return []
    if not sidecar_path:
        raise ValueError("enabled reliability requires a sidecar")
    terms = []
    for name in dir(env_cfg.rewards):
        if name.startswith("_"):
            continue
        term = getattr(env_cfg.rewards, name)
        if term is None or not hasattr(term, "func"):
            continue
        function = getattr(term, "func", None)
        function_name = getattr(function, "__name__", "")
        if function_name in SUPPORTED | HAND_TERMS:
            terms.append((name, term, function_name))
    if not terms:
        raise ValueError("no supported contact rewards found; verify the official Track 3 task")
    for name, term, function_name in terms:
        term.func = weighted_contact_reward if function_name in SUPPORTED else weighted_hand_reward
        term.params = {**term.params, "term_name": function_name, "sidecar_path": str(sidecar_path)}
    return [name for name, _, _ in terms]


def configure_variant(env_cfg, variant="baseline", sidecar_path=None):
    """Explicit per-task fallback ablation, preserving the CWS weight schedule.

    The combination weights force closure by confidence; it is not a learned
    switch. No runtime performance claim is implied by configuring a variant.
    """
    from .track3 import VARIANTS
    if variant not in VARIANTS:
        raise ValueError("unknown experiment variant")
    if variant == "baseline":
        return []
    weighted = variant.startswith("reliability")
    if weighted:
        if not sidecar_path:
            raise ValueError("confidence variant requires sidecar")
        reference = getattr(env_cfg, "motion_file", None)
        if not reference or read_json(sidecar_path).get("reference_sha256") != sha256(reference):
            raise ValueError("sidecar must bind the resolved motion_file hash")
    changed = []
    if variant in ("force_closure", "reliability_force_closure"):
        from robotic_grounding.tasks.v2d_whole_body.mdp.rewards.contact_rewards import force_closure_reward
        target = getattr(env_cfg.rewards, "contact_wrench_support_reward", None)
        if target is None or getattr(target.func, "__name__", "") != "contact_wrench_support_reward":
            raise ValueError("fallback requires the reviewed ReconHand CWS reward")
        existing = getattr(env_cfg.rewards, "force_closure", None)
        if existing is not None and existing.weight != 0:
            raise ValueError("refusing to double-count an existing force-closure reward")
        target.func = force_closure_reward
        target.params = {"command_name": "motion", "min_support": .01}
        changed.append("contact_wrench_support_reward -> force_closure_reward")
        curriculum = getattr(getattr(env_cfg, "curriculum", None), "fixed_timestep_curriculum", None)
        schedules = curriculum.params.get("reward_weight_schedules", {}) if curriculum else {}
        for name in ("missed_contact_penalty", "unintended_contact_penalty"):
            term = getattr(env_cfg.rewards, name, None)
            if term is not None:
                term.weight = 0.0
                if name in schedules:
                    schedules[name] = [0.0] * len(schedules[name])
                changed.append(name + " disabled")
    if weighted:
        changed.extend(configure(env_cfg, sidecar_path, enabled=True))
    return changed
