"""Fit bounded phase-policy residuals against native measured physics.

CEM is parameter optimization, not PPO or imitation of achieved trajectories.
Each candidate runs the complete source sequence from frame zero. Candidates
are repeated in two independent worlds; earlier accepted phases remain part of
the objective, so a better lid cannot silently break a previously fitted carry.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import warp as wp

from flash_chord.configuration import instantiate_typed, resolved_dict
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from audit_native_task import phase_metrics
from v2d_dex3_flashchord.kinematics import URDFKinematics
from v2d_dex3_flashchord.phase_prior import (
    parameter_bounds,
    project,
    schedule,
    phase_envelope,
    position_basis,
    carry_height_envelope,
    pose_refit_baseline,
)
from v2d_dex3_flashchord.sampling import resample_targets
from v2d_dex3_flashchord.native_depth import maximum_assigned_depth
from v2d_dex3_flashchord.fit_curriculum import window_end, complete, initial_sigma, project_pose_and_height
from v2d_dex3_flashchord.fit_objective import constraint_cost
from v2d_dex3_flashchord.fit_recovery import failed_worlds, retry_candidates, repeat_count


def phase_layout(env, scene, manifest, spec, names, side, phase):
    hand = scene.layout.hand(side)
    obj = next(
        b.root.body_id
        for b in scene.objects
        if b.name == manifest["objects"][phase["object"]]["name"]
    )
    tracker = env.contact.layout
    hi = tracker.sides.index(side)
    links = [g.link_name for g in hand.contact_links]
    required = [
        p["link"]
        for p in phase["pad_normal_jacobians"]
        if any(d in p["link"] for d in phase["required_digits"])
    ]
    if len(required) != len(phase["required_digits"]):
        raise ValueError("Required pad mapping differs")
    oi = list(tracker.object_body_ids).index(obj)
    slots = [
        tracker.hand_slot_starts[hi] + oi * tracker.link_counts[hi] + links.index(link)
        for link in required
    ]
    kin = URDFKinematics(Path(manifest["_root"]) / f"robot/urdfs/g1/dex3_{side}.urdf")
    required_joints = [
        n for n in names[side] if any(d in n for d in phase["required_digits"])
    ]
    indices = [names[side].index(n) for n in required_joints]
    return dict(
        side=side,
        phase=phase,
        hand=hand.palm_frame.body_id,
        obj=obj,
        slots=slots,
        q_ids=[hand.finger_q_ids[i] for i in indices],
        lower=np.array([kin.limits[n][0] for n in required_joints]),
        upper=np.array([kin.limits[n][1] for n in required_joints]),
    )


@hydra.main(
    version_base="1.3",
    config_path="/opt/flash-chord/src/flash_chord/configs",
    config_name="train",
)
def main(cfg):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(cfg.task.parquet).parent
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    input_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (
        manifest.get("motor_plan_source_motion_speed") != 1
        or cfg.task.motion_speed != 1
    ):
        raise ValueError(
            "Native fitting needs a verified 1x force plan and source clock"
        )
    OmegaConf.update(
        cfg, "embodiment.config.reference_manifest_sha256", input_sha, force_add=True
    )
    config = apply_evaluation_composition(
        resolved_dict(cfg), world_count=int(cfg.scene.world_count)
    )
    cfg = OmegaConf.create(config)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if cfg.action.input_mode != "raw" or (
        cfg.action.wrist_pos_scale,
        cfg.action.wrist_ori_scale,
        cfg.action.finger_joint_pos_scale,
    ) != (0.05, 0.15, 0.15):
        raise ValueError("Native fitting requires the raw replay action scales (.05, .15, .15)")
    with wp.ScopedDevice("cuda:0"):
        ec = instantiate_typed(cfg.env, RLEnvConfig)
        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=ec.sim.fps,
            motion_speed=1.0,
            embodiment=instantiate_typed(cfg.embodiment, Embodiment),
            collision=instantiate_typed(cfg.collision, CollisionPolicy),
            world_count=cfg.scene.world_count,
            include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            contact_friction=cfg.scene.contact_friction,
        )
        env = RLEnv(setup.scene, setup.reference, config=ec)
        if not env.action.simulator_contract_sha256:
            raise ValueError("Native fitting requires NativeContactSimConfig to bind all solver settings")
        env.reset()
        env.capture_step()
        env.reset(frame_id=0)
        worlds = env.world_count
        repeats = repeat_count(cfg.fit.get("worlds_per_candidate", 2), worlds)
        if env.action.action_dim != 26:
            raise ValueError(
                "Need paired candidate worlds and the measured dual-Dex3 action schema"
            )
        sides = list(env.action.sides)
        names = {s: list(setup.scene.layout.hand(s).finger_joint_names) for s in sides}
        spec = json.loads((root / "task-grips.json").read_text())
        with np.load(root / "targets.npz") as f:
            target = resample_targets(
                {k: f[k].copy() for k in f.files}, setup.reference.num_frames
            )
        times = target["timestamps_s"]
        side, pi = str(cfg.fit.side), int(cfg.fit.phase)
        phase = spec["phases"][side][pi]
        curriculum_step = float(cfg.fit.get("curriculum_step_s", 0))
        prefix_end = window_end(
            phase, curriculum_step, requested=cfg.fit.get("curriculum_start_source_s")
        )
        support_weight = float(cfg.fit.get("support_penalty_weight", 2))
        if not np.isfinite(support_weight) or not 2 <= support_weight <= 50:
            raise ValueError("Unsupported sampled-support objective weight")
        objective = str(cfg.fit.get("objective", "legacy"))
        if objective not in ("legacy", "constraint_violation"):
            raise ValueError("Unknown native grasp search objective")
        prior_path = cfg.fit.get("prior")
        prior_journal = cfg.fit.get("prior_journal")
        if prior_path and prior_journal:
            raise ValueError("Use a saved plan or a recorded search journal, not both")
        fit_mode = str(cfg.fit.get("mode", "pose"))
        if fit_mode not in ("pose", "carry_height_sweep", "pose_and_carry_height"):
            raise ValueError("Unknown grasp fitting mode")
        plan = (
            json.loads(Path(prior_path).read_text())
            if prior_path
            else dict(
                schema="v2d.native-phase-policy-prior.v1",
                input_manifest_sha256=input_sha,
                finger_joint_names=names,
                phase_actions={},
                fit_reports={},
            )
        )
        if prior_journal:
            journal_path = Path(prior_journal)
            source_config_path = journal_path.parent / "config.json"
            source_config = json.loads(source_config_path.read_text())
            if source_config["embodiment"]["config"]["reference_manifest_sha256"] != input_sha:
                raise ValueError("Recovered search used other input bytes")
            for key in ("action", "embodiment", "task", "sim"):
                if source_config[key] != config[key]:
                    raise ValueError(f"Recovered search changes {key}")
            scenes = [deepcopy(c["scene"]) for c in (source_config, config)]
            for scene in scenes:
                scene.pop("world_count", None)
            if scenes[0] != scenes[1]:
                raise ValueError("Recovered search changes scene geometry or materials")
            rows = journal_path.read_text().splitlines()
            last = json.loads(rows[-1])
            if not last.get("native_motor_caps_verified"):
                raise ValueError("Recovered search has no native motor verification")
            plan["phase_actions"][phase["sample"]] = last["best_raw_action"]
            plan["fit_reports"][phase["sample"]] = dict(
                robust_phase_pass=False,
                recovered_from_incomplete_search=True,
                selected_phase_objective_end_source_s=last["selected_phase_objective_end_source_s"],
                selected_phase_full_window_covered=last["selected_phase_full_window_covered"],
            )
            plan["recovered_prior_journal_sha256"] = hashlib.sha256(journal_path.read_bytes()).hexdigest()
            plan["recovered_prior_config_sha256"] = hashlib.sha256(source_config_path.read_bytes()).hexdigest()
        if plan["input_manifest_sha256"] != input_sha:
            raise ValueError("Existing policy parameters belong to other inputs")
        if (
            any(r["robust_phase_pass"] for r in plan.get("fit_reports", {}).values())
            and (
                plan.get("native_physics_model_sha256") != env.action.native_physics_model_sha256
                or plan.get("action_control_contract_sha256") != env.action.policy_control_contract_sha256
                or not plan.get("simulator_contract_sha256")
                or plan["simulator_contract_sha256"] != env.action.simulator_contract_sha256
            )
        ):
            raise ValueError(
                "Previously accepted phases need refitting after a physics change"
            )
        bounds_profile = str(cfg.fit.get("bounds_profile", "local"))
        position_frame = str(cfg.fit.get("position_frame", "world"))
        if (
            any(r["robust_phase_pass"] for r in plan.get("fit_reports", {}).values())
            and plan.get("position_frame", "world") != position_frame
        ):
            raise ValueError(
                "Previously accepted phases use another policy position frame"
            )
        plan.update(
            bounds_profile=bounds_profile,
            position_frame=position_frame,
            native_physics_model_sha256=env.action.native_physics_model_sha256,
            control_fps=float(ec.sim.fps),
            physics_substeps=int(ec.sim.substeps),
            action_control_contract_sha256=env.action.policy_control_contract_sha256,
            simulator_contract_sha256=env.action.simulator_contract_sha256,
        )
        previous = [
            name
            for name, row in plan.get("fit_reports", {}).items()
            if row["robust_phase_pass"] and name != phase["sample"]
        ]
        baseline = deepcopy(plan)
        if fit_mode in ("pose", "pose_and_carry_height"):
            baseline = pose_refit_baseline(plan, phase["sample"])
            if fit_mode == "pose_and_carry_height":
                baseline.setdefault("carry_height_bias_m", {}).pop(phase["sample"], None)
        else:
            if not (prior_path or prior_journal) or phase["sample"] not in baseline["phase_actions"]:
                raise ValueError("Carry height sweep needs an existing grasp candidate")
            baseline.setdefault("carry_height_bias_m", {}).pop(phase["sample"], None)
        fixed = schedule(spec, baseline, sides, names, times)
        manifest["_root"] = str(root)
        layouts = {}
        for s, phases in spec["phases"].items():
            for p in phases:
                if p["sample"] in previous + [phase["sample"]]:
                    layouts[p["sample"]] = phase_layout(
                        env, setup.scene, manifest, spec, names, s, p
                    )
        bounds = parameter_bounds(bounds_profile)
        lower, upper = -bounds.copy(), bounds.copy()
        for i, name in enumerate(names[side]):
            if fit_mode == "carry_height_sweep":
                lower[6 + i] = upper[6 + i] = 0
                continue
            if not any(d in name for d in phase["required_digits"]):
                lower[6 + i] = upper[6 + i] = 0
            else:
                j = phase["joint_names"].index(name)
                lo, hi = phase["joint_limits_rad"][j]
                closed = phase["closed_rad"][j]
                lower[6 + i] = max(lower[6 + i], (lo + 0.004 - closed) / 0.15)
                upper[6 + i] = min(upper[6 + i], (hi - 0.004 - closed) / 0.15)
        if np.any(lower > upper):
            raise ValueError("Required pad posture is outside interior joint limits")
        native = env.solver.mj_model
        limited = np.asarray(native.jnt_actfrclimited, bool)
        addresses = np.asarray(native.jnt_dofadr, int)[limited]
        ranges = np.asarray(native.jnt_actfrcrange)[limited]
        if np.isclose(ranges[:, 1], 0.76).sum() < 14:
            raise ValueError("Native finger motor caps missing")
        stride = env.model.body_count // worlds
        owner = np.full(stride, -1, dtype=np.int32)
        for i, label in enumerate(env.model.body_label[:stride]):
            for si, s in enumerate(sides):
                if label.startswith("dex3_" + s + "/"):
                    owner[i] = si
        assigned = []
        for s in sides:
            objects = {p["object"] for p in spec["phases"][s]}
            if len(objects) != 1:
                raise ValueError(
                    "Depth-bounded fitting needs one assigned object per hand"
                )
            name = manifest["objects"][next(iter(objects))]["name"]
            assigned.append(
                next(b.root.body_id for b in setup.scene.objects if b.name == name)
            )
        owners = wp.array(owner, dtype=wp.int32, device=env.device)
        object_ids = wp.array(assigned, dtype=wp.int32, device=env.device)
        sampled_depth = wp.zeros(
            worlds * len(sides), dtype=wp.float32, device=env.device
        )
        slot_count = env.contact.layout.slots_per_world
        wrist_envelope = phase_envelope(phase, times)
        carry_gate = carry_height_envelope(spec, phase, times) if fit_mode in ("carry_height_sweep", "pose_and_carry_height") else None
        basis = position_basis(spec, phase, times, position_frame)
        start = sides.index(side) * 13
        rng = np.random.default_rng(int(cfg.fit.get("seed", 42)))
        population = worlds // repeats
        mu = np.asarray(plan["phase_actions"].get(phase["sample"], np.zeros(13)), float)
        if fit_mode == "carry_height_sweep":
            mu = np.zeros(13)
        sigma_scale = float(cfg.fit.get("initial_sigma_scale", 1.0))
        sigma = initial_sigma(bounds_profile, sigma_scale)
        if fit_mode == "pose_and_carry_height":
            mu = np.r_[mu, plan.get("carry_height_bias_m", {}).get(phase["sample"], 0.0)]
            sigma = np.r_[sigma, .003 * sigma_scale]

        def bounded(values):
            if fit_mode == "pose_and_carry_height":
                return project_pose_and_height(values, lower, upper, bounds_profile)
            return project(values, lower, upper, profile=bounds_profile)

        best_cost, best_raw, best_metrics = float("inf"), mu.copy(), None
        journal = (out / "iterations.jsonl").open("x")
        maximum_iterations = int(cfg.fit.iterations)
        fixed_parameters = bool(cfg.fit.get("fixed_parameters", False))
        if fixed_parameters and (
            maximum_iterations != 1 or curriculum_step != 0
            or not complete(phase, prefix_end) or fit_mode == "carry_height_sweep"
            or bool(cfg.fit.get("retry_failed_cohorts", False))
            or not np.allclose(bounded(mu), mu, atol=1e-12, rtol=0)
        ):
            raise ValueError("Fixed controller probes require one full clean window and unchanged bounded parameters")
        if fit_mode == "carry_height_sweep" and (maximum_iterations != 1 or curriculum_step != 0 or population < 6):
            raise ValueError("Carry height sweep uses one complete-window grid with at least6 paired candidates")
        simulated_steps = 0
        discarded_steps = 0
        allow_retry = bool(cfg.fit.get("retry_failed_cohorts", False))
        if int(native.na) != 0:
            raise ValueError("Native fitting requires stateless actuators")
        for iteration in range(maximum_iterations):
            if fixed_parameters:
                candidates = np.broadcast_to(mu, (population, len(mu))).copy()
            elif fit_mode == "carry_height_sweep":
                candidates = np.zeros((population, 13))
                candidates[:, 2] = np.linspace(0, 0.025, population)
            else:
                candidates = bounded(rng.normal(mu, sigma, (population, len(mu))))
                candidates[0] = bounded(best_raw)
            replay_candidates = candidates
            excluded_pairs = np.array([], dtype=int)
            failed_evidence = []
            for attempt in range(2 if allow_retry else 1):
                actions = np.repeat(replay_candidates, repeats, axis=0)
                env.reset(frame_id=0)
                env.solver.mjw_data.qacc_warmstart.zero_()
                bodies, joints, contacts, depths = [], [], [], []
                cap_ok = np.ones(worlds, bool)
                for frame in range(setup.reference.num_frames):
                    raw = np.broadcast_to(fixed[frame], (worlds, 26)).copy()
                    if fit_mode == "carry_height_sweep":
                        raw[:, start + 2] += actions[:, 2] / 0.05 * carry_gate[frame]
                    else:
                        rotated_actions = actions[:, :13].copy()
                        rotated_actions[:, :3] = np.einsum(
                            "ij,wj->wi", basis[frame], actions[:, :3]
                        )
                        raw[:, start : start + 13] += rotated_actions * wrist_envelope[frame]
                        if fit_mode == "pose_and_carry_height":
                            raw[:, start + 2] += actions[:, 13] / .05 * carry_gate[frame]
                    env.step(wp.array(raw.ravel(), dtype=wp.float32, device=env.device))
                    bodies.append(
                        env.state_0.body_q.numpy().reshape(worlds, stride, 7).copy()
                    )
                    joints.append(env.state_0.joint_q.numpy().reshape(worlds, -1).copy())
                    contacts.append(
                        env.contact.contact_force_w.numpy()
                        .reshape(worlds, slot_count, 3)
                        .copy()
                    )
                    sampled_depth.zero_()
                    c = env.contacts
                    wp.launch(
                        maximum_assigned_depth,
                        dim=c.rigid_contact_shape0.shape[0],
                        inputs=[
                            c.rigid_contact_count,
                            c.rigid_contact_shape0,
                            c.rigid_contact_shape1,
                            c.rigid_contact_point0,
                            c.rigid_contact_point1,
                            c.rigid_contact_normal,
                            env.model.shape_body,
                            env.state_0.body_q,
                            stride,
                            len(sides),
                            owners,
                            object_ids,
                            sampled_depth,
                        ],
                        device=env.device,
                    )
                    depths.append(sampled_depth.numpy().reshape(worlds, len(sides)).copy())
                    actual = env.solver.mjw_data.qfrc_actuator.numpy()[:, addresses]
                    cap_ok &= (
                        np.isfinite(actual).all(axis=1)
                        & np.all(actual >= ranges[:, 0] - 1e-5, axis=1)
                        & np.all(actual <= ranges[:, 1] + 1e-5, axis=1)
                    )
                simulated_steps += worlds * setup.reference.num_frames
                invalid = failed_worlds([bodies, joints, contacts, depths], cap_ok)
                if not len(invalid):
                    break
                discarded_steps += worlds * setup.reference.num_frames
                suffix = "" if attempt == 0 else "-retry"
                evidence = out / f"failed-cohort-{iteration + 1}{suffix}.npz"
                np.savez_compressed(
                    evidence, candidate_parameters=candidates,
                    repeated_world_actions=actions, body_poses=np.asarray(bodies),
                    joint_positions=np.asarray(joints), contact_forces=np.asarray(contacts),
                    sampled_depths=np.asarray(depths),
                    final_native_limited_motor_force=actual,
                    native_limited_motor_ranges=ranges, native_cap_ok=cap_ok,
                    invalid_world_indices=invalid,
                )
                evidence_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
                failed_evidence.append(dict(path=evidence.name, sha256=evidence_sha))
                evidence.with_suffix(".json").write_text(json.dumps(dict(
                    iteration=iteration + 1, attempt=attempt,
                    raw_evidence_sha256=evidence_sha, invalid_world_indices=invalid.tolist(),
                    native_motor_or_finiteness_gate_failed=True,
                    candidate_cohort_rejected=True, failed_states_repaired=False,
                    original_cohort_used_for_scoring=False, physical_task_accepted=False,
                ), indent=2) + "\n")
                if not allow_retry or attempt != 0:
                    raise RuntimeError("Candidate cohort failed native motor/finiteness gates")
                replay_candidates, excluded_pairs = retry_candidates(candidates, invalid, best_raw, repeats)
            else:
                raise RuntimeError("No fully clean candidate cohort")
            bodies, joints, contacts = (
                np.asarray(bodies),
                np.asarray(joints),
                np.asarray(contacts),
            )
            depths = np.asarray(depths)
            costs = np.zeros(worlds)
            metrics = [[] for _ in range(worlds)]
            for name, layout in layouts.items():
                p = layout["phase"]
                end = (
                    prefix_end
                    if name == phase["sample"]
                    else p["release_start_s"] - 0.05
                )
                indices = np.flatnonzero(
                    (times >= p["close_s"] + 0.05 - 1e-8) & (times <= end + 1e-8)
                )
                ref = target["object_poses"][indices, p["object"]]
                for world in range(worlds):
                    h = bodies[indices, world, layout["hand"]]
                    obj = bodies[indices, world, layout["obj"]]
                    force = np.linalg.norm(
                        contacts[indices, world][:, layout["slots"]], axis=-1
                    )
                    try:
                        m = phase_metrics(h, obj, ref, force)
                        angle = float(
                            np.rad2deg(
                                (
                                    Rotation.from_quat(ref[:, 3:]).inv()
                                    * Rotation.from_quat(obj[:, 3:])
                                ).magnitude()
                            ).max()
                        )
                        q = joints[indices, world][:, layout["q_ids"]]
                        margin = float(
                            np.minimum(q - layout["lower"], layout["upper"] - q).min()
                        )
                        passed = bool(
                            m["relative_retention_10mm_5deg_with_sampled_support"]
                            and m["maximum_source_position_error_m"] <= 0.03
                            and angle <= 15
                            and margin >= 0.003
                            and depths[
                                indices, world, sides.index(layout["side"])
                            ].max()
                            <= 0.002
                        )
                        cost = (
                            2 * m["maximum_relative_position_drift_m"] / 0.01
                            + m["maximum_relative_rotation_drift_deg"] / 5
                            + support_weight
                            * (1 - m["all_required_pad_sampled_support_fraction"])
                            + max(
                                0,
                                m["maximum_reference_lift_m"]
                                - m["maximum_actual_lift_m"]
                                - 0.01,
                            )
                            / 0.01
                            + m["maximum_source_position_error_m"] / 0.03
                            + angle / 15
                            + 1000 * max(0, 0.003 - margin)
                            + 5
                            * max(
                                0,
                                float(
                                    depths[
                                        indices, world, sides.index(layout["side"])
                                    ].max()
                                )
                                - 0.001,
                            )
                            / 0.001
                        )
                        if not passed:
                            cost += 1000
                        if objective == "constraint_violation":
                            cost = constraint_cost(m, margin, float(depths[
                                indices, world, sides.index(layout["side"])
                            ].max()))
                        row = dict(
                            sample=name,
                            **m,
                            minimum_required_joint_margin_rad=margin,
                            maximum_sampled_hand_object_penetration_m=float(
                                depths[
                                    indices, world, sides.index(layout["side"])
                                ].max()
                            ),
                            phase_pass=passed,
                        )
                    except ValueError:
                        cost = 1e6
                        row = dict(sample=name, phase_pass=False, nonfinite=True)
                    costs[world] += cost
                    metrics[world].append(row)
            pair_cost = costs.reshape(population, repeats).max(axis=1)
            pair_cost[excluded_pairs] = np.inf
            order = np.flatnonzero(np.isfinite(pair_cost))
            order = order[np.argsort(pair_cost[order])]
            if not len(order):
                raise RuntimeError("No eligible finite proposals")
            winner = int(order[0])
            candidate_cost = float(pair_cost[order].max()) if fixed_parameters else float(pair_cost[winner])
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_raw = candidates[winner].copy()
                best_metrics = metrics if fixed_parameters else metrics[repeats * winner : repeats * (winner + 1)]
            elite = candidates[order[: max(4, population // 5)]]
            mu = bounded(0.3 * mu + 0.7 * elite.mean(0))
            sigma_floor = [0.005] * 6 + [0.02] * 7
            if fit_mode == "pose_and_carry_height":
                sigma_floor += [.0002]
            sigma = np.maximum(
                0.65 * sigma + 0.35 * elite.std(0), sigma_floor
            )
            report = dict(
                iteration=iteration + 1,
                best_cost=best_cost,
                best_raw_action=best_raw[:13].tolist(),
                fit_mode=fit_mode,
                objective=objective,
                fixed_parameters=fixed_parameters,
                best_carry_height_bias_m=(float(best_raw[2]) if fit_mode == "carry_height_sweep" else float(best_raw[13]) if fit_mode == "pose_and_carry_height" else None),
                carry_height_grid_results=[
                    dict(height_bias_m=float(candidates[i, 2]), worst_pair_cost=float(pair_cost[i]) if np.isfinite(pair_cost[i]) else None,
                         pair_metrics=metrics[repeats * i:repeats * (i + 1)])
                    for i in range(population)
                ] if fit_mode == "carry_height_sweep" else None,
                best_pair_metrics=best_metrics,
                worlds_per_candidate=repeats,
                native_motor_caps_verified=True,
                scored_cohort_all_finite=True,
                reset_native_warmstart_cleared=True,
                discarded_proposal_indices=excluded_pairs.tolist(),
                failed_cohort_evidence=failed_evidence,
                discarded_cohort_environment_steps=discarded_steps,
                source_frames_per_world=setup.reference.num_frames,
                selected_phase_objective_end_source_s=prefix_end,
                selected_phase_full_window_covered=complete(phase, prefix_end),
                simulated_environment_steps=simulated_steps,
            )
            journal.write(json.dumps(report, allow_nan=False) + "\n")
            journal.flush()
            print(json.dumps(report, allow_nan=False), flush=True)
            if best_metrics and all(
                m["phase_pass"] for rows in best_metrics for m in rows
            ):
                if complete(phase, prefix_end):
                    break
                if iteration + 1 < maximum_iterations:
                    prefix_end = window_end(phase, curriculum_step, prefix_end)
                    mu = best_raw.copy()
                    best_cost, best_metrics = float("inf"), None
        journal.close()
        robust = bool(
            complete(phase, prefix_end)
            and best_metrics
            and all(m["phase_pass"] for rows in best_metrics for m in rows)
        )
        if fit_mode == "carry_height_sweep":
            plan.setdefault("carry_height_bias_m", {})[phase["sample"]] = float(best_raw[2])
        else:
            plan["phase_actions"][phase["sample"]] = best_raw[:13].tolist()
            if fit_mode == "pose_and_carry_height":
                plan.setdefault("carry_height_bias_m", {})[phase["sample"]] = float(best_raw[13])
        plan.setdefault("fit_reports", {})[phase["sample"]] = dict(
            robust_phase_pass=robust,
            best_cost=best_cost,
            best_pair_metrics=best_metrics,
            candidates_repeated_worlds=worlds if fixed_parameters else repeats,
            iterations=iteration + 1,
            fit_mode=fit_mode,
            selected_phase_objective_end_source_s=prefix_end,
            selected_phase_full_window_covered=complete(phase, prefix_end),
        )
        plan.update(
            optimizer="fixed_native_controller_probe" if fixed_parameters else "bounded_carry_height_grid_native_physics" if fit_mode == "carry_height_sweep" else "bounded_CEM_native_physics",
            objective=objective,
            no_achieved_trajectory_used_as_reference=True,
            original_source_arrays_changed=False,
            physical_task_accepted=False,
            fitting_source_sha256=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        )
        (out / "phase-policy.json").write_text(
            json.dumps(plan, indent=2, allow_nan=False) + "\n"
        )
        (out / "report.json").write_text(
            json.dumps(
                dict(
                    schema="v2d.native-phase-policy-fit.v1",
                    selected_phase=phase["sample"],
                    robust_phase_pass=robust,
                    source_frames_per_world=setup.reference.num_frames,
                    iterations=iteration + 1,
                    population=population,
                    repeated_worlds=repeats,
                    simulated_environment_steps=simulated_steps,
                    PPO_training_steps=0,
                    native_motor_caps_verified=True,
                    scored_cohort_all_finite=True,
                    reset_native_warmstart_cleared=True,
                    discarded_proposal_indices=excluded_pairs.tolist(),
                    failed_cohort_evidence=failed_evidence,
                    discarded_cohort_environment_steps=discarded_steps,
                    physical_task_accepted=False,
                    selected_phase_objective_end_source_s=prefix_end,
                    selected_phase_full_window_covered=complete(phase, prefix_end),
                    curriculum_step_s=curriculum_step,
                    initial_sigma_scale=sigma_scale,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
