"""Record a full unassisted zero-residual baseline in the PPO physics runtime.

This exercises the repaired reference and motor controller before learning. It
is a diagnostic controller rollout, not a trained policy or official submission.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import warp as wp

from flash_chord.configuration import instantiate_typed, resolved_dict
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import (
    apply_evaluation_composition,
    evaluation_thresholds,
    reference_tracking_criteria,
    score_reference_tracking,
    score_rollout,
)
from flash_chord.evaluation.rollout import (
    record_object_pose_rollout,
    write_object_trajectories_parquet,
)
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.runtime.contact import read_contacts_w, contact_gaps_w
from flash_chord.training.environment import WarpRLEnv
from train import audit_scene
from v2d_dex3_flashchord.phase_prior import model_signature


class ReadbackEnv:
    """Record native solver state without changing actions, contacts or reset rules."""

    def __init__(self, wrapped, env):
        self.wrapped, self.env = wrapped, env
        self.body, self.joints, self.motor = [], [], []
        self.joint_motor = []
        self.contacts = []
        self.contact_gaps = []
        self.motor_target, self.feedback_weight = [], []
        self.velocity_target = []
        self.applied_motor_target, self.direct_effort = [], []
        self.native_model = {}
        model = getattr(env.solver, "mj_model", None)
        self.solver_options = {}
        option = getattr(model, "opt", None)
        for name in (
            "cone",
            "impratio",
            "solver",
            "integrator",
            "iterations",
            "ls_iterations",
            "timestep",
            "enableflags",
            "disableflags",
        ):
            value = getattr(option, name, None)
            if value is not None:
                self.native_model["option_" + name] = np.asarray(value).copy()
                self.solver_options[name] = np.asarray(value).item()
        for name in (
            "geom_type",
            "geom_condim",
            "geom_priority",
            "geom_bodyid",
            "geom_dataid",
            "geom_pos",
            "geom_quat",
            "geom_size",
            "geom_friction",
            "geom_solref",
            "geom_solimp",
            "geom_margin",
            "geom_gap",
            "mesh_vert",
            "mesh_vertadr",
            "mesh_vertnum",
            "mesh_pos",
            "mesh_quat",
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
            "dof_armature",
        ):
            value = getattr(model, name, None)
            if value is not None:
                self.native_model[name] = np.asarray(value).copy()
        self.ranges = getattr(model, "actuator_forcerange", None)
        self.limited = getattr(model, "actuator_forcelimited", None)
        self.joint_ranges = getattr(model, "jnt_actfrcrange", None)
        self.joint_limited = getattr(model, "jnt_actfrclimited", None)
        self.joint_dofadr = getattr(model, "jnt_dofadr", None)

    def step(self, action):
        result = self.wrapped.step(action)
        self.body.append(self.env.state_0.body_q.numpy().copy())
        self.joints.append(self.env.state_0.joint_q.numpy().copy())
        self.motor_target.append(self.env.action.joint_target.numpy().copy())
        self.velocity_target.append(self.env.control.joint_target_vel.numpy().copy())
        self.applied_motor_target.append(
            self.env.control.joint_target_pos.numpy().copy()
        )
        self.direct_effort.append(self.env.control.joint_f.numpy().copy())
        feedback = getattr(self.env.action, "actual_feedback", None)
        if feedback is not None:
            self.feedback_weight.append(feedback.numpy().copy())
        data = getattr(self.env.solver, "mjw_data", None)
        force = getattr(data, "actuator_force", None)
        if force is not None:
            self.motor.append(force.numpy().copy())
        force = getattr(data, "qfrc_actuator", None)
        if force is not None:
            self.joint_motor.append(force.numpy().copy())
        self.contacts.append(
            read_contacts_w(self.env.model, self.env.state_0, self.env.contacts)
        )
        self.contact_gaps.append(
            contact_gaps_w(self.env.model, self.env.state_0, self.env.contacts)
        )
        return result

    def save(self, output):
        # Newton updates the GPU step size during step(), independently of the
        # CPU construction model's default timestep.
        gpu_model = getattr(self.env.solver, "mjw_model", None)
        timestep = getattr(getattr(gpu_model, "opt", None), "timestep", None)
        if timestep is not None:
            dt = (
                timestep.numpy() if hasattr(timestep, "numpy") else np.asarray(timestep)
            )
            dt = np.asarray(dt).ravel()
            self.solver_options["runtime_gpu_timestep_s"] = dt.tolist()
            self.solver_options["runtime_step_matches_config"] = bool(
                np.isfinite(dt).all() and np.allclose(dt, self.env.sim_dt)
            )
        if "timestep" in self.solver_options:
            self.solver_options["construction_timestep_s"] = self.solver_options.pop(
                "timestep"
            )
        arrays = dict(
            body_poses=np.asarray(self.body),
            joint_positions=np.asarray(self.joints),
            motor_target=np.asarray(self.motor_target),
            motor_velocity_target=np.asarray(self.velocity_target),
            applied_motor_position_target=np.asarray(self.applied_motor_target),
            direct_joint_effort=np.asarray(self.direct_effort),
        )
        if self.feedback_weight:
            arrays["measured_feedback_weight"] = np.asarray(self.feedback_weight)
        verification = dict(
            native_actuator_force_readback_available=bool(self.motor),
            native_capped_drive_verified=False,
            native_solver_options=self.solver_options,
        )
        direct = np.asarray(self.direct_effort).reshape(
            len(self.direct_effort), self.env.world_count, -1
        )
        wrist = self.env.action.buffers.wrist_ori_dof_ids.numpy()
        caps = self.env.action.wrist.tau_max.numpy()
        actual = direct[..., wrist].reshape(len(direct), self.env.world_count, -1, 3)
        verification["direct_wrist_effort_within_shared_cap"] = bool(
            np.isfinite(actual).all()
            and np.all(abs(actual) <= caps[None, None, :, None] + 1e-5)
        )
        verification["maximum_actual_direct_wrist_effort_Nm"] = float(abs(actual).max())
        if self.motor and self.ranges is not None and self.limited is not None:
            force = np.asarray(self.motor)
            ranges, limited = np.asarray(self.ranges), np.asarray(self.limited, bool)
            arrays.update(
                native_actuator_force=force,
                native_actuator_forcerange=ranges,
                native_actuator_forcelimited=limited,
            )
            flat = force.reshape(-1, ranges.shape[0])
            within = bool(
                np.isfinite(flat).all()
                and np.all(flat[:, limited] >= ranges[limited, 0] - 1e-5)
                and np.all(flat[:, limited] <= ranges[limited, 1] + 1e-5)
            )
            finger = (
                limited
                & np.isclose(ranges[:, 0], -0.76)
                & np.isclose(ranges[:, 1], 0.76)
            )
            verification.update(
                limited_finger_actuators=int(finger.sum()),
                all_limited_actuator_forces_within_native_bounds=within,
                maximum_actual_finger_actuator_force_Nm=float(
                    abs(flat[:, finger]).max()
                )
                if finger.any()
                else None,
                native_capped_drive_verified=bool(within and finger.sum() >= 14),
            )
        if (
            self.joint_motor
            and self.joint_ranges is not None
            and self.joint_limited is not None
        ):
            force = np.asarray(self.joint_motor)
            ranges, limited = (
                np.asarray(self.joint_ranges),
                np.asarray(self.joint_limited, bool),
            )
            addresses = np.asarray(self.joint_dofadr, int)
            finger = (
                limited
                & np.isclose(ranges[:, 0], -0.76)
                & np.isclose(ranges[:, 1], 0.76)
            )
            values = force[..., addresses[finger]]
            within = bool(np.isfinite(values).all() and np.all(abs(values) <= 0.76001))
            limited_values = force[..., addresses[limited]]
            all_within = bool(
                np.isfinite(limited_values).all()
                and np.all(limited_values >= ranges[limited, 0] - 1e-5)
                and np.all(limited_values <= ranges[limited, 1] + 1e-5)
            )
            root = (
                limited & np.isclose(ranges[:, 0], -60) & np.isclose(ranges[:, 1], 60)
            )
            arrays.update(
                native_combined_joint_motor_force=force,
                joint_motor_range=ranges,
                joint_motor_limited=limited,
                joint_dof_addresses=addresses,
            )
            verification.update(
                limited_finger_joints=int(finger.sum()),
                limited_root_position_joints=int(root.sum()),
                all_native_limited_joint_forces_within_bounds=all_within,
                maximum_actual_combined_finger_motor_torque_Nm=float(abs(values).max())
                if finger.any()
                else None,
                native_capped_drive_verified=bool(
                    finger.sum() >= 14 and within and all_within
                ),
                verification_scope="Combined joint actuator force after native joint limits; separate position/velocity actuator components need not be individually capped",
            )
        np.savez_compressed(output / "native-state-readback.npz", **arrays)
        np.savez_compressed(output / "native-model-readback.npz", **self.native_model)
        contact_arrays = {
            name: np.concatenate([getattr(c, name) for c in self.contacts], axis=0)
            for name in (
                "body0_ids",
                "body1_ids",
                "contact_pos_w",
                "contact_normal_w",
                "contact_force_w",
                "contact_torque_w",
            )
        }
        np.savez_compressed(
            output / "native-contacts-readback.npz",
            frame_offsets=np.cumsum([0] + [len(c) for c in self.contacts]),
            signed_gap_m=np.concatenate(self.contact_gaps),
            **contact_arrays,
        )
        verification["contact_readback_scope"] = (
            "Measured contacts at the final physics substep of each source frame; "
            "not continuous contact coverage or a support certificate"
        )
        (output / "body-labels.json").write_text(
            json.dumps(list(self.env.model.body_label), indent=2) + "\n"
        )
        return verification


@hydra.main(
    version_base="1.3",
    config_path="/opt/flash-chord/src/flash_chord/configs",
    config_name="train",
)
def main(cfg):
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(cfg.task.parquet).parent / "manifest.json"
    OmegaConf.update(
        cfg,
        "embodiment.config.reference_manifest_sha256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        force_add=True,
    )
    config = resolved_dict(cfg)
    thresholds = evaluation_thresholds(config)
    criteria = reference_tracking_criteria(config)
    config = apply_evaluation_composition(
        config, world_count=int(cfg.scene.world_count)
    )
    cfg = OmegaConf.create(config)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with wp.ScopedDevice("cuda:0"):
        env_config = instantiate_typed(cfg.env, RLEnvConfig)
        embodiment = instantiate_typed(cfg.embodiment, Embodiment)
        collision = instantiate_typed(cfg.collision, CollisionPolicy)
        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=env_config.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=cfg.scene.world_count,
            include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            contact_friction=cfg.scene.contact_friction,
        )
        env = RLEnv(setup.scene, setup.reference, config=env_config)
        env.reset()
        env.capture_step()
        env.reset(frame_id=0)
        audit_scene(env, setup.scene, output, json.loads(manifest_path.read_text()))
        jax_env = WarpRLEnv(env, publish_diagnostics=True, publish_step_metrics=True)
        readback = ReadbackEnv(jax_env, env)
        observation = jnp.asarray(
            env.observation.numpy().reshape(env.world_count, env.observation_dim)
        )
        zero = jnp.zeros((env.world_count, env.action.action_dim), dtype=jnp.float32)
        phase_policy = None
        plan_path = cfg.get("residual_plan")
        plan_sha = None
        embedded = cfg.action.get("phase_policy_prior")
        if embedded:
            plan_sha = hashlib.sha256(
                (manifest_path.parent / embedded).read_bytes()
            ).hexdigest()
        if embedded and plan_path:
            raise ValueError(
                "Independent replay cannot apply the same phase prior twice"
            )
        if plan_path:
            from v2d_dex3_flashchord.phase_prior import schedule
            from v2d_dex3_flashchord.sampling import resample_targets

            path = Path(plan_path)
            plan = json.loads(path.read_text())
            if (
                plan["input_manifest_sha256"]
                != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            ):
                raise ValueError("Diagnostic policy plan belongs to other inputs")
            with np.load(manifest_path.parent / "targets.npz") as f:
                data = resample_targets(
                    {k: f[k].copy() for k in f.files}, setup.reference.num_frames
                )
            phase_policy = schedule(
                json.loads((manifest_path.parent / "task-grips.json").read_text()),
                plan,
                list(env.action.sides),
                {
                    s: setup.scene.layout.hand(s).finger_joint_names
                    for s in env.action.sides
                },
                data["timestamps_s"],
            )
            plan_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cursor = [0]

        def policy(_state, _observation, key):
            if phase_policy is not None:
                action = jnp.broadcast_to(
                    jnp.asarray(phase_policy[cursor[0]]),
                    (env.world_count, env.action.action_dim),
                )
                cursor[0] += 1
                return action, key
            return zero, key

        rollout = record_object_pose_rollout(
            env,
            readback,
            policy,
            None,
            None,
            observation,
            reference=setup.reference,
            step_count=setup.reference.num_frames,
        )
        metrics = asdict(score_rollout(rollout, thresholds))
        reference_metrics = asdict(score_reference_tracking(rollout, criteria))
        write_object_trajectories_parquet(
            output / "objects.parquet",
            rollout,
            provenance={
                "controller": "zero_policy_residual_with_inferred_motor_feedforward",
                "input_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            },
        )
        report = dict(
            schema="v2d.dex3-full-task-training-diagnostic.v1",
            metrics=metrics,
            reference_tracking=reference_metrics,
            frame_count=setup.reference.num_frames,
            control_fps=float(env_config.sim.fps),
            physics_fps=float(env_config.sim.physics_fps),
            source_frame_count=len(
                np.load(manifest_path.parent / "targets.npz")["frame_indices"]
            ),
            source_fps=float(setup.reference.metadata.source_fps),
            source_manifest_sha256=hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            learned_policy=False,
            action_control_contract_sha256=getattr(
                env.action, "policy_control_contract_sha256", None
            ),
            native_physics_model_sha256=getattr(
                env.action,
                "native_physics_model_sha256",
                model_signature(env.model, env.world_count),
            ),
            fitted_phase_policy_sha256=plan_sha,
            virtual_object_assistance=0.0,
            source_motion_speed=float(cfg.task.motion_speed),
            physical_task_accepted=False,
            official_evaluation_score_claimed=False,
            native_motor_verification=readback.save(output),
        )
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
