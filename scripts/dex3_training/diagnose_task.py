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
from flash_chord.training.environment import WarpRLEnv
from train import audit_scene


class ReadbackEnv:
    """Record native solver state without changing actions, contacts or reset rules."""

    def __init__(self, wrapped, env):
        self.wrapped, self.env = wrapped, env
        self.body, self.joints, self.motor = [], [], []
        model = getattr(env.solver, "mj_model", None)
        self.ranges = getattr(model, "actuator_forcerange", None)
        self.limited = getattr(model, "actuator_forcelimited", None)

    def step(self, action):
        result = self.wrapped.step(action)
        self.body.append(self.env.state_0.body_q.numpy().copy())
        self.joints.append(self.env.state_0.joint_q.numpy().copy())
        data = getattr(self.env.solver, "mjw_data", None)
        force = getattr(data, "actuator_force", None)
        if force is not None:
            self.motor.append(force.numpy().copy())
        return result

    def save(self, output):
        arrays = dict(
            body_poses=np.asarray(self.body), joint_positions=np.asarray(self.joints)
        )
        verification = dict(
            native_actuator_force_readback_available=bool(self.motor),
            native_capped_drive_verified=False,
        )
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
        np.savez_compressed(output / "native-state-readback.npz", **arrays)
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
        audit_scene(env, setup.scene, output)
        jax_env = WarpRLEnv(env, publish_diagnostics=True, publish_step_metrics=True)
        readback = ReadbackEnv(jax_env, env)
        observation = jnp.asarray(
            env.observation.numpy().reshape(env.world_count, env.observation_dim)
        )
        zero = jnp.zeros((env.world_count, env.action.action_dim), dtype=jnp.float32)

        def policy(_state, _observation, key):
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
            source_manifest_sha256=hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            learned_policy=False,
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
