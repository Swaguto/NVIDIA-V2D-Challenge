"""Local-metrics PPO pilot using the pinned upstream environment and checkpoint API.

The learning algorithm, networks, action schema, and evaluator are upstream.
This composition adds a JSONL logger and an untouched initial-policy checkpoint.
"""

import hashlib
import json
from pathlib import Path
from dataclasses import asdict

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import warp as wp

from flash_chord.configuration import instantiate_typed, resolved_dict
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint_metadata import (
    build_checkpoint_metadata,
    policy_schema,
    resolved_reference_config,
)
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.ppo.config import TrainingConfig
from flash_chord.training.ppo.runner import metric_definitions, run_training
from flash_chord.training.ppo.learner import create_learner
from flash_chord.training.checkpoint import save_checkpoint
from flash_chord.training.progress import ConsoleTrainingProgress


class LocalMetricLogger:
    def __init__(self, output, definitions):
        self.stream = (output / "metrics.jsonl").open("x")
        (output / "metric-definitions.json").write_text(
            json.dumps(definitions, indent=2) + "\n"
        )

    def log(self, metrics, step):
        self.stream.write(
            json.dumps(dict(environment_steps=step, **metrics), allow_nan=False) + "\n"
        )
        self.stream.flush()

    def close(self):
        self.stream.close()


def audit_scene(env, scene, output, manifest=None):
    applied_voc = env.reset_policy.applied_voc_scale.numpy()
    if not np.all(applied_voc == 0):
        raise ValueError("Unassisted pilot has nonzero object assistance")
    model = scene.model
    audit = dict(
        schema="v2d.dex3-training-scene.v1",
        world_count=scene.world_count,
        body_count=int(model.body_count),
        shape_count=int(model.shape_count),
        collision_policy=asdict(scene.collision_policy),
        hands=[
            dict(
                side=h.side,
                fingers=list(h.finger_joint_names),
                dp_frames=[f.name for f in h.dp_frames],
                contact_links=[
                    g.link_name for g in h.link_geometry if g.object_contact
                ],
            )
            for h in scene.layout.hands
        ],
        actual_applied_object_assistance=applied_voc.tolist(),
        support_shape_count=scene.support_shape_count,
        geometry_scope=(
            manifest.get(
                "collision_geometry_contract", "Frozen Newton CoACD decomposition"
            )
            if manifest is not None
            else "Geometry contract not supplied"
        ),
        source_inertias_may_be_corrected_by_newton=True,
        physical_task_accepted=False,
    )
    for name in (
        "body_mass",
        "body_inv_mass",
        "body_inertia",
        "body_flags",
        "shape_body",
        "shape_flags",
        "joint_target_ke",
        "joint_target_kd",
        "joint_effort_limit",
        "joint_armature",
        "gravity",
    ):
        value = getattr(model, name, None)
        if value is not None:
            array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite scene physics array {name}")
            audit[name] = array.tolist()
    (output / "scene-audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )


@hydra.main(
    version_base="1.3",
    config_path="/opt/flash-chord/src/flash_chord/configs",
    config_name="train",
)
def main(cfg: DictConfig):
    if (
        cfg.logging.mode != "disabled"
        or cfg.env.voc_scale != 0
        or cfg.voc_scale_override != 0
        or cfg.curriculum is not None
    ):
        raise ValueError(
            "Pilot requires local logging, no object assistance, and no curriculum"
        )
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    input_manifest_path = Path(cfg.task.parquet).parent / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text())
    input_sha = hashlib.sha256(input_manifest_path.read_bytes()).hexdigest()
    OmegaConf.update(
        cfg, "embodiment.config.reference_manifest_sha256", input_sha, force_add=True
    )
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    training = instantiate_typed(cfg.training, TrainingConfig)
    if training.resume:
        raise ValueError("Use a fresh output for the initial controlled pilot")
    with wp.ScopedDevice(training.device):
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
        env.reset()
        audit_scene(env, setup.scene, output, input_manifest)
        jax_env = WarpRLEnv(env)
        config = resolved_dict(cfg)
        config["task"]["parquet"] = str(setup.parquet_path)
        config["reference"] = resolved_reference_config(
            setup.reference, str(setup.parquet_path)
        )
        metadata = build_checkpoint_metadata(
            config, policy_schema(env.action, env.observation_strategy)
        )
        metadata["dex3_input_manifest_sha256"] = hashlib.sha256(
            (setup.parquet_path.parent / "manifest.json").read_bytes()
        ).hexdigest()
        metadata["dex3_runtime_patches_sha256"] = hashlib.sha256(
            Path("/opt/flash-chord/dex3-extension-patches.json").read_bytes()
        ).hexdigest()
        metadata["dex3_image_context_sha256"] = hashlib.sha256(
            Path("/opt/dex3-extension/image-context-manifest.json").read_bytes()
        ).hexdigest()
        metadata["dex3_pilot_scope"] = (
            "full source episode with inferred robot targets; original source confidence masks preserved; unassisted development experiment"
            if input_manifest["schema"] == "v2d.dex3-training-task.v3"
            else "confident development window; unassisted; not full episode acceptance"
        )
        initial = create_learner(jax_env, training)
        save_checkpoint(
            output / "model_0.safetensors",
            initial,
            metadata={**metadata, "iteration": 0, "environment_steps": 0},
        )
        logger = LocalMetricLogger(output, metric_definitions(jax_env))
        learner = run_training(
            jax_env,
            training,
            logger=logger,
            checkpoint_dir=output,
            log_interval=1,
            voc_scale_override=0.0,
            checkpoint_metadata=metadata,
            progress=ConsoleTrainingProgress(
                training.max_iterations * training.world_count * training.rollout_steps,
                output,
            ),
        )
        print(
            f"Completed {int(learner.iteration)} iterations; results: {output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
