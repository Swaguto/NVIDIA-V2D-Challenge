"""Local-metrics PPO pilot using the pinned upstream environment and checkpoint API.

The learning algorithm, networks, action schema, and evaluator are upstream.
This composition adds a JSONL logger and an untouched initial-policy checkpoint.
"""

import hashlib
import json
from pathlib import Path
from dataclasses import asdict, replace

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
from flash_chord.training.checkpoint import save_checkpoint, read_checkpoint_metadata
from flash_chord.training.progress import ConsoleTrainingProgress
from v2d_dex3_flashchord.reset_curriculum import validate as validate_reset_curriculum
from v2d_dex3_flashchord.reset_curriculum import assistance_contract
from v2d_dex3_flashchord.training_metrics import nonfinite_fields
from v2d_dex3_flashchord.resume_contract import validate as validate_resume_contract


class LocalMetricLogger:
    def __init__(self, output, definitions, env=None):
        self.output, self.env = output, env
        definitions = dict(definitions)
        if env is not None:
            definitions.update({
                "assistance/applied_scale_mean_at_log_boundary": "Mean actual per-world VOC scale at the logger boundary; includes reset settling, not a rollout average.",
                "assistance/nonzero_world_fraction_at_log_boundary": "Fraction of worlds with nonzero actual VOC at the logger boundary; includes reset settling.",
            })
        self.failure_policy = getattr(env, "numerical_failure_policy", "abort")
        if self.failure_policy == "terminate":
            definitions = dict(definitions)
            definitions["numerical_failures/cumulative_world_transitions"] = (
                "Invalid simulated world-transitions ended as numerical failures, with raw evidence retained."
            )
            (output / "numerical-failure-contract.json").write_text(
                json.dumps(
                    dict(
                        training_only=True,
                        policy="terminate",
                        raw_failure_evidence_preserved=True,
                        invalid_transition_reward="Existing termination penalty only",
                        invalid_tracking_diagnostic_sentinel_m_rad=1000.0,
                        training_tracking_means_include_conservative_invalid_state_sentinel=True,
                        physics_states_repaired=False,
                        reset_discards_previous_episode_native_acceleration_warmstart=True,
                        repeated_failure_after_reset_aborts_at_next_logger_boundary=True,
                        evaluation_behavior_changed=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
        self.stream = (output / "metrics.jsonl").open("x")
        (output / "metric-definitions.json").write_text(
            json.dumps(definitions, indent=2) + "\n"
        )

    def log(self, metrics, step):
        invalid = nonfinite_fields(metrics)
        trace = getattr(self.env, "transition_trace", None)
        trace_count = int(trace.counts.numpy().sum()) if trace is not None else 0
        trace_report = None
        if (
            trace_count
            and not (self.output / "first-nonfinite-transitions.json").exists()
        ):
            # Persist at the first logger boundary, rather than relying on a
            # clean process shutdown after a potentially hours-long run.
            trace_report = trace.save(self.output)
        repeated = (
            int(self.env.maximum_failure_streak.numpy().max())
            if self.failure_policy == "terminate"
            else 0
        )
        if (
            invalid
            or (trace_count and self.failure_policy != "terminate")
            or repeated >= 2
        ):
            counts = {}
            if trace is not None and trace_report is None:
                trace_report = trace.save(
                    self.output,
                    stem=(
                        "final-nonfinite-transitions"
                        if (self.output / "first-nonfinite-transitions.json").exists()
                        else "first-nonfinite-transitions"
                    ),
                )
            if self.env is not None:
                env = self.env
                arrays = dict(
                    body_q=env.state_0.body_q.numpy(),
                    body_qd=env.state_0.body_qd.numpy(),
                    joint_q=env.state_0.joint_q.numpy(),
                    joint_qd=env.state_0.joint_qd.numpy(),
                    raw_action=env.action.buffers.raw_action.numpy(),
                    observation=env.observation.numpy(),
                    objective_terms=env.objective.terms.numpy(),
                    objective_score=env.objective.score.numpy(),
                    timestep=env.timestep.numpy(),
                )
                counts = {
                    name: int(np.sum(~np.isfinite(a))) for name, a in arrays.items()
                }
                with (self.output / "nonfinite-training-state.npz").open(
                    "xb"
                ) as stream:
                    np.savez_compressed(stream, **arrays)
            (self.output / "nonfinite-training-metrics.json").write_text(
                json.dumps(
                    dict(
                        environment_steps=step,
                        nonfinite_metrics=invalid,
                        nonfinite_state_counts=counts,
                        first_failed_transitions=trace_report,
                        maximum_consecutive_failed_transitions=repeated,
                        training_aborted=True,
                    ),
                    indent=2,
                )
                + "\n"
            )
            raise FloatingPointError(
                "Nonfinite training metrics: " + ", ".join(invalid)
                if invalid
                else "Repeated numerical failure immediately after reset"
                if repeated >= 2
                else "Nonfinite transition captured before automatic reset"
            )
        if self.failure_policy == "terminate":
            metrics = dict(
                metrics,
                **{"numerical_failures/cumulative_world_transitions": trace_count},
            )
        if self.env is not None:
            metrics = dict(metrics)
            scales = self.env.reset_policy.applied_voc_scale.numpy()
            metrics["assistance/applied_scale_mean_at_log_boundary"] = float(scales.mean())
            metrics["assistance/nonzero_world_fraction_at_log_boundary"] = float(np.mean(scales > 0))
        self.stream.write(
            json.dumps(dict(environment_steps=step, **metrics), allow_nan=False) + "\n"
        )
        self.stream.flush()

    def close(self):
        trace = getattr(self.env, "transition_trace", None)
        if trace is not None:
            stem = (
                "final-nonfinite-transitions"
                if (self.output / "first-nonfinite-transitions.json").exists()
                else "first-nonfinite-transitions"
            )
            if not (self.output / (stem + ".json")).exists():
                trace.save(self.output, stem=stem)
        self.stream.close()


def audit_scene(env, scene, output, manifest=None, training_assistance=None):
    applied_voc = env.reset_policy.applied_voc_scale.numpy()
    if training_assistance is None and not np.all(applied_voc == 0):
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
        training_assistance=training_assistance,
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
    assist = assistance_contract(resolved_dict(cfg))
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
        env_type = RLEnv
        failure_policy = cfg.get("numerical_failure_policy", "abort")
        if failure_policy not in ("abort", "terminate"):
            raise ValueError("Unknown explicit numerical-failure policy")
        if failure_policy == "terminate":
            from v2d_dex3_flashchord.numerical_failure import NumericalFailureRLEnv

            env_type = NumericalFailureRLEnv
        elif cfg.get("trace_training_transitions", False):
            from v2d_dex3_flashchord.transition_trace import TracedRLEnv

            env_type = TracedRLEnv
        env = env_type(setup.scene, setup.reference, config=env_config)
        validate_reset_curriculum(
            resolved_dict(cfg).get("curriculum"),
            dict(
                zip(
                    env.objective.term_names,
                    env.objective.weights.numpy().tolist(),
                    strict=True,
                )
            ),
            allow_training_assistance=assist is not None,
        )
        if assist is not None:
            # Bind stage zero before reset/capture and learner initialization.
            WarpRLEnv(env).set_curriculum(training.curriculum.stage_at(0))
            (output / "training-assistance-contract.json").write_text(
                json.dumps(assist, indent=2) + "\n"
            )
        env.reset()
        env.capture_step()
        env.reset()
        if hasattr(env, "transition_trace"):
            env.transition_trace.clear()
        audit_scene(env, setup.scene, output, input_manifest, assist)
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
        if cfg.curriculum is not None:
            metadata["dex3_pilot_scope"] = (
                "Reference-state initialization during training with zero VOC; final stage uses only frame-zero starts. Full-task success requires separate complete evaluation from frame zero."
                if assist is None else
                "Training-only decaying VOC with reset stabilization; final target scale zero, reset assistance remains training-only. Evaluation must start at frame zero with zero VOC and no settling."
            )
        if assist is not None:
            metadata["dex3_training_assistance_json"] = json.dumps(assist, sort_keys=True)
        actor_checkpoint = cfg.get("actor_initialization_checkpoint")
        if actor_checkpoint and training.resume:
            raise ValueError("Actor initialization is a new experiment, not a resume")
        runner_training = training
        if training.resume:
            context = cfg.get("resume_context_manifest")
            if not context:
                raise ValueError("Resume requires the parent code context manifest")
            lineage = validate_resume_contract(
                read_checkpoint_metadata(training.checkpoint),
                metadata,
                Path(context),
                Path("/opt/dex3-extension/image-context-manifest.json"),
                Path(training.checkpoint),
            )
            if training.curriculum is not None:
                jax_env.set_curriculum(
                    training.curriculum.stage_at(lineage["parent_iteration"])
                )
            lineage["initial_reset_matches_active_curriculum"] = True
            (output / "resume-lineage.json").write_text(
                json.dumps(lineage, indent=2) + "\n"
            )
            metadata["dex3_resume_lineage_json"] = json.dumps(lineage, sort_keys=True)
        else:
            initial = create_learner(jax_env, training)
            if actor_checkpoint:
                from v2d_dex3_flashchord.actor_initialization import (
                    initialize, validate as validate_actor_initialization,
                )

                context = cfg.get("actor_initialization_context_manifest")
                if not context:
                    raise ValueError("Actor initialization requires its parent code context")
                lineage = validate_actor_initialization(
                    read_checkpoint_metadata(actor_checkpoint), metadata, Path(context),
                    Path("/opt/dex3-extension/image-context-manifest.json"), Path(actor_checkpoint),
                )
                initial = initialize(initial, actor_checkpoint, len(training.network.actor_hidden_dims))
                (output / "actor-initialization.json").write_text(json.dumps(lineage, indent=2) + "\n")
                metadata["dex3_actor_initialization_json"] = json.dumps(lineage, sort_keys=True)
            save_checkpoint(
                output / "model_0.safetensors",
                initial,
                metadata={**metadata, "iteration": 0, "environment_steps": 0},
            )
            if actor_checkpoint:
                # The upstream loop loads this new, zero-counter learner. Its
                # optimizer, critic and RNG remain fresh; no parent step count
                # is presented as work performed by the new experiment.
                runner_training = replace(training, resume=True, checkpoint=str(output / "model_0.safetensors"))
        logger = LocalMetricLogger(output, metric_definitions(jax_env), env)
        learner = run_training(
            jax_env,
            runner_training,
            logger=logger,
            checkpoint_dir=output,
            log_interval=1,
            voc_scale_override=cfg.voc_scale_override,
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
