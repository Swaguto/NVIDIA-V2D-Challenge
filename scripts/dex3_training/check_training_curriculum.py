"""GPU gate for training assistance, residual reach and evaluation composition."""

import json
import hashlib
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
import warp as wp
from flash_chord.configuration import instantiate_typed, resolved_dict
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.ppo.config import TrainingConfig
from omegaconf import OmegaConf
from v2d_dex3_flashchord.phase_prior import model_signature
from v2d_dex3_flashchord.reset_curriculum import assistance_contract, validate


def build(cfg):
    config = instantiate_typed(cfg.env, RLEnvConfig)
    setup = setup_scene(
        parquet=cfg.task.parquet, control_fps=config.sim.fps,
        motion_speed=cfg.task.motion_speed,
        embodiment=instantiate_typed(cfg.embodiment, Embodiment),
        collision=instantiate_typed(cfg.collision, CollisionPolicy),
        world_count=4, include_support=cfg.scene.support,
        decompose_objects=cfg.scene.decompose_objects,
        contact_friction=cfg.scene.contact_friction,
    )
    return RLEnv(setup.scene, setup.reference, config=config), setup


def main():
    results, physical = {}, []
    with initialize_config_dir(config_dir="/opt/flash-chord/src/flash_chord/configs", version_base="1.3"):
        for name in ("dex3_grip_action_baseline", "dex3_grip_reachable", "dex3_grip_assisted"):
            cfg = compose(config_name="train", overrides=[
                "hydra.searchpath=[file:///opt/dex3-extension/configs]",
                "experiment=" + name, "scene.world_count=4", "training.world_count=4",
            ])
            OmegaConf.update(cfg, "embodiment.config.reference_manifest_sha256", hashlib.sha256(
                (Path(cfg.task.parquet).parent / "manifest.json").read_bytes()
            ).hexdigest(), force_add=True)
            assist = assistance_contract(resolved_dict(cfg))
            env, setup = build(cfg)
            training = instantiate_typed(cfg.training, TrainingConfig)
            validate(resolved_dict(cfg).get("curriculum"), dict(zip(
                env.objective.term_names, env.objective.weights.numpy().tolist(), strict=True,
            )), allow_training_assistance=assist is not None)
            physical.append(model_signature(setup.scene.model, 4))
            wrapped = WarpRLEnv(env)
            if assist:
                wrapped.set_curriculum(training.curriculum.stage_at(0))
            env.reset(frame_id=40)
            before = env.state_0.body_q.numpy().copy()
            # Constant +30mm request; test actual upstream filtering/clipping.
            values = np.zeros((4, env.action.action_dim), np.float32)
            values[:, [2, 15]] = 3.0
            action = wp.array(values.ravel())
            for _ in range(32):
                env.action.process(action, env.timestep, env.state_0)
            residual = env.action.buffers.filtered_action.numpy().reshape(4, -1)[:, [2, 15]]
            expected = min(0.03, cfg.action.wrist_pos_clip)
            np.testing.assert_allclose(residual, expected, atol=1e-7)
            np.testing.assert_array_equal(env.state_0.body_q.numpy(), before)
            entry = dict(actual_filtered_height_residual_m=float(residual[0, 0]))
            if assist:
                ages, scales, clocks = [], [], []
                for iteration in (0, 250, 500, 750, 1000, 1999):
                    stage = training.curriculum.stage_at(iteration)
                    wrapped.set_curriculum(stage)
                    env.reset()  # sampled training reset exercises the warm-up
                    start = env.timestep.numpy().copy()
                    first = env.object_control_scale.numpy().copy()
                    np.testing.assert_array_equal(first, 1.0)
                    for age in range(1, 21):
                        env.reset_policy.advance(env.done, env.truncation, env.timestep, env.episode_step, env.voc_scale)
                        expected_scale = 1 + (stage.voc_scale - 1) * min(age / 20, 1)
                        np.testing.assert_allclose(env.object_control_scale.numpy(), expected_scale, atol=1e-6)
                        np.testing.assert_array_equal(env.timestep.numpy(), start + (age >= 20))
                    ages.append(env.reset_policy.steps_since_reset.numpy().tolist())
                    scales.append(stage.voc_scale)
                    clocks.append(start.tolist())
                assert clocks[-1] == [0] * 4
                entry.update(actual_target_scales=scales, actual_reset_frames=clocks, reset_age=ages)
            # Use the real official composition, then exercise captured physics.
            evaluation = apply_evaluation_composition(resolved_dict(cfg), world_count=4)
            evaluation_env, _ = build(OmegaConf.create(evaluation))
            evaluation_env.reset(frame_id=0)
            evaluation_env.capture_step()
            evaluation_env.reset(frame_id=0)
            for _ in range(4):
                evaluation_env.step(wp.zeros_like(evaluation_env.action_input))
                assert np.all(evaluation_env.object_control_scale.numpy() == 0)
                assert np.isfinite(evaluation_env.state_0.body_q.numpy()).all()
                assert np.isfinite(evaluation_env.observation.numpy()).all()
            assert evaluation_env.config.voc.max_force == 0
            assert evaluation_env.config.voc.max_torque == 0
            assert evaluation_env.config.reset.voc_decay_steps == 0
            entry["captured_unassisted_evaluation_steps_verified"] = 4
            results[name] = entry
    assert len(set(physical)) == 1, "Comparison changed physical parameters"
    results["same_physical_model_sha256"] = physical[0]
    Path("/outputs/training-curriculum-gpu-contract.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
