"""Exercise sampled and captured frame-zero resets with no VOC warm-up."""

import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import warp as wp
from flash_chord.configuration import instantiate_typed, resolved_dict
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.ppo.config import TrainingConfig
from check_training_curriculum import build
from v2d_dex3_flashchord.phase_prior import model_signature
from v2d_dex3_flashchord.reset_curriculum import assistance_contract, validate


@hydra.main(version_base="1.3", config_path="/opt/flash-chord/src/flash_chord/configs", config_name="train")
def main(cfg):
    path = Path(cfg.task.parquet).parent / "manifest.json"
    OmegaConf.update(cfg, "embodiment.config.reference_manifest_sha256", hashlib.sha256(path.read_bytes()).hexdigest(), force_add=True)
    assert assistance_contract(resolved_dict(cfg)) is None
    with wp.ScopedDevice("cuda:0"):
        env, setup = build(cfg)
        training = instantiate_typed(cfg.training, TrainingConfig)
        validate(resolved_dict(cfg).get("curriculum"), dict(zip(env.objective.term_names, env.objective.weights.numpy().tolist(), strict=True)), allow_estimated_contact_guidance=True)
        wrapped = WarpRLEnv(env)
        for iteration in (0, 500, 1499):
            wrapped.set_curriculum(training.curriculum.stage_at(iteration))
            env.reset()
            np.testing.assert_array_equal(env.timestep.numpy(), 0)
            np.testing.assert_array_equal(env.object_control_scale.numpy(), 0)
        env.capture_step()
        env.reset()
        for _ in range(24):
            env.step(wp.zeros_like(env.action_input))
            np.testing.assert_array_equal(env.object_control_scale.numpy(), 0)
            assert np.isfinite(env.state_0.body_q.numpy()).all()
        result = dict(schema="v2d.unassisted-training-reset-gpu-contract.v1",
                      input_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      native_physics_model_sha256=model_signature(setup.scene.model, 4),
                      support_shape_count=setup.scene.support_shape_count,
                      sampled_training_resets_at_frame_zero=True,
                      all_tested_stages_zero_VOC=True, captured_zero_assistance_steps=24,
                      policy_observation_dim=env.observation_dim, physical_task_accepted=False)
        Path("/outputs/unassisted-reset-gpu-contract.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
