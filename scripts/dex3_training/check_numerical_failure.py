"""GPU contract for explicit failed training transitions and real automatic reset."""

from pathlib import Path
import hashlib, json

import hydra
import numpy as np
from omegaconf import OmegaConf
import warp as wp
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnvConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from v2d_dex3_flashchord.numerical_failure import (
    NumericalFailureRLEnv,
    clear_reset_warmstart,
    track_failure_streak,
    terminate_invalid_transition,
)


@wp.kernel
def inject_invalid_score(
    enabled: wp.array(dtype=wp.int32),
    bad_value: wp.array(dtype=wp.float32),
    score: wp.array(dtype=wp.float32),
):
    if enabled[0] != 0:
        score[0] = bad_value[0]


@wp.kernel
def inject_invalid_warmstart(
    bad_value: wp.array(dtype=wp.float32), warmstart: wp.array2d(dtype=wp.float32)
):
    dof = wp.tid()
    warmstart[0, dof] = bad_value[0]


def kernel_check():
    def a(values, dtype=wp.float32):
        return wp.array(np.asarray(values), dtype=dtype)

    flags = a([23, 0], wp.int32)
    done = a([0, 0], wp.int32)
    truncated = a([1, 1], wp.int32)
    wrist = a([1, 0], wp.int32)
    objects = a([1, 0], wp.int32)
    numerical = a([0, 0], wp.int32)
    objective = a([np.nan, 0, np.nan, 0.2, 0.3, 0.4])
    score = a([np.nan, 1.25])
    weights = a([1, 1, -100])
    diagnostics = a([np.nan] * 7 + [0, 0, 0, 0.1, 0.2, 0.3, 0.4])
    errors = [a([np.nan, x]) for x in [0.1, 0.2, 0.3, 0.4]]
    wp.launch(
        terminate_invalid_transition,
        dim=2,
        inputs=[
            flags,
            3,
            2,
            0.05,
            weights,
            done,
            truncated,
            wrist,
            objects,
            numerical,
            objective,
            score,
            diagnostics,
            *errors,
        ],
    )
    assert done.numpy().tolist() == [1, 0] and truncated.numpy().tolist() == [0, 1]
    assert numerical.numpy().tolist() == [1, 0]
    assert wrist.numpy().tolist() == [0, 0] and objects.numpy().tolist() == [0, 0]
    np.testing.assert_allclose(objective.numpy(), [0, 0, 1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(score.numpy(), [-5, 1.25])
    np.testing.assert_allclose(
        diagnostics.numpy(),
        [0, 0, 1, 1000, 1000, 1000, 1000, 0, 0, 0, 0.1, 0.2, 0.3, 0.4],
    )
    streak, maximum = a([0, 0], wp.int32), a([0, 0], wp.int32)
    for _ in range(2):
        wp.launch(track_failure_streak, dim=2, inputs=[flags, streak, maximum])
    assert maximum.numpy().tolist() == [2, 0]
    flags.zero_()
    wp.launch(track_failure_streak, dim=2, inputs=[flags, streak, maximum])
    assert streak.numpy().tolist() == [0, 0]
    assert maximum.numpy().tolist() == [2, 0]


@hydra.main(
    version_base="1.3",
    config_path="/opt/flash-chord/src/flash_chord/configs",
    config_name="train",
)
def main(cfg):
    with wp.ScopedDevice("cuda:0"):
        kernel_check()
        manifest = Path(cfg.task.parquet).parent / "manifest.json"
        OmegaConf.update(
            cfg,
            "embodiment.config.reference_manifest_sha256",
            hashlib.sha256(manifest.read_bytes()).hexdigest(),
            force_add=True,
        )
        config = instantiate_typed(cfg.env, RLEnvConfig)
        embodiment = instantiate_typed(cfg.embodiment, Embodiment)
        collision = instantiate_typed(cfg.collision, CollisionPolicy)
        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=config.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=2,
            include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            contact_friction=cfg.scene.contact_friction,
        )
        env = NumericalFailureRLEnv(setup.scene, setup.reference, config=config)
        env.set_reset_to_first_frame_probability(1.0)
        evaluate = env.objective.evaluate
        bad_value = wp.array(np.array([np.nan], dtype=np.float32))
        enabled = wp.array(np.array([1], dtype=np.int32))

        def injected(state, timestep, terminated):
            evaluate(state, timestep, terminated)
            wp.launch(
                inject_invalid_score,
                dim=1,
                inputs=[enabled, bad_value, env.objective.score],
            )

        env.objective.evaluate = injected
        env.reset(frame_id=0)
        env.capture_step()
        env.reset(frame_id=0)
        env.transition_trace.clear()
        env.step(wp.zeros_like(env.action_input))
        assert np.isfinite(env.observation.numpy()).all()
        assert np.isfinite(env.state_0.body_q.numpy()).all()
        assert env.done.numpy()[0] == 1 and env.truncation.numpy()[0] == 0
        assert env.termination.numerical.numpy().tolist() == [1, 0]
        assert env.episode_step.numpy()[0] == 0
        assert np.isnan(env.transition_trace.arrays["objective_score"].numpy()[0])
        assert env.transition_trace.counts.numpy().tolist() == [1, 0]
        assert np.all(env.reset_policy.applied_voc_scale.numpy() == 0)
        np.testing.assert_allclose(
            env.reward.numpy()[0],
            env.frame_dt * env.objective.weights.numpy()[env.termination_index],
        )
        enabled.zero_()
        env.reset(frame_id=0)
        env.transition_trace.clear()
        env.failure_streak.zero_()
        env.maximum_failure_streak.zero_()
        data = env.solver.mjw_data
        # The masked clear must not reset a healthy world's cache or touch the
        # globally packed contact count/membership. Test the real native arrays.
        data.qacc_warmstart.fill_(7.0)
        nacon = data.nacon.numpy().copy()
        contact_worlds = data.contact.worldid.numpy().copy()
        selected = wp.array(np.array([1, 0], dtype=np.int32))
        wp.launch(
            clear_reset_warmstart,
            dim=data.qacc_warmstart.shape,
            inputs=[selected, data.qacc_warmstart],
        )
        np.testing.assert_array_equal(data.qacc_warmstart.numpy()[0], 0.0)
        np.testing.assert_array_equal(data.qacc_warmstart.numpy()[1], 7.0)
        np.testing.assert_array_equal(data.nacon.numpy(), nacon)
        np.testing.assert_array_equal(data.contact.worldid.numpy(), contact_worlds)
        env.reset(frame_id=0)
        # Exercise actual native physics poisoning, not only a NaN reward.
        wp.launch(
            inject_invalid_warmstart,
            dim=data.qacc_warmstart.shape[1],
            inputs=[bad_value, data.qacc_warmstart],
        )
        env.step(wp.zeros_like(env.action_input))
        assert env.termination.numerical.numpy().tolist() == [1, 0]
        assert env.transition_trace.flags.numpy()[0] & 7
        assert np.isfinite(env.observation.numpy()).all()
        np.testing.assert_array_equal(data.qacc_warmstart.numpy()[0], 0.0)
        for _ in range(6):
            env.step(wp.zeros_like(env.action_input))
            assert env.termination.numerical.numpy().tolist() == [0, 0]
            assert np.isfinite(env.state_0.body_q.numpy()).all()
            assert np.isfinite(env.observation.numpy()).all()
            assert np.isfinite(data.qacc_warmstart.numpy()).all()
        assert env.transition_trace.counts.numpy().tolist() == [1, 0]
        assert env.maximum_failure_streak.numpy().tolist() == [1, 0]
        print(
            json.dumps(
                dict(
                    invalid_transition_is_failure_not_reference_end=True,
                    only_existing_negative_termination_reward_used=True,
                    valid_world_kernel_values_unchanged=True,
                    actual_rl_auto_reset_returns_finite_observations=True,
                    raw_invalid_reward_preserved_before_reset=True,
                    actual_native_warmstart_nan_causes_failed_transition=True,
                    six_subsequent_captured_steps_healthy_after_reset=True,
                    masked_reset_preserves_other_world_cache_and_packed_contacts=True,
                    consecutive_failure_guard_gpu_counters_verified=True,
                    no_object_assistance_during_reset=True,
                    undefined_training_tracking_diagnostic_sentinel_m_rad=1000.0,
                    evaluation_behavior_changed=False,
                )
            )
        )


if __name__ == "__main__":
    main()
