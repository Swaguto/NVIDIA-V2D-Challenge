"""GPU contracts for physical grip rewards and force-aware observations."""

import json
import numpy as np
import warp as wp
from v2d_dex3_flashchord.grip_training import evaluate_grip, append_force_observation


def check():
    wp.init()
    with wp.ScopedDevice("cuda:0"):

        def arr(values, dtype=wp.float32):
            return wp.array(values, dtype=dtype, device="cuda:0")

        worlds, bodies, slots = 4, 2, 3
        pose = np.tile([0, 0, 0, 0, 0, 0, 1.0], (worlds * bodies, 1))
        pose[1::2, 2] = 0.05
        pose[5, 2] = 0.0
        velocity = np.zeros((worlds * bodies, 6))
        velocity[7, 5] = 2.0
        force = np.tile([0, 0, 0.981], (worlds * slots, 1))
        force[3:6] = 0
        terms, score = wp.zeros(worlds * 6), wp.zeros(worlds)
        inputs = [
            arr([0] * worlds, wp.int32),
            1,
            bodies,
            1,
            slots,
            arr([0] * worlds),
            1,
            arr([0] * worlds),
            0.05,
            arr(pose, wp.transform),
            arr(velocity, wp.spatial_vector),
            arr([[0, 0, 0]] * (worlds * bodies), wp.vec3),
            arr([0], wp.int32),
            arr([1], wp.int32),
            arr([0], wp.int32),
            arr([slots], wp.int32),
            arr([0, 1, 2], wp.int32),
            arr([1]),
            arr([1]),
            arr([0]),
            1,
            arr([[0, 0, 0.05]], wp.vec3),
            arr([[0, 0, 2.943, 0, 0, 0]], wp.spatial_vector),
            arr([[0, 0, 0]] * (worlds * slots), wp.vec3),
            arr(force, wp.vec3),
            arr([0, 1, 1, 1, 1, -0.02]),
            terms,
            score,
        ]
        wp.launch(evaluate_grip, dim=worlds, inputs=inputs)
        values = terms.numpy().reshape(worlds, 6)
        np.testing.assert_allclose(values[0, 1:], [1, 1, 1, 1, 0], atol=1e-6)
        np.testing.assert_allclose(values[1], 0, atol=1e-6)
        assert values[2, 4] < 0.001 and values[2, 1] == 1
        assert values[3, 2] < 0.02 and values[3, 4] == 1
        np.testing.assert_allclose(score.numpy()[0], 0.2, atol=1e-6)
        before = values.copy()
        inputs[-4] = arr(np.zeros_like(force), wp.vec3)
        wp.launch(evaluate_grip, dim=worlds, inputs=inputs)
        assert not terms.numpy().reshape(worlds, 6)[:, 1:].any()
        assert before[0, 1] == 1  # independent snapshot, no shared world state

        q = np.tile([0, 0, 0, 0, 0, 0, 1.0], (4, 1))
        q[1, 3:] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
        output = wp.zeros(2 * 6)
        wp.launch(
            append_force_observation,
            dim=2,
            inputs=[
                arr([11, 12, 21, 22]),
                2,
                6,
                1,
                2,
                3,
                arr([0, 2], wp.int32),
                arr([1, 1], wp.int32),
                arr([0, 1], wp.int32),
                arr([1], wp.int32),
                arr([0], wp.int32),
                arr(q, wp.transform),
                arr([[0, 20, 0], [40, 0, 0]], wp.vec3),
                20.0,
                100.0,
                output,
            ],
        )
        np.testing.assert_allclose(
            output.numpy().reshape(2, 6),
            [[11, 12, 1, 0, 0, 0], [21, 22, 2, 0, 0, 1]],
            atol=1e-6,
        )
        reset_output = wp.zeros(2 * 6)
        wp.launch(
            append_force_observation,
            dim=2,
            inputs=[
                arr([11, 12, 21, 22]),
                2,
                6,
                1,
                2,
                3,
                arr([0, 2], wp.int32),
                arr([0, 1], wp.int32),
                arr([0, 1], wp.int32),
                arr([1], wp.int32),
                arr([0], wp.int32),
                arr(q, wp.transform),
                arr([[0, 20, 0], [40, 0, 0]], wp.vec3),
                20.0,
                100.0,
                reset_output,
            ],
        )
        np.testing.assert_allclose(
            reset_output.numpy().reshape(2, 6),
            [[11, 0, 0, 0, 0, 0], [21, 22, 2, 0, 0, 1]],
            atol=1e-6,
        )
        bounded = wp.zeros(6)
        wp.launch(
            append_force_observation,
            dim=1,
            inputs=[
                arr([1e20, float("nan")]),
                2,
                6,
                1,
                2,
                3,
                arr([0], wp.int32),
                arr([1], wp.int32),
                arr([0, 0], wp.int32),
                arr([1], wp.int32),
                arr([0], wp.int32),
                arr(q[:2], wp.transform),
                arr([[0, 0, 0]], wp.vec3),
                20.0,
                100.0,
                bounded,
            ],
        )
        assert bounded.numpy()[0] == 100 and np.isnan(bounded.numpy()[1])
        return dict(
            schema="v2d.grip-training-gpu-contract.v1",
            balanced_loaded_grip_reward_verified=True,
            missed_grasp_has_no_contact_or_carry_reward=True,
            stationary_missed_lift_not_rewarded=True,
            angular_slip_reduces_stability=True,
            reward_has_no_accumulated_state=True,
            force_observation_object_frame_and_world_isolation_verified=True,
            reset_world_has_no_stale_contact_observations=True,
            finite_observation_outliers_bounded_and_nan_not_hidden=True,
            physical_task_accepted=False,
        )


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
