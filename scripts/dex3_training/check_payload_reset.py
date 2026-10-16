"""Exercise payload reset isolation and shared wrist limits on the runtime GPU."""

import json
import numpy as np
import warp as wp
from v2d_dex3_flashchord.payload_kernels import (
    reset_payload_clock,
    payload_position_bias,
    payload_orientation_effort,
)


def check(device="cuda:0"):
    wp.init()
    with wp.ScopedDevice(device):

        def arr(x, dtype=wp.float32):
            return wp.array(x, dtype=dtype, device=device)

        clock = arr([0, 0], wp.int32)
        wp.launch(reset_payload_clock, dim=2, inputs=[arr([1, 0], wp.int32), clock])
        np.testing.assert_array_equal(clock.numpy(), [-1, 0])
        poses = arr([[0, 0, 0, 0, 0, 0, 1], [0, 0.2, 0, 0, 0, 0, 1]] * 2, wp.transform)
        reference = arr([[0, 0.2, 0, 0, 0, 0, 1]], wp.transform)
        force = arr([[0, 0, 3, 0, 0, 0]], wp.spatial_vector)
        stiffness = arr([400.0] * 12)
        nominal = arr([0.0] * 12)
        target = arr([0.0] * 12)
        effort = arr([0.5, 0, 0, 0.5, 0, 0] * 2)
        kwargs = dict(dim=(2, 1), device=device)
        for _ in range(3):
            wp.launch(
                payload_position_bias,
                inputs=[
                    clock,
                    1,
                    1,
                    6,
                    2,
                    arr([1], wp.int32),
                    reference,
                    poses,
                    arr([0, 1, 2], wp.int32),
                    force,
                    stiffness,
                    nominal,
                    target,
                ],
                **kwargs,
            )
        out = target.numpy().reshape(2, 6)
        np.testing.assert_allclose(out[0], 0)
        np.testing.assert_allclose(out[1, :3], [0, 0, 0.0075])
        # Repeated substeps must not accumulate the setpoint bias.
        wp.launch(
            payload_orientation_effort,
            inputs=[
                clock,
                1,
                1,
                6,
                2,
                arr([1], wp.int32),
                arr([0], wp.int32),
                reference,
                poses,
                arr([[0, 0, 0]] * 4, wp.vec3),
                arr([3, 4, 5], wp.int32),
                force,
                arr([1.0]),
                effort,
            ],
            **kwargs,
        )
        out = effort.numpy().reshape(2, 6)
        assert out[0, 3] == 0.5  # Reset world receives no stale wrench.
        assert out[1, 3] == 1.0  # .5 + .6 is clipped using one shared budget.
        return dict(
            schema="v2d.payload-reset-control.v1",
            device=device,
            selective_reset_verified=True,
            inactive_wrench_zero=True,
            substep_position_bias_does_not_accumulate=True,
            shared_orientation_effort_cap_verified=True,
        )


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
