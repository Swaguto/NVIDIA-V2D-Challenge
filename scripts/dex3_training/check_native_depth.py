"""GPU regression for signed penetration, contact ownership and world isolation."""

import json
import numpy as np
import warp as wp
from v2d_dex3_flashchord.native_depth import maximum_assigned_depth


if __name__ == "__main__":
    wp.init()
    with wp.ScopedDevice("cuda:0"):

        def arr(x, dtype):
            return wp.array(x, dtype=dtype, device="cuda:0")

        poses = [[0, 0, 0, 0, 0, 0, 1]] * 3 + [[10, 0, 0, 0, 0, 0, 1]] * 3
        depth = wp.zeros(2, dtype=wp.float32, device="cuda:0")
        wp.launch(
            maximum_assigned_depth,
            dim=4,
            inputs=[
                arr([3], wp.int32),
                arr([0, 2, 3], wp.int32),
                arr([1, 1, 4], wp.int32),
                arr([[0, 0, 0]] * 3, wp.vec3),
                arr([[0, 0, -0.003], [0, 0, -1.0], [0, 0, 0.001]], wp.vec3),
                arr([[0, 0, 1]] * 3, wp.vec3),
                arr(list(range(6)), wp.int32),
                arr(poses, wp.transform),
                3,
                1,
                arr([0, -1, -1], wp.int32),
                arr([1], wp.int32),
                depth,
            ],
        )
        np.testing.assert_allclose(depth.numpy(), [0.003, 0], atol=1e-8)
        print(
            json.dumps(
                dict(
                    schema="v2d.native-depth-kernel-check.v1",
                    signed_penetration_verified=True,
                    unassigned_contacts_ignored=True,
                    world_isolation_verified=True,
                )
            )
        )
