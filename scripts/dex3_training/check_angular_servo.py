"""GPU test of child-frame angular reference, caps, and reset-world isolation."""

import json
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from v2d_dex3_flashchord.angular_servo import reference_angular_effort


if __name__ == "__main__":
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        def array(values, dtype=wp.float32):
            return wp.array(np.asarray(values), dtype=dtype)

        q = np.tile(np.r_[[0, 0, 0, 1], Rotation.from_euler('z', 90, degrees=True).as_quat()], 2)
        qd = np.zeros(12)
        qd[0] = -10
        base = np.arange(12) * .01
        result = array(base)
        wp.launch(reference_angular_effort, dim=(2, 2), inputs=[
            array([0, -1], wp.int32), 2, 2, 8, 6,
            array([0, 4], wp.int32), array([0, 3], wp.int32),
            array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], wp.vec3),
            .4, array([.5, .5]), array(q), array(qd), result,
        ])
        expected = base.copy()
        for hand, velocity in enumerate([[1, 0, 0], [0, 1, 0]]):
            local = Rotation.from_quat(q[4 * hand:4 * hand + 4]).inv().apply(velocity)
            at = slice(3 * hand, 3 * hand + 3)
            expected[at] = np.clip(base[at] + .4 * (local - qd[at]), -.5, .5)
        np.testing.assert_allclose(result.numpy(), expected, atol=2e-7)
        np.testing.assert_array_equal(result.numpy()[6:], base[6:].astype(np.float32))
        print(json.dumps(dict(child_frame_reference_matches_independent_cpu=True,
                              combined_robot_effort_capped=True,
                              inactive_reset_world_unchanged=True,
                              physical_task_accepted=False)))
