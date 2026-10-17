"""GPU check: retain a failed transition after inputs reset, without modifying it."""

from types import SimpleNamespace as NS
import json

import numpy as np
import warp as wp

from v2d_dex3_flashchord.transition_trace import TransitionTrace


def check():
    with wp.ScopedDevice("cuda:0"):

        def array(values, dtype=wp.float32):
            return wp.array(np.asarray(values), dtype=dtype)

        poses = np.tile([0, 0, 0, 0, 0, 0, 1.0], (4, 1))
        env = NS(
            world_count=2,
            device=wp.get_device(),
            state_0=NS(
                body_q=array(poses, wp.transform),
                body_qd=wp.zeros(4, dtype=wp.spatial_vector),
                joint_q=wp.zeros(6),
                joint_qd=wp.zeros(4),
            ),
            action=NS(buffers=NS(raw_action=wp.zeros(4))),
            objective=NS(terms=wp.zeros(6), score=wp.zeros(2)),
            contact=NS(contact_force_w=wp.zeros(4, dtype=wp.vec3)),
            termination=NS(packed_diagnostics=wp.zeros(12)),
            timestep=array([20, 30], wp.int32),
            episode_step=array([5, 8], wp.int32),
        )
        trace = TransitionTrace(env)
        trace.record()
        assert not trace.counts.numpy().any()
        broken = poses.copy()
        broken[2, 0] = np.nan
        env.state_0.body_q.assign(broken)
        trace.record()
        assert trace.flags.numpy().tolist() == [0, 1]
        assert np.array_equal(env.state_0.body_q.numpy(), broken, equal_nan=True)
        env.state_0.body_q.assign(poses)
        env.timestep.assign(np.array([21, 31], dtype=np.int32))
        trace.record()
        assert trace.first_frame.numpy().tolist() == [0, 30]
        assert np.isnan(trace.arrays["body_q"].numpy()[2, 0])
        assert trace.counts.numpy().tolist() == [0, 1]
        trace.clear()
        with wp.ScopedCapture() as capture:
            trace.record()
        objective = np.zeros(6)
        objective[0] = np.nan
        contact = np.zeros((4, 3))
        contact[0, 0] = np.inf
        diagnostics = np.zeros(12)
        diagnostics[0] = np.nan
        env.objective.terms.assign(objective)
        env.contact.contact_force_w.assign(contact)
        env.termination.packed_diagnostics.assign(diagnostics)
        wp.capture_launch(capture.graph)
        assert trace.flags.numpy().tolist() == [16 + 32 + 64, 0]
        assert trace.counts.numpy().tolist() == [1, 0]
        assert np.isnan(trace.arrays["objective_terms"].numpy()[0])
        assert np.isinf(trace.arrays["contact_force"].numpy()[0, 0])
        print(
            json.dumps(
                dict(
                    read_only_input_preservation_verified=True,
                    first_nonfinite_snapshot_survives_input_reset=True,
                    force_objective_and_tracking_categories_verified=True,
                    world_isolation_and_cuda_capture_verified=True,
                    nonfinite_values_repaired=False,
                )
            )
        )


if __name__ == "__main__":
    check()
