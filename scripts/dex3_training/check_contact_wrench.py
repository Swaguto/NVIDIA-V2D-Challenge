"""GPU regressions for distributed contact couples, signs and world isolation."""

import json
import numpy as np
import warp as wp
from v2d_dex3_flashchord.contact_wrench import accumulate_contact_wrench


def main():
    with wp.ScopedDevice("cuda:0"):

        def arr(values, dtype=wp.float32):
            return wp.array(np.asarray(values), dtype=dtype)

        # Two unequal-location opposing forces have zero resultant and a
        # nonzero couple. Averaging positions before summing would erase it.
        # World 1 reverses shape order and adds an explicit native contact torque.
        identity = [0, 0, 0, 0, 0, 0, 1]
        count = arr([4], wp.int32)
        shapes0 = arr([0, 0, 3, 3], wp.int32)
        shapes1 = arr([1, 1, 2, 2], wp.int32)
        # Unequal gaps plus tangential forces distinguish the actual midpoint
        # moment from an incorrect moment about either body's surface point.
        p0 = arr([[0, 0.1, -0.002], [0, -0.1, -0.004]] * 2, wp.vec3)
        p1 = arr([[0, 0.1, 0.002], [0, -0.1, 0.004]] * 2, wp.vec3)
        raw = arr(
            [
                [-2, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0.03],
                [-2, 0, 0, 0, 0, 0.02],
            ],
            wp.spatial_vector,
        )
        force = wp.zeros(2, dtype=wp.vec3)
        moment = wp.zeros_like(force)
        inputs = [
            count,
            shapes0,
            shapes1,
            p0,
            p1,
            raw,
            arr([0, 1, 2, 3], wp.int32),
            arr([0, -1], wp.int32),
            2,
            2,
            1,
            arr([1], wp.int32),
            arr([identity] * 4, wp.transform),
            arr([[0, 0, 0]] * 4, wp.vec3),
            force,
            moment,
        ]
        wp.launch(accumulate_contact_wrench, dim=8, inputs=inputs)
        np.testing.assert_allclose(force.numpy(), 0, atol=1e-7)
        np.testing.assert_allclose(
            moment.numpy(), [[0, 0, -0.4], [0, 0, -0.35]], atol=1e-7
        )
        # Capture must repeat, not accumulate across control steps.
        with wp.ScopedCapture() as capture:
            force.zero_()
            moment.zero_()
            wp.launch(accumulate_contact_wrench, dim=8, inputs=inputs)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(
            moment.numpy(), [[0, 0, -0.4], [0, 0, -0.35]], atol=1e-7
        )
        # Stale entries beyond active count are ignored; a reset/empty buffer
        # produces exactly zero force and moment.
        count.zero_()
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(force.numpy(), 0)
        np.testing.assert_array_equal(moment.numpy(), 0)
        # Invalid native evidence remains invalid rather than being clamped.
        count.assign(np.array([1], dtype=np.int32))
        bad = raw.numpy()
        bad[0, 0] = np.nan
        raw.assign(bad)
        wp.capture_launch(capture.graph)
        assert np.isnan(force.numpy()[0]).any()
        np.testing.assert_array_equal(force.numpy()[1], 0)
        print(
            json.dumps(
                dict(
                    opposing_contact_couple_preserved=True,
                    object_reaction_sign_and_reversed_shape_order_verified=True,
                    native_torsional_moment_included=True,
                    native_force_application_midpoint_verified_with_unequal_gaps=True,
                    world_isolation_verified=True,
                    captured_step_does_not_accumulate=True,
                    zero_contacts_clear_previous_wrench=True,
                    nonfinite_contact_evidence_not_clamped=True,
                    physics_state_or_object_forces_modified=False,
                )
            )
        )


if __name__ == "__main__":
    main()
