# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Pink regression: the first solve must see the selected initial pose."""
from typing import Literal

import numpy as np
import pink
import pytest
from robotic_grounding.retarget import hand_kinematics
from robotic_grounding.retarget.retarget_utils import setup_dex3_kinematics
from scipy.spatial.transform import Rotation


@pytest.mark.parametrize("side", ["left", "right"])
def test_initial_pose_and_reset_refresh_fk_before_solving(
    side: Literal["left", "right"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare real cached FK/Jacobians with independent forward kinematics."""
    kin = setup_dex3_kinematics(side=side, max_iter=1)
    calls = []

    def inspect_first_solve(
        configuration: pink.Configuration, **kwargs: object
    ) -> np.ndarray:
        fresh = pink.Configuration(
            kin.robot.model, kin.robot.data, configuration.q, copy_data=True
        )
        for frame in kin.robot_frame_names.values():
            actual = configuration.get_transform_frame_to_world(frame)
            expected = fresh.get_transform_frame_to_world(frame)
            np.testing.assert_allclose(
                actual.translation, expected.translation, atol=1e-12
            )
            np.testing.assert_allclose(actual.rotation, expected.rotation, atol=1e-12)
            np.testing.assert_allclose(
                configuration.get_frame_jacobian(frame),
                fresh.get_frame_jacobian(frame),
                atol=1e-12,
            )
        calls.append(configuration.q.copy())
        return np.zeros(kin.robot.model.nv)

    monkeypatch.setattr(hand_kinematics, "solve_ik", inspect_first_solve)
    for position, angles in (
        ([0.12, -0.23, 0.45], [0.1, 0.2, -0.3]),
        ([-0.3, 0.18, 0.5], [-0.2, 0.3, 0.4]),
    ):
        initial = kin.robot.q0.copy()
        initial[:3] = position
        initial[3:7] = Rotation.from_euler("xyz", angles).as_quat()
        kin.compute(
            np.tile(initial[:3], (21, 1)),
            np.tile(
                Rotation.from_euler("xyz", angles).as_quat(scalar_first=True), (21, 1)
            ),
            qpos=initial,
        )
        np.testing.assert_array_equal(calls[-1], initial)
    kin.compute(np.zeros((21, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (21, 1)))
    np.testing.assert_array_equal(calls[-1], kin.robot.q0)
    assert len(calls) == 3
