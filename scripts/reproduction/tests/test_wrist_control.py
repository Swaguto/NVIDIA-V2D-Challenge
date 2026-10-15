"""Independent mechanics checks for the experimental wrist controller."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _dex3_wrist_control import finger_position_target, implicit_root_wrench


def inputs():
    return dict(
        link_pose_xyzw=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        target_link_pose_xyzw=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        root_com_offset_b=np.zeros(3),
        root_com_velocity_w=np.zeros(6),
        root_mass_matrix_w=np.diag([0.5, 0.5, 0.5, 0.0004, 0.0005, 0.0006]),
        gravity_compensation_w=np.zeros(6),
        dt=0.01,
    )


def test_implicit_prediction_stabilizes_small_inertia_at_original_timestep():
    # A single rotational DOF with known inertia; semi-implicit Euler plant.
    # The original Kp=12/Kd=.1 at dt=.01 diverges for I=.0004.
    angles = []
    for implicit in (False, True):
        angle, velocity = 0.05, 0.0
        peak = abs(angle)
        for _ in range(200):
            if implicit:
                data = inputs()
                data["link_pose_xyzw"][3:] = Rotation.from_rotvec(
                    [angle, 0, 0]
                ).as_quat()
                data["root_com_velocity_w"][3] = velocity
                torque = implicit_root_wrench(**data)[0][3]
            else:
                torque = np.clip(-12 * angle - 0.1 * velocity, -60, 60)
            velocity += 0.01 * torque / 0.0004
            angle += 0.01 * velocity
            peak = max(peak, abs(angle))
        angles.append((abs(angle), peak))
    assert angles[0][1] > 1
    assert angles[1][0] < 1e-8
    assert angles[1][1] <= 0.05


def test_world_frame_equivariance_and_com_offset():
    data = inputs()
    data["link_pose_xyzw"][:3] = [0.1, -0.2, 0.3]
    data["target_link_pose_xyzw"][:3] = [0.2, -0.1, 0.35]
    data["target_link_pose_xyzw"][3:] = Rotation.from_euler(
        "xyz", [0.2, 0.1, -0.3]
    ).as_quat()
    data["root_com_offset_b"] = np.array([0.02, -0.03, 0.01])
    data["root_com_velocity_w"] = np.array([0.1, -0.2, 0.3, 0.2, 0.1, -0.1])
    data["gravity_compensation_w"] = np.array([0, 0, 4.905, 0.02, 0.03, 0])
    expected = implicit_root_wrench(**data)[0]
    rotation = Rotation.from_euler("xyz", [0.7, -0.6, 0.4])
    matrix = rotation.as_matrix()
    transform = np.zeros((6, 6))
    transform[:3, :3] = transform[3:, 3:] = matrix
    for name in ["link_pose_xyzw", "target_link_pose_xyzw"]:
        data[name] = np.r_[
            rotation.apply(data[name][:3]),
            (rotation * Rotation.from_quat(data[name][3:])).as_quat(),
        ]
    data["root_mass_matrix_w"] = transform @ data["root_mass_matrix_w"] @ transform.T
    data["root_com_velocity_w"] = transform @ data["root_com_velocity_w"]
    data["gravity_compensation_w"] = transform @ data["gravity_compensation_w"]
    np.testing.assert_allclose(
        implicit_root_wrench(**data)[0], transform @ expected, atol=1e-12
    )


def test_quaternion_double_cover_gravity_and_input_immutability():
    data = inputs()
    data["gravity_compensation_w"][2] = 4.905
    data["target_link_pose_xyzw"][3:] *= -1
    originals = {k: v.copy() for k, v in data.items() if isinstance(v, np.ndarray)}
    np.testing.assert_allclose(
        implicit_root_wrench(**data)[0], data["gravity_compensation_w"]
    )
    for key, value in originals.items():
        np.testing.assert_array_equal(data[key], value)


@pytest.mark.parametrize(
    "key,value",
    [
        ("dt", 0),
        ("dt", float("nan")),
        ("root_mass_matrix_w", np.zeros((6, 6))),
        ("root_com_velocity_w", np.full(6, np.nan)),
        ("target_link_pose_xyzw", np.zeros(7)),
    ],
)
def test_bad_runtime_measurements_are_rejected(key, value):
    data = inputs()
    data[key] = value
    with pytest.raises(ValueError):
        implicit_root_wrench(**data)


def test_finger_clamp_respects_effort_in_independent_pd_equation():
    q, v = np.array([0.1, -0.2]), np.array([0.3, -0.4])
    stiffness, damping, limit = np.array([50, 30]), np.array([2, 3]), np.array([1, 2])
    target = finger_position_target(np.array([5, -5]), q, v, stiffness, damping, limit)
    actual_torque = stiffness * (target - q) - damping * v
    np.testing.assert_allclose(actual_torque, [1, -2], atol=1e-12)


def test_reference_velocity_does_not_brake_perfect_constant_speed_tracking():
    data = inputs()
    data["root_com_velocity_w"][:3] = [0.3, -0.2, 0.1]
    data["root_com_velocity_w"][3:] = [0.2, -0.3, 0.4]
    data["target_link_pose_next_xyzw"] = data["target_link_pose_xyzw"].copy()
    data["target_link_pose_next_xyzw"][:3] += (
        data["dt"] * data["root_com_velocity_w"][:3]
    )
    data["target_link_pose_next_xyzw"][3:] = Rotation.from_rotvec(
        data["dt"] * data["root_com_velocity_w"][3:]
    ).as_quat()
    data["gravity_compensation_w"][2] = 4.905
    np.testing.assert_allclose(
        implicit_root_wrench(**data)[0], data["gravity_compensation_w"], atol=1e-12
    )


def test_stationary_next_reference_preserves_original_implicit_controller():
    data = inputs()
    data["target_link_pose_xyzw"][:3] = [0.01, -0.02, 0.01]
    expected = implicit_root_wrench(**data)[0]
    data["target_link_pose_next_xyzw"] = data["target_link_pose_xyzw"].copy()
    np.testing.assert_array_equal(implicit_root_wrench(**data)[0], expected)


def test_bounded_integral_rejects_constant_external_load():
    # A force-driven 0.5 kg body under a constant 3 N disturbance. Compare
    # measured plant motion, independently integrated, with/without integral.
    residuals = []
    for enabled in (False, True):
        data = inputs()
        integral = np.zeros(3)
        peak = 0.0
        for _ in range(1500):
            if enabled:
                data["integral_force_world"] = integral
            wrench, details = implicit_root_wrench(**data)
            integral = np.array(details["integral_force_world_next_N"])
            data["root_com_velocity_w"][0] += data["dt"] * (wrench[0] - 3.0) / 0.5
            data["link_pose_xyzw"][0] += data["dt"] * data["root_com_velocity_w"][0]
            peak = max(peak, abs(data["link_pose_xyzw"][0]))
            assert np.abs(integral).max() <= 10.0
            assert np.abs(wrench).max() <= 60.0
        residuals.append(abs(data["link_pose_xyzw"][0]))
        assert peak < 0.1
    assert residuals[0] > 0.05
    assert residuals[1] < 1e-5


def test_force_saturation_prevents_outward_integral_windup():
    data = inputs()
    data["target_link_pose_xyzw"][0] = 10.0
    integral = np.zeros(3)
    for _ in range(100):
        data["integral_force_world"] = integral
        wrench, details = implicit_root_wrench(**data)
        integral = np.array(details["integral_force_world_next_N"])
        assert wrench[0] == 60.0
        assert details["integral_update_frozen_axes"][0]
    np.testing.assert_array_equal(integral, np.zeros(3))
    data["target_link_pose_xyzw"][0] = 0.0
    np.testing.assert_array_equal(implicit_root_wrench(**data)[0], np.zeros(6))
