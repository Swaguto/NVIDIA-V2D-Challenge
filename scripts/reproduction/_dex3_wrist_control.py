"""Experimental timestep-aware wrist control, independent of simulator imports.

All spatial vectors use world axes at the root body's COM. All quaternions are
xyzw (the pinned Lab 3 runtime convention). The root 6x6 mass block assumes
locked finger joints for the implicit prediction; it is not full inverse dynamics.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def implicit_root_wrench(
    *,
    link_pose_xyzw,
    target_link_pose_xyzw,
    root_com_offset_b,
    root_com_velocity_w,
    root_mass_matrix_w,
    gravity_compensation_w,
    dt,
    linear_stiffness=50.0,
    linear_damping=10.0,
    angular_stiffness=12.0,
    angular_damping=0.1,
    max_force=60.0,
    max_torque=60.0,
):
    """Return bounded world-frame force/torque using a backward-Euler PD prediction.

    For constant mass M and local rotation linearization, solve
    (M + dt D + dt² K) a = K(e - dt v) - D v, then apply M a + gravity.
    The desired COM includes the rotated local root-body COM offset, so position
    error, velocity, mass matrix and applied wrench share the same point.
    Inputs are copied; simulator state is never mutated.
    """
    pose = np.asarray(link_pose_xyzw, dtype=float)
    target = np.asarray(target_link_pose_xyzw, dtype=float)
    offset = np.asarray(root_com_offset_b, dtype=float)
    velocity = np.asarray(root_com_velocity_w, dtype=float)
    mass = np.asarray(root_mass_matrix_w, dtype=float)
    gravity = np.asarray(gravity_compensation_w, dtype=float)
    for value, shape in (
        (pose, (7,)),
        (target, (7,)),
        (offset, (3,)),
        (velocity, (6,)),
        (mass, (6, 6)),
        (gravity, (6,)),
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(
                f"Nonfinite or incorrectly shaped wrist input: expected {shape}"
            )
    parameters = np.array(
        [
            dt,
            linear_stiffness,
            linear_damping,
            angular_stiffness,
            angular_damping,
            max_force,
            max_torque,
        ]
    )
    if not np.isfinite(parameters).all() or np.any(parameters <= 0):
        raise ValueError("Wrist timestep, gains and caps must be finite and positive")
    if not np.allclose(mass, mass.T, rtol=1e-5, atol=1e-7):
        raise ValueError("Root mass matrix is not symmetric")
    mass = (mass + mass.T) / 2
    if np.linalg.eigvalsh(mass).min() <= 0:
        raise ValueError("Root mass matrix is not positive definite")
    if any(abs(np.linalg.norm(q) - 1) > 1e-4 for q in (pose[3:], target[3:])):
        raise ValueError("Wrist quaternions must be unit xyzw")
    current_rotation = Rotation.from_quat(pose[3:])
    target_rotation = Rotation.from_quat(target[3:])
    position_error = (
        target[:3]
        + target_rotation.apply(offset)
        - (pose[:3] + current_rotation.apply(offset))
    )
    rotation_error = (target_rotation * current_rotation.inv()).as_rotvec()
    error = np.r_[position_error, rotation_error]
    stiffness = np.array([linear_stiffness] * 3 + [angular_stiffness] * 3)
    damping = np.array([linear_damping] * 3 + [angular_damping] * 3)
    acceleration = np.linalg.solve(
        mass + np.diag(dt * damping + dt**2 * stiffness),
        stiffness * (error - dt * velocity) - damping * velocity,
    )
    requested = mass @ acceleration + gravity
    caps = np.array([max_force] * 3 + [max_torque] * 3)
    applied = np.clip(requested, -caps, caps)
    return applied, {
        "position_error_at_root_com_m": position_error.tolist(),
        "rotation_error_world_rad": rotation_error.tolist(),
        "root_mass_eigenvalues": np.linalg.eigvalsh(mass).tolist(),
        "requested_wrench_world": requested.tolist(),
        "saturated_components": (np.abs(requested) > caps).tolist(),
    }


def finger_position_target(target, position, velocity, stiffness, damping, effort):
    """Keep the original per-joint effort clamp; reject zero stiffness and NaNs."""
    target, position, velocity, stiffness, damping, effort = (
        np.asarray(x, dtype=float)
        for x in (target, position, velocity, stiffness, damping, effort)
    )
    if any(
        x.shape != target.shape or not np.isfinite(x).all()
        for x in (target, position, velocity, stiffness, damping, effort)
    ):
        raise ValueError("Finger controller arrays must be finite and aligned")
    if np.any(stiffness <= 0) or np.any(damping < 0) or np.any(effort <= 0):
        raise ValueError("Invalid finger drive constants")
    return np.clip(
        target,
        position + (-effort + damping * velocity) / stiffness,
        position + (effort + damping * velocity) / stiffness,
    )
