"""Small declared lift diagnostic for a dynamically controlled wrist."""

import numpy as np
from scipy.spatial.transform import Rotation


def contact_solver_options(profile):
    if profile == "default":
        return {}
    if profile != "grip":
        raise ValueError("Unknown contact solver profile")
    return dict(
        enable_external_forces_every_iteration=True,
        solve_articulation_contact_last=True,
        min_velocity_iteration_count=4,
    )


def static_load_schedule(dt, acquisition):
    """Keep fixture event times and 20 fps rendering invariant under substeps."""
    if not np.isfinite(dt) or not 0.0005 <= dt <= 0.01:
        raise ValueError("Static physics timestep must be in [0.0005, 0.01]")

    def steps(seconds):
        count = round(seconds / dt)
        if not np.isclose(count * dt, seconds, atol=1e-12, rtol=0):
            raise ValueError("Static timestep must divide all fixture event times")
        return count

    return dict(
        last_step=steps(1.5 if acquisition else 1.0),
        load_start_step=steps(0.5) if acquisition else 0,
        closure_steps=steps(0.25),
        control_last_step=steps(1.0),
        averaging_steps=steps(0.2),
        render_steps=steps(0.05),
    )


def payload_gravity_wrench(wrist_com, payload_com, mass_kg):
    if not np.isfinite(mass_kg) or not 0 < mass_kg <= 1:
        raise ValueError("Diagnostic payload mass must be in (0, 1] kg")
    offset = np.asarray(payload_com) - np.asarray(wrist_com)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Payload and wrist COMs must be finite 3D points")
    force = np.array([0.0, 0.0, mass_kg * 9.81])
    return np.r_[force, np.cross(offset, force)]


def payload_contact_hold(
    forces, body_paths, remaining_s, dt, required_digits=("thumb", "index")
):
    """Require measured thumb AND index contact; bridge a 30 ms contact dropout."""
    magnitudes = np.linalg.norm(np.asarray(forces), axis=1)
    loads = [
        sum(v for p, v in zip(body_paths, magnitudes, strict=True) if f"_{digit}_" in p)
        for digit in required_digits
    ]
    return 0.03 if min(loads) >= 0.5 else max(0.0, remaining_s - dt)


def lift_target(initial, time_s):
    pose = np.asarray(initial, dtype=float).copy()
    u = np.clip((time_s - 0.5) / 0.5, 0.0, 1.0)
    pose[2] += 0.05 * u * u * (3 - 2 * u)
    return pose


def lift_summary(rows):
    loaded = [r for r in rows if r.get("time_s", r["step"] * 0.01) >= 0.5 - 1e-12]
    obj = np.array([r["object_root_state_stage_xyzw"][:3] for r in loaded])
    hand = np.array([r["actual_wrist_pose_stage_xyzw"] for r in loaded])
    target = np.array([r["target_wrist_pose_stage_xyzw"] for r in loaded])
    relative = Rotation.from_quat(hand[:, 3:7]).inv().apply(obj - hand[:, :3])
    drift = np.linalg.norm(relative - relative[0], axis=1)
    error = np.linalg.norm(hand[:, :3] - target[:, :3], axis=1)
    rotation = (
        Rotation.from_quat(hand[:, 3:7]) * Rotation.from_quat(target[:, 3:7]).inv()
    ).magnitude()
    height = float(obj[-1, 2] - obj[0, 2])
    return dict(
        object_lift_m=height,
        maximum_object_drift_relative_to_wrist_m=float(drift.max()),
        maximum_wrist_position_error_m=float(error.max()),
        maximum_wrist_rotation_error_deg=float(np.rad2deg(rotation.max())),
        lifted_and_retained_diagnostic_only=bool(
            height >= 0.04
            and drift.max() <= 0.01
            and error.max() <= 0.02
            and rotation.max() <= np.deg2rad(5)
        ),
    )
