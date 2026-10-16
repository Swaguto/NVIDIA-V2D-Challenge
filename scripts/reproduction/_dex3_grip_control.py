"""Small declared lift diagnostic for a dynamically controlled wrist."""

import numpy as np
from scipy.spatial.transform import Rotation


def payload_gravity_wrench(wrist_com, payload_com, mass_kg):
    if not np.isfinite(mass_kg) or not 0 < mass_kg <= 1:
        raise ValueError("Diagnostic payload mass must be in (0, 1] kg")
    offset = np.asarray(payload_com) - np.asarray(wrist_com)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Payload and wrist COMs must be finite 3D points")
    force = np.array([0.0, 0.0, mass_kg * 9.81])
    return np.r_[force, np.cross(offset, force)]


def payload_contact_hold(forces, body_paths, remaining_s, dt):
    """Require measured thumb AND index contact; bridge a 30 ms contact dropout."""
    magnitudes = np.linalg.norm(np.asarray(forces), axis=1)
    loads = [
        sum(v for p, v in zip(body_paths, magnitudes, strict=True) if f"_{digit}_" in p)
        for digit in ["thumb", "index"]
    ]
    return 0.03 if min(loads) >= 0.5 else max(0.0, remaining_s - dt)


def lift_target(initial, time_s):
    pose = np.asarray(initial, dtype=float).copy()
    u = np.clip((time_s - 0.5) / 0.5, 0.0, 1.0)
    pose[2] += 0.05 * u * u * (3 - 2 * u)
    return pose


def lift_summary(rows):
    loaded = [r for r in rows if r["step"] >= 50]
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
