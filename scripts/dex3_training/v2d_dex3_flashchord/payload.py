"""Nominal payload compensation applied only through capped robot wrist drives."""

import numpy as np
from scipy.spatial.transform import Rotation
from _dex3_task_grip import SmoothObjectReference


def payload_plan(times, poses, phases, mass, inertia, com, gain=1.0):
    if not np.isfinite(gain) or not 0 < gain <= 1:
        raise ValueError("Payload gain must be in (0, 1]")
    poses = np.asarray(poses).copy()
    poses[:, :3] += Rotation.from_quat(poses[:, 3:]).apply(
        np.broadcast_to(com, (len(times), 3))
    )
    reference = SmoothObjectReference(times, poses)
    weights = np.zeros(len(times))

    def smooth(x):
        x = np.clip(x, 0, 1)
        return x * x * (3 - 2 * x)

    for phase in phases:
        # Start after acquisition begins, fade out as the hand opens.
        w = smooth((times - phase["close_s"] + 0.15) / 0.3)
        w *= 1 - smooth((times - phase["release_start_s"]) / 0.2)
        weights = np.maximum(weights, w)
    wrench = np.array([reference.load_wrench(t, 1.0, mass, inertia) for t in times])
    wrench *= (gain * weights)[:, None]
    if not np.isfinite(wrench).all() or abs(wrench).max() > 60:
        raise ValueError("Nominal payload wrench exceeds the declared wrist budget")
    return wrench


def child_moment(force, com_moment, object_com, wrist_position, wrist_quaternion):
    """CPU reference for the moment shift and child-frame ball-joint convention."""
    moment = com_moment + np.cross(object_com - wrist_position, force)
    return Rotation.from_quat(wrist_quaternion).inv().apply(moment)
