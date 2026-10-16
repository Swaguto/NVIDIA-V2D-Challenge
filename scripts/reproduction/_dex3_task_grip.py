"""Explicit inferred robot grasp schedule; no simulator state writes."""

import numpy as np
from scipy.spatial.transform import Rotation


def smooth(value):
    u = np.clip(value, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def blend_pose(before, after, weight):
    a, b = np.asarray(before, float), np.asarray(after, float)
    ra, rb = Rotation.from_quat(a[3:]), Rotation.from_quat(b[3:])
    return np.r_[
        a[:3] * (1 - weight) + b[:3] * weight,
        (Rotation.from_rotvec(weight * (rb * ra.inv()).as_rotvec()) * ra).as_quat(),
    ]


def relative_pose(hand, obj):
    hand, obj = np.asarray(hand), np.asarray(obj)
    rotation = Rotation.from_quat(obj[3:]).inv()
    return np.r_[
        rotation.apply(hand[:3] - obj[:3]),
        (rotation * Rotation.from_quat(hand[3:])).as_quat(),
    ]


def compose_pose(obj, relative):
    obj, relative = np.asarray(obj), np.asarray(relative)
    rotation = Rotation.from_quat(obj[3:])
    return np.r_[
        rotation.apply(relative[:3]) + obj[:3],
        (rotation * Rotation.from_quat(relative[3:])).as_quat(),
    ]


class TaskGripReference:
    """Robot commands use the original source clock and object reference.

    During approach only, an explicitly supplied measured object pose accounts
    for initial settling. It blends back to the source over 0.5 source seconds
    after closure. This is a diagnostic feedback contract, not a trained actor.
    """

    def __init__(self, spec):
        if spec["schema"] != "v2d.dex3-task-grips.v1" or spec["physical_task_accepted"]:
            raise ValueError("Need an explicitly unaccepted task grasp reference")
        self.spec = spec

    def command(self, side, time_s, object_pose, measured_object_pose=None):
        phases = self.spec["phases"][side]
        if not np.isfinite(time_s):
            raise ValueError("Nonfinite source time")
        object_pose = np.asarray(object_pose, float).copy()
        scene = self.spec.get("scene_projection")
        if scene is not None:
            oi = phases[0]["object"]
            object_pose[2] += np.interp(
                time_s,
                scene["source_times_s"],
                np.asarray(scene["upward_corrections_m"])[:, oi],
            )

        def at(phase, parked=False):
            start, close, end = [
                phase[k] for k in ("approach_start_s", "close_s", "release_start_s")
            ]
            obj = np.asarray(object_pose, float)
            feedback = smooth((time_s - start) / 0.1) * (
                1 - smooth((time_s - close) / 0.5)
            )
            if measured_object_pose is not None:
                obj = blend_pose(obj, measured_object_pose, feedback)
            pose = compose_pose(obj, phase["hand_in_object_xyzw"])
            descend = smooth((time_s - start) / (close - 0.25 - start))
            retreat = smooth((time_s - end - 0.25) / 0.5)
            height = 1.0 if parked else 1 - descend + retreat
            pose[2] += self.spec["park_height_m"] * height
            closed = smooth((time_s - close + 0.25) / 0.125)
            opening = smooth((time_s - end) / 0.25)
            if parked:
                closed, opening = 0.0, 0.0
            closure = closed * (1 - opening)
            opened = np.asarray(phase["open_rad"])
            q = opened + closure * (np.asarray(phase["closed_rad"]) - opened)
            squeeze = smooth((time_s - close + 0.125) / 0.125) * (1 - opening)
            if parked:
                squeeze = 0.0
            times = np.asarray(phase["effort_times_s"])
            efforts = np.asarray(phase["effort_Nm"])
            effort = np.array(
                [np.interp(time_s, times, efforts[:, j]) for j in range(7)]
            )
            return dict(
                pose=pose,
                fingers=q,
                effort=effort * squeeze,
                squeeze_scale=float(squeeze),
                phase=phase["sample"],
                required_digits=phase["required_digits"],
                pad_normal_jacobians=phase.get("pad_normal_jacobians", []),
                joint_limits_rad=phase.get("joint_limits_rad"),
                object_feedback_weight=float(feedback)
                if measured_object_pose is not None
                else 0.0,
            )

        for phase in phases:
            if phase["approach_start_s"] <= time_s <= phase["release_start_s"] + 0.75:
                return at(phase)
        if time_s < phases[0]["approach_start_s"]:
            return at(phases[0], True)
        if time_s > phases[-1]["release_start_s"] + 0.75:
            return at(phases[-1], True)
        for before, after in zip(phases, phases[1:]):
            begin, end = before["release_start_s"] + 0.75, after["approach_start_s"]
            if begin < time_s < end:
                a, b = at(before, True), at(after, True)
                u = smooth((time_s - begin) / (end - begin))
                a["pose"] = blend_pose(a["pose"], b["pose"], u)
                a["fingers"] = (1 - u) * a["fingers"] + u * b["fingers"]
                return a
        raise ValueError("Overlapping or incomplete grasp schedule")


def contact_wrench(normal_data, friction_data, wrist_com):
    """Sum forces on sensor bodies and moments at the wrist COM.

    Buffers follow the PhysX RigidContactView contract, including per-pair counts
    and offsets. Unused capacity must never contribute to the result.
    """
    magnitudes, points, normals, _separation, counts, starts = [
        np.asarray(x) for x in normal_data
    ]
    friction, friction_points, fcounts, fstarts = [np.asarray(x) for x in friction_data]
    com = np.asarray(wrist_com, float)
    if com.shape != (3,) or not np.isfinite(com).all():
        raise ValueError("Invalid wrist COM")
    result = np.zeros(6)
    for forces, positions, sizes, offsets in (
        (magnitudes.reshape(-1, 1) * normals, points, counts, starts),
        (friction, friction_points, fcounts, fstarts),
    ):
        if sizes.shape != offsets.shape or np.any(sizes < 0):
            raise ValueError("Invalid contact count layout")
        for count, offset in zip(sizes.flat, offsets.flat, strict=True):
            count, offset = int(count), int(offset)
            if not count:
                continue
            if offset < 0 or offset + count > len(forces):
                raise ValueError("Contact buffer overflow")
            f, p = forces[offset : offset + count], positions[offset : offset + count]
            if not np.isfinite(f).all() or not np.isfinite(p).all():
                raise ValueError("Nonfinite active contact data")
            result[:3] += f.sum(0)
            result[3:] += np.cross(p - com, f).sum(0)
    return result


def filtered_reaction_compensation(measured, previous, dt, enabled):
    """Bound and filter the opposite measured hand/object reaction at 20 ms."""
    measured, previous = np.asarray(measured), np.asarray(previous)
    if (
        measured.shape != (6,)
        or previous.shape != (6,)
        or not np.isfinite(np.r_[measured, previous, dt]).all()
        or dt <= 0
    ):
        raise ValueError("Invalid measured reaction state")
    desired = -measured if enabled else np.zeros(6)
    desired = np.clip(
        desired, -np.array([10] * 3 + [2] * 3), np.array([10] * 3 + [2] * 3)
    )
    return previous + (1 - np.exp(-dt / 0.02)) * (desired - previous)


def seat_contact_pads(
    base_q, pad_jacobians, measured_normals_N, integral_m, limits, dt, enabled
):
    """Close a missing pad with bounded normal extension, preserving motor caps.

    Nominal forces alone do not guarantee actual contact. A small unilateral PI
    adjustment seeks 4 N on each distal pad; above 6 N the integral unwinds.
    It adds at most 4 mm predicted normal extension / 0.08 rad per joint and
    never opens past the original target or exceeds a declared URDF limit.
    """
    q, jac, loads, state, bounds = [
        np.asarray(x, float)
        for x in (base_q, pad_jacobians, measured_normals_N, integral_m, limits)
    ]
    if (
        q.shape != (7,)
        or jac.ndim != 2
        or jac.shape[1] != 7
        or loads.shape != (len(jac),)
        or state.shape != loads.shape
        or bounds.shape != (7, 2)
        or not np.isfinite(
            np.r_[q, jac.ravel(), loads, state, bounds.ravel(), dt]
        ).all()
        or dt <= 0
        or np.any(loads < 0)
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("Invalid tactile seating inputs")
    if not enabled:
        return q.copy(), np.zeros_like(state), np.zeros(7)
    error = 4.0 - loads
    change = np.maximum(error, 0) - np.maximum(loads - 6.0, 0)
    state = np.clip(state + 0.002 * change * dt, 0, 0.003)
    extension = np.clip(state + 0.00025 * np.maximum(error, 0), 0, 0.004)
    norm2 = np.einsum("ij,ij->i", jac, jac)
    if np.any(norm2 < 1e-8):
        raise ValueError("Degenerate pad seating Jacobian")
    delta = np.sum(jac * (extension / norm2)[:, None], axis=0)
    delta = np.clip(delta, -0.08, 0.08)
    adjusted = np.clip(q + delta, bounds[:, 0] + 1e-5, bounds[:, 1] - 1e-5)
    return adjusted, state, adjusted - q
