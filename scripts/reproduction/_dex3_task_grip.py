"""Explicit inferred robot grasp schedule; no simulator state writes."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp


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


class SmoothObjectReference:
    """Inferred robot reference through frozen pose knots; no observation edits.

    Translation is C2 and rotation has continuous angular rate/acceleration in
    the source interval. Outside it, the last/first pose is held for settling.
    The endpoint hold is not a claim of smooth angular acceleration there.
    """

    def __init__(self, times, poses):
        self.times = np.asarray(times, float).copy()
        self.poses = np.asarray(poses, float).copy()
        if (
            self.times.ndim != 1
            or len(self.times) < 3
            or self.poses.shape != (len(self.times), 7)
            or not np.isfinite(np.r_[self.times, self.poses.ravel()]).all()
            or np.any(np.diff(self.times) <= 0)
            or not np.allclose(np.linalg.norm(self.poses[:, 3:], axis=1), 1, atol=1e-6)
        ):
            raise ValueError("Smooth reference needs aligned finite pose knots")
        self.position = CubicSpline(
            self.times, self.poses[:, :3], axis=0, bc_type="clamped"
        )
        self.rotation = RotationSpline(
            self.times, Rotation.from_quat(self.poses[:, 3:])
        )

    def pose(self, time_s):
        t = float(np.clip(time_s, self.times[0], self.times[-1]))
        return np.r_[self.position(t), self.rotation(t).as_quat()]

    def load_wrench(self, time_s, speed, mass_kg=0.3, inertia_body=None):
        if not np.isfinite([time_s, speed, mass_kg]).all() or not 0 < speed <= 1:
            raise ValueError("Invalid physical reference time or speed")
        inertia = (
            np.eye(3) * 0.001 if inertia_body is None else np.asarray(inertia_body)
        )
        if (
            not 0 < mass_kg <= 1
            or inertia.shape != (3, 3)
            or not np.isfinite(inertia).all()
            or not np.allclose(inertia, inertia.T)
            or np.linalg.eigvalsh(inertia).min() <= 0
        ):
            raise ValueError("Invalid nominal payload mass/inertia")
        acceleration, omega, alpha = np.zeros(3), np.zeros(3), np.zeros(3)
        if self.times[0] < time_s < self.times[-1]:
            acceleration = self.position(time_s, 2) * speed**2
            omega = self.rotation(time_s, 1) * speed
            alpha = self.rotation(time_s, 2) * speed**2
        rotation = self.rotation(float(np.clip(time_s, *self.times[[0, -1]])))
        # RotationSpline returns body-axis angular derivatives.  Euler's rigid
        # body equation includes the gyroscopic moment before rotating to world.
        moment = rotation.apply(inertia @ alpha + np.cross(omega, inertia @ omega))
        return np.r_[mass_kg * (acceleration + [0, 0, 9.81]), moment]

    def deviation_audit(self):
        grid = np.unique(
            np.r_[
                self.times,
                np.linspace(
                    self.times[0], self.times[-1], 20 * (len(self.times) - 1) + 1
                ),
            ]
        )
        linear = np.column_stack(
            [np.interp(grid, self.times, self.poses[:, j]) for j in range(3)]
        )
        baseline = Slerp(self.times, Rotation.from_quat(self.poses[:, 3:]))(grid)
        return dict(
            samples=len(grid),
            maximum_position_deviation_from_piecewise_m=float(
                np.linalg.norm(self.position(grid) - linear, axis=1).max()
            ),
            maximum_rotation_deviation_from_piecewise_deg=float(
                np.rad2deg((self.rotation(grid) * baseline.inv()).magnitude()).max()
            ),
            knots_preserved=True,
            original_source_arrays_changed=False,
            continuous_collision_clearance_certified=False,
        )


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
        self.aborted_phases = {}
        self.object_references = {
            int(oi): SmoothObjectReference(spec["source_times_s"], poses)
            for oi, poses in spec.get("smooth_robot_object_reference", {}).items()
        }

    def command(
        self, side, time_s, object_pose, measured_object_pose=None, *, commit=True
    ):
        phases = self.spec["phases"][side]
        if not np.isfinite(time_s):
            raise ValueError("Nonfinite source time")
        object_pose = np.asarray(object_pose, float).copy()
        scene = self.spec.get("scene_projection")
        oi = phases[0].get("object", 0)
        if oi in self.object_references:
            object_pose = self.object_references[oi].pose(time_s)
        elif scene is not None:
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
            rejection = None
            proposed_abort = None
            if (
                self.spec.get("experimental_bounded_reacquisition", False)
                and measured_object_pose is not None
                and feedback > 0
                and not parked
            ):
                measured = np.asarray(measured_object_pose, float)
                distance = np.linalg.norm(measured[:3] - obj[:3])
                angle = (
                    Rotation.from_quat(measured[3:]) * Rotation.from_quat(obj[3:]).inv()
                ).magnitude()
                if distance > 0.06 or angle > np.deg2rad(20):
                    proposed_abort = time_s
                    if commit:
                        self.aborted_phases.setdefault(phase["sample"], time_s)
                # Fade settling feedback to zero before the rejection boundary.
                feedback *= (1 - smooth((distance - 0.03) / 0.03)) * (
                    1 - smooth((angle - np.deg2rad(10)) / np.deg2rad(10))
                )
            aborted_at = self.aborted_phases.get(phase["sample"], proposed_abort)
            abort = (
                smooth((time_s - aborted_at) / 0.25) if aborted_at is not None else 0.0
            )
            if aborted_at is not None:
                feedback = 0.0
                rejection = "Object outside bounded reacquisition window; phase aborted"
            if measured_object_pose is not None:
                obj = blend_pose(obj, measured_object_pose, feedback)
            pose = compose_pose(obj, phase["hand_in_object_xyzw"])
            descend = smooth((time_s - start) / (close - 0.25 - start))
            retreat = smooth((time_s - end - 0.25) / 0.5)
            height = 1.0 if parked else 1 - descend + retreat
            height = height + (1 - min(height, 1.0)) * abort
            pose[2] += self.spec["park_height_m"] * height
            closed = smooth((time_s - close + 0.25) / 0.125)
            opening = smooth((time_s - end) / 0.25)
            if parked:
                closed, opening = 0.0, 0.0
            closure = closed * (1 - opening) * (1 - abort)
            opened = np.asarray(phase["open_rad"])
            q = opened + closure * (np.asarray(phase["closed_rad"]) - opened)
            squeeze = (
                smooth((time_s - close + 0.125) / 0.125) * (1 - opening) * (1 - abort)
            )
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
                desired_object_wrench_stage=self.object_references[oi].load_wrench(
                    time_s, self.spec["source_motion_speed"]
                )
                if self.spec.get("live_contact_load_model") == "reference_acceleration"
                else None,
                reacquisition_failure=rejection,
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


def contact_wrench(normal_data, friction_data, wrist_com, filter_index=None):
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
        if filter_index is not None:
            if sizes.ndim != 2 or not 0 <= filter_index < sizes.shape[1]:
                raise ValueError("Invalid primary contact filter")
            sizes = sizes[:, filter_index : filter_index + 1]
            offsets = offsets[:, filter_index : filter_index + 1]
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
