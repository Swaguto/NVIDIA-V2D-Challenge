"""Explicit inferred robot grasp schedule; no simulator state writes."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear
from scipy.sparse import diags, eye, vstack
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


def predict_measured_pose(pose, link_twist_world, dt):
    """One-step actor-frame prediction; never write the predicted object pose."""
    pose, twist = np.asarray(pose, float), np.asarray(link_twist_world, float)
    if (
        pose.shape != (7,)
        or twist.shape != (6,)
        or not np.isfinite(np.r_[pose, twist, dt]).all()
        or not 0 <= dt <= 0.02
        or abs(np.linalg.norm(pose[3:]) - 1) > 1e-5
    ):
        raise ValueError("Invalid measured actor pose/twist prediction")
    return np.r_[
        pose[:3] + dt * twist[:3],
        (Rotation.from_rotvec(dt * twist[3:]) * Rotation.from_quat(pose[3:])).as_quat(),
    ]


def registered_carry_gain_scale(feedback_weight, registered_carry):
    """Retain world gains while adding measured-grasp impedance.

    In the rigid-grasp limit e_blend=(1-w)*e_world, also for velocity error.
    Dividing gains by (1-w) prevents unintended tenfold softening. The implicit
    discrete prediction remains approximate for coupled contact.
    """
    if not np.isfinite(feedback_weight) or not 0 <= feedback_weight <= 1:
        raise ValueError("Invalid measured feedback gain weight")
    if not registered_carry:
        return 1.0
    if feedback_weight > 0.900000001:
        raise ValueError("Registered carry feedback exceeds its declared bound")
    return 1 / max(0.1, 1 - feedback_weight)


def hand_control_settings(spec, side):
    """Resolve explicitly scoped hand settings; default historical globals stay valid."""
    defaults = dict(
        experimental_register_settled_grasp=False,
        experimental_carry_compliance=False,
        carry_object_feedback_weight=0.9,
        wrist_payload_compensation="measured_primary_reaction",
    )
    overrides = spec.get("hand_control_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - {"left", "right"}:
        raise ValueError("Invalid hand controller override sides")
    for settings in overrides.values():
        if not isinstance(settings, dict) or set(settings) - set(defaults):
            raise ValueError("Unknown scoped hand controller setting")
    result = {key: spec.get(key, value) for key, value in defaults.items()}
    result.update(overrides.get(side, {}))
    if (
        not isinstance(result["experimental_register_settled_grasp"], bool)
        or not isinstance(result["experimental_carry_compliance"], bool)
        or (
            result["experimental_carry_compliance"]
            and not result["experimental_register_settled_grasp"]
        )
        or not np.isfinite(result["carry_object_feedback_weight"])
        or not 0 <= result["carry_object_feedback_weight"] <= 0.9
        or result["wrist_payload_compensation"]
        not in {"nominal_gravity", "measured_reaction", "measured_primary_reaction"}
    ):
        raise ValueError("Invalid scoped hand controller settings")
    return result


def aborted_hand_hold(anchor_pose, initial_fingers, open_fingers, elapsed_source_s):
    """Hold the measured wrist and open smoothly after unavailable reacquisition."""
    return np.asarray(anchor_pose, float).copy(), (
        np.asarray(initial_fingers, float)
        + smooth(elapsed_source_s / 0.25)
        * (np.asarray(open_fingers, float) - np.asarray(initial_fingers, float))
    )


def denoise_robot_reference(times, poses, horizon_s=0.1):
    """Bounded acceleration regularization of robot targets, never observations.

    Every translation knot stays within 3 mm and every rotation knot within two
    degrees of its input. Quaternion signs are unwrapped before chordal fitting.
    The caller separately audits the interpolated path against the input path.
    """
    times, poses = np.asarray(times, float), np.asarray(poses, float)
    if (
        times.ndim != 1
        or len(times) < 3
        or poses.shape != (len(times), 7)
        or not np.isfinite(np.r_[times, poses.ravel(), horizon_s]).all()
        or np.any(np.diff(times) <= 0)
        or not 0.05 <= horizon_s <= 0.2
        or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-6)
    ):
        raise ValueError("Invalid bounded robot reference denoising inputs")
    n = len(times)
    before, after = np.diff(times)[:-1], np.diff(times)[1:]
    second = diags(
        [
            2 / (before * (before + after)),
            -2 / (before * after),
            2 / (after * (before + after)),
        ],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csr",
    )
    penalty = horizon_s**2 * second
    matrix = vstack([eye(n, format="csr"), penalty], format="csr")
    values = poses.copy()
    for i in range(1, n):
        if values[i, 3:] @ values[i - 1, 3:] < 0:
            values[i, 3:] *= -1
    result = values.copy()
    # The coordinate boxes imply Euclidean bounds. For a unit quaternion q,
    # normalizing q+e changes its rotation by <= 2 asin(||e||).
    bounds = [0.003 / np.sqrt(3)] * 3 + [np.sin(np.deg2rad(2) / 2) / 2] * 4
    for j, bound in enumerate(bounds):
        offset = lsq_linear(
            matrix,
            np.r_[np.zeros(n), -penalty @ values[:, j]],
            bounds=(-bound, bound),
            tol=1e-10,
            lsmr_tol=1e-11,
        )
        if not offset.success:
            raise ValueError("Bounded robot reference optimization failed")
        result[:, j] += offset.x
    result[:, 3:] /= np.linalg.norm(result[:, 3:], axis=1)[:, None]
    return result


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

    By default during approach only, a supplied measured object pose accounts
    for initial settling. It blends back to the source over 0.5 source seconds
    after closure. Optional registered carry compliance retains bounded pose
    feedback during load. This diagnostic controller is not a trained actor.
    """

    def __init__(self, spec):
        if spec["schema"] != "v2d.dex3-task-grips.v1" or spec["physical_task_accepted"]:
            raise ValueError("Need an explicitly unaccepted task grasp reference")
        self.spec = spec
        heights = spec.get("park_height_by_hand_m", {})
        if (
            not isinstance(heights, dict)
            or set(heights) - {"left", "right"}
            or any(
                not np.isfinite(v) or not 0.12 <= v <= 0.25 for v in heights.values()
            )
        ):
            raise ValueError("Per-hand parking heights must be in [0.12, 0.25] m")
        self.carry_feedback_weight = float(
            spec.get("carry_object_feedback_weight", 0.9)
        )
        if (
            not np.isfinite(self.carry_feedback_weight)
            or not 0 <= self.carry_feedback_weight <= 0.9
        ):
            raise ValueError("Carry object feedback must be in [0, 0.9]")
        self.aborted_phases = {}
        self.grasp_registrations = {}
        self.object_references = {
            int(oi): SmoothObjectReference(spec["source_times_s"], poses)
            for oi, poses in spec.get("smooth_robot_object_reference", {}).items()
        }
        for side in spec["phases"]:
            hand_control_settings(spec, side)

    def command(
        self, side, time_s, object_pose, measured_object_pose=None, *, commit=True
    ):
        phases = self.spec["phases"][side]
        settings = hand_control_settings(self.spec, side)
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
            obj = np.asarray(object_pose, float).copy()
            registration = self.grasp_registrations.get(phase["sample"])
            register = settings["experimental_register_settled_grasp"]
            # Capture once at closure. Future command prediction must not mutate
            # feedback state or register a phase before the physical clock does.
            if (
                register
                and registration is None
                and measured_object_pose is not None
                and close <= time_s <= close + 0.05
                and not parked
                and phase["sample"] not in self.aborted_phases
            ):
                measured = np.asarray(measured_object_pose, float)
                distance = np.linalg.norm(measured[:3] - obj[:3])
                angle = (
                    Rotation.from_quat(measured[3:]) * Rotation.from_quat(obj[3:]).inv()
                ).magnitude()
                if distance <= 0.03 and angle <= np.deg2rad(10):
                    registration = dict(
                        captured_source_time_s=float(time_s),
                        position_offset_world_m=(measured[:3] - obj[:3]).tolist(),
                        rotation_offset_body_xyzw=(
                            Rotation.from_quat(obj[3:]).inv()
                            * Rotation.from_quat(measured[3:])
                        )
                        .as_quat()
                        .tolist(),
                        mode="world_translation_and_body_rotation_bias",
                        position_offset_m=float(distance),
                        rotation_offset_deg=float(np.rad2deg(angle)),
                    )
                    if commit:
                        self.grasp_registrations[phase["sample"]] = registration
            if registration is not None:
                # The table/gravity frame is already fixed. Preserve world
                # translation and world angular increments instead of rotating
                # the whole transport path by an object's settling tilt.
                obj = np.r_[
                    obj[:3] + registration["position_offset_world_m"],
                    (
                        Rotation.from_quat(obj[3:])
                        * Rotation.from_quat(registration["rotation_offset_body_xyzw"])
                    ).as_quat(),
                ]
            feedback = smooth((time_s - start) / 0.1) * (
                1 - smooth((time_s - close) / 0.5)
            )
            carry_compliance = (
                registration is not None and settings["experimental_carry_compliance"]
            )
            if registration is not None:
                # Track the desired object motion through a compliant offset
                # from the measured grasp. This attenuates target noise instead
                # of imposing the complete pose error across millimetre pads.
                feedback = (
                    settings["carry_object_feedback_weight"]
                    * smooth((time_s - close) / 0.1)
                    * (1 - smooth((time_s - end) / 0.25))
                    if carry_compliance and not parked
                    else 0.0
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
            if measured_object_pose is not None and (
                registration is None or carry_compliance
            ):
                obj = blend_pose(obj, measured_object_pose, feedback)
            pose = compose_pose(obj, phase["hand_in_object_xyzw"])
            descend = smooth((time_s - start) / (close - 0.25 - start))
            retreat = smooth((time_s - end - 0.25) / 0.5)
            height = 1.0 if parked else 1 - descend + retreat
            height = height + (1 - min(height, 1.0)) * abort
            pose[2] += (
                self.spec.get("park_height_by_hand_m", {}).get(
                    side, self.spec["park_height_m"]
                )
                * height
            )
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
            desired_wrench = (
                self.object_references[oi].load_wrench(
                    time_s, self.spec["source_motion_speed"]
                )
                if self.spec.get("live_contact_load_model") == "reference_acceleration"
                else None
            )
            # A constant world translation and right rotation bias preserve
            # world acceleration/angular rate. With the declared isotropic
            # nominal inertia, the reference load wrench is unchanged.
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
                and (registration is None or carry_compliance)
                else 0.0,
                desired_object_wrench_stage=desired_wrench,
                settled_grasp_registration=registration,
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
