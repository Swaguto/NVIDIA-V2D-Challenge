"""Training signals from simulated robot contacts and inferred grasp phases.

No human contact labels or externally applied object forces are used. The
reference trajectories, physical parameters and evaluation gates are unchanged.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from flash_chord.envs.observation import PolicyObservationConfig
from .objective import Dex3ObjectiveConfig
from .payload import payload_plan
from .sampling import resample_targets


@wp.kernel
def append_force_observation(
    base: wp.array(dtype=wp.float32),
    base_dim: int,
    total_dim: int,
    slots: int,
    bodies: int,
    frames: int,
    timestep: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
    contact_axis: wp.array(dtype=wp.int32),
    object_ids: wp.array(dtype=wp.int32),
    slot_objects: wp.array(dtype=wp.int32),
    poses: wp.array(dtype=wp.transform),
    force: wp.array(dtype=wp.vec3),
    force_scale: float,
    maximum_observation_abs: float,
    output: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    for axis in range(base_dim):
        observed = base[world * base_dim + axis]
        if maximum_observation_abs > 0.0 and wp.isfinite(observed):
            observed = wp.clamp(
                observed, -maximum_observation_abs, maximum_observation_abs
            )
        output[world * total_dim + axis] = observed
        if episode_step[world] == 0 and contact_axis[axis] != 0:
            output[world * total_dim + axis] = 0.0
    for slot in range(slots):
        obj = world * bodies + object_ids[slot_objects[slot]]
        local_force = wp.quat_rotate_inv(
            wp.transform_get_rotation(poses[obj]), force[world * slots + slot]
        )
        if episode_step[world] == 0:
            local_force = wp.vec3()
        for axis in range(3):
            force_value = local_force[axis] / force_scale
            if wp.isfinite(force_value):
                force_value = wp.clamp(force_value, -5.0, 5.0)
            output[world * total_dim + base_dim + slot * 3 + axis] = force_value
    output[world * total_dim + total_dim - 1] = wp.clamp(
        float(timestep[world]) / float(wp.max(1, frames - 1)), 0.0, 1.0
    )


@dataclass(frozen=True)
class GripObservationConfig(PolicyObservationConfig):
    contact_force_scale_N: float = 20.0
    maximum_observation_abs: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if (
            not np.isfinite(self.contact_force_scale_N)
            or self.contact_force_scale_N <= 0
        ):
            raise ValueError("Need a positive contact force observation scale")
        if self.maximum_observation_abs is not None and (
            not np.isfinite(self.maximum_observation_abs)
            or self.maximum_observation_abs < 10
        ):
            raise ValueError(
                "Observation magnitude bound must be finite and at least10"
            )

    def build(
        self,
        scene,
        action,
        command,
        contact,
        timestep,
        episode_step,
        reference_joint_q_offset,
        device=None,
    ):
        if contact is None:
            raise ValueError("Grip observations require actual simulator contacts")
        base = super().build(
            scene,
            action,
            command,
            contact,
            timestep,
            episode_step,
            reference_joint_q_offset,
            device,
        )
        return GripObservation(
            base,
            scene,
            contact,
            timestep,
            episode_step,
            action.buffers.reference.num_frames,
            self.contact_force_scale_N,
            self.maximum_observation_abs or 0.0,
            device,
        )


class GripObservation:
    def __init__(
        self,
        base,
        scene,
        contact,
        timestep,
        episode_step,
        frames,
        scale,
        maximum_observation_abs,
        device,
    ):
        self.base, self.contact, self.timestep = base, contact, timestep
        self.maximum_observation_abs = maximum_observation_abs
        self.episode_step = episode_step
        contact_axis = np.zeros(base.observation_dim, dtype=np.int32)
        for name, (start, end) in zip(base.block_names, base.block_ranges, strict=True):
            if name.startswith("contact_"):
                contact_axis[start:end] = 1
        self.contact_axis = wp.array(contact_axis, dtype=wp.int32, device=device)
        self.worlds, self.bodies = (
            scene.world_count,
            scene.model.body_count // scene.world_count,
        )
        self.slots, self.frames, self.scale = (
            contact.layout.slots_per_world,
            frames,
            scale,
        )
        self.observation_dim = base.observation_dim + self.slots * 3 + 1
        self.observation = wp.zeros(
            self.worlds * self.observation_dim, dtype=wp.float32, device=device
        )
        self.block_names = base.block_names + (
            "measured_contact_forces_object_frame",
            "reference_phase",
        )
        self.block_ranges = base.block_ranges + (
            (base.observation_dim, self.observation_dim - 1),
            (self.observation_dim - 1, self.observation_dim),
        )

    def compute_into(self, state, output):
        self.base.compute(state)
        wp.launch(
            append_force_observation,
            dim=self.worlds,
            inputs=[
                self.base.observation,
                self.base.observation_dim,
                self.observation_dim,
                self.slots,
                self.bodies,
                self.frames,
                self.timestep,
                self.episode_step,
                self.contact_axis,
                self.contact.object_body_ids,
                self.contact.slot_object_ids,
                state.body_q,
                self.contact.contact_force_w,
                self.scale,
                self.maximum_observation_abs,
                output,
            ],
        )
        return output

    def compute(self, state):
        return self.compute_into(state, self.observation)


@wp.func
def grip_components(
    support: float,
    force_error: wp.vec3,
    moment_error: wp.vec3,
    relative_velocity: wp.vec3,
    relative_omega: wp.vec3,
    height_error: float,
    carry: float,
    excess: float,
):
    stability = support * wp.exp(
        -wp.length_sq(relative_velocity) / 0.0025 - wp.length_sq(relative_omega)
    )
    load = (
        carry
        * support
        * wp.exp(-wp.length_sq(force_error) / 9.0 - wp.length_sq(moment_error) / 0.0064)
    )
    lift = carry * support * wp.exp(-height_error * height_error / 0.0001)
    return wp.spatial_vector(support, stability, load, lift, excess, 0.0)


@wp.kernel
def evaluate_grip(
    timestep: wp.array(dtype=wp.int32),
    frames: int,
    bodies: int,
    hands: int,
    slots: int,
    base_terms: wp.array(dtype=wp.float32),
    base_count: int,
    base_score: wp.array(dtype=wp.float32),
    frame_dt: float,
    poses: wp.array(dtype=wp.transform),
    velocities: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    hand_ids: wp.array(dtype=wp.int32),
    object_ids: wp.array(dtype=wp.int32),
    starts: wp.array(dtype=wp.int32),
    link_counts: wp.array(dtype=wp.int32),
    required: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.float32),
    carry: wp.array(dtype=wp.float32),
    lift_origin: wp.array(dtype=wp.float32),
    shaping: int,
    target_com: wp.array(dtype=wp.vec3),
    load: wp.array(dtype=wp.spatial_vector),
    contact_pos: wp.array(dtype=wp.vec3),
    contact_force: wp.array(dtype=wp.vec3),
    weights: wp.array(dtype=wp.float32),
    terms: wp.array(dtype=wp.float32),
    score: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    frame = wp.clamp(timestep[world], 0, frames - 1)
    total_count = base_count + 5
    for ti in range(base_count):
        terms[world * total_count + ti] = base_terms[world * base_count + ti]
    accumulated = wp.spatial_vector()
    count = float(0.0)
    for hand in range(hands):
        ri = frame * hands + hand
        if active[ri] > 0.0:
            count += 1.0
            obj, wrist = (
                world * bodies + object_ids[hand],
                world * bodies + hand_ids[hand],
            )
            center = wp.transform_point(poses[obj], body_com[obj])
            wrist_com = wp.transform_point(poses[wrist], body_com[wrist])
            f, moment = wp.vec3(), wp.vec3()
            excess = float(0.0)
            for link in range(link_counts[hand]):
                slot = world * slots + starts[hand] + link
                measured = contact_force[slot]
                f += measured
                moment += wp.cross(contact_pos[slot] - center, measured)
                above = wp.max(0.0, wp.length(measured) - 20.0) / 20.0
                excess += wp.min(above * above, 100.0)
            support = float(1.0)
            for pad in range(3):
                slot = required[ri * 3 + pad]
                if slot >= 0:
                    support = wp.min(
                        support,
                        wp.clamp(
                            wp.length(contact_force[world * slots + slot]) / 0.5,
                            0.0,
                            1.0,
                        ),
                    )
            ov, hv = velocities[obj], velocities[wrist]
            relative_velocity = (
                wp.spatial_top(ov)
                - wp.spatial_top(hv)
                - wp.cross(wp.spatial_bottom(hv), center - wrist_com)
            )
            relative_omega = wp.spatial_bottom(ov) - wp.spatial_bottom(hv)
            desired = load[ri]
            value = grip_components(
                support,
                f - wp.spatial_top(desired),
                moment - wp.spatial_bottom(desired),
                relative_velocity,
                relative_omega,
                center[2] - target_com[ri][2],
                carry[ri],
                excess,
            )
            if shaping == 1:
                force_error = f - wp.spatial_top(desired)
                moment_error = moment - wp.spatial_bottom(desired)
                load_value = (
                    carry[ri]
                    * support
                    / (
                        1.0
                        + wp.length_sq(force_error) / 9.0
                        + wp.length_sq(moment_error) / 0.0064
                    )
                )
                wanted_lift = wp.max(0.001, target_com[ri][2] - lift_origin[ri])
                achieved_lift = wp.max(0.0, center[2] - lift_origin[ri])
                progress = wp.clamp(achieved_lift / wanted_lift, 0.0, 1.0)
                height_error = center[2] - target_com[ri][2]
                height_value = (
                    carry[ri]
                    * support
                    * progress
                    / (1.0 + height_error * height_error / 0.0001)
                )
                value = wp.spatial_vector(
                    value[0], value[1], load_value, height_value, value[4], 0.0
                )
            accumulated += value
    total = base_score[world]
    for ti in range(5):
        component = accumulated[ti] / wp.max(1.0, count)
        terms[world * total_count + base_count + ti] = component
        total += frame_dt * weights[base_count + ti] * component
    score[world] = total


@dataclass(frozen=True)
class GripObjectiveConfig(Dex3ObjectiveConfig):
    input_root: str = "/data"
    pad_support_weight: float = 3.0
    grip_stability_weight: float = 1.0
    carried_load_weight: float = 4.0
    carried_height_weight: float = 6.0
    excess_pad_force_weight: float = -0.02
    load_shaping: str = "gaussian"

    def __post_init__(self):
        super().__post_init__()
        if self.load_shaping not in ("gaussian", "rational_lift"):
            raise ValueError("Unknown explicit grip shaping mode")
        values = self.grip_weights
        if (
            not np.isfinite(values).all()
            or min(values[:4]) < 0
            or max(values[:4]) > 50
            or not -1 <= values[4] <= 0
        ):
            raise ValueError("Invalid bounded simulator grip objective weights")

    @property
    def grip_weights(self):
        return (
            self.pad_support_weight,
            self.grip_stability_weight,
            self.carried_load_weight,
            self.carried_height_weight,
            self.excess_pad_force_weight,
        )

    def build(
        self,
        model,
        embodiment,
        robot_reference,
        reference,
        command,
        contact,
        action,
        world_count,
        frame_dt,
        device=None,
        reference_joint_q_offset=None,
        episode_start_frame=None,
    ):
        base = super().build(
            model,
            embodiment,
            robot_reference,
            reference,
            command,
            contact,
            action,
            world_count,
            frame_dt,
            device,
            reference_joint_q_offset,
            episode_start_frame,
        )
        return GripObjective(
            base,
            self,
            model,
            embodiment,
            reference,
            contact,
            world_count,
            frame_dt,
            device,
        )


class GripObjective:
    def __init__(
        self,
        base,
        config,
        model,
        embodiment,
        reference,
        contact,
        worlds,
        frame_dt,
        device,
    ):
        if contact is None:
            raise ValueError("Grip objective requires measured simulator contacts")
        root = Path(config.input_root)
        manifest = json.loads((root / "manifest.json").read_text())
        for name in ("task-grips.json", "targets.npz"):
            if (
                hashlib.sha256((root / name).read_bytes()).hexdigest()
                != manifest["files"][name]
            ):
                raise ValueError("Grip reward inputs differ from the frozen manifest")
        spec = json.loads((root / "task-grips.json").read_text())
        with np.load(root / "targets.npz") as f:
            data = resample_targets(
                {k: f[k].copy() for k in f.files}, reference.num_frames
            )
        self.base, self.model, self.contact = base, model, contact
        self.worlds, self.dt = worlds, frame_dt
        self.frames, self.hands = reference.num_frames, len(contact.layout.sides)
        self.bodies = model.body_count // worlds
        labels = list(model.body_label[: self.bodies])
        hand_ids, object_ids, starts, counts = [], [], [], []
        active, carry, target_com, load, required, origin = [], [], [], [], [], []
        mass, inertia, com = (
            model.body_mass.numpy(),
            model.body_inertia.numpy(),
            model.body_com.numpy(),
        )
        for hi, side in enumerate(contact.layout.sides):
            phases = spec["phases"][side]
            assignment = {p["object"] for p in phases}
            if len(assignment) != 1:
                raise ValueError("Grip reward needs one assigned object per hand")
            oi = next(iter(assignment))
            obj = labels.index(manifest["objects"][oi]["name"] + "/object")
            hand_ids.append(embodiment.hand(side).palm_frame.body_id)
            object_ids.append(obj)
            object_slot = list(contact.layout.object_body_ids).index(obj)
            count = contact.layout.link_counts[hi]
            start = contact.layout.hand_slot_starts[hi] + object_slot * count
            starts.append(start)
            counts.append(count)
            links = [g.link_name for g in embodiment.hand(side).contact_links]
            positions = data["object_poses"][:, oi].copy()
            positions[:, :3] += Rotation.from_quat(positions[:, 3:]).apply(
                np.broadcast_to(com[obj], (self.frames, 3))
            )
            target_com.append(positions[:, :3])
            load.append(
                payload_plan(
                    data["timestamps_s"],
                    data["object_poses"][:, oi],
                    phases,
                    float(mass[obj]),
                    inertia[obj],
                    com[obj],
                )
            )
            a, c = np.zeros(self.frames), np.zeros(self.frames)
            phase_origin = np.zeros(self.frames)
            req = np.full((self.frames, 3), -1, dtype=np.int32)
            for phase in phases:
                indices = np.flatnonzero(
                    (data["timestamps_s"] >= phase["close_s"] - 1e-8)
                    & (data["timestamps_s"] <= phase["release_start_s"] - 0.05 + 1e-8)
                )
                pads = [
                    p["link"]
                    for p in phase["pad_normal_jacobians"]
                    if any(d in p["link"] for d in phase["required_digits"])
                ]
                if not 1 <= len(pads) <= 3 or len(pads) != len(
                    phase["required_digits"]
                ):
                    raise ValueError("Incomplete robot pad objective mapping")
                a[indices] = 1
                z0 = np.interp(phase["close_s"], data["timestamps_s"], positions[:, 2])
                phase_origin[indices] = z0
                c[indices] = np.clip((positions[indices, 2] - z0) / 0.01, 0, 1)
                req[np.ix_(indices, range(len(pads)))] = [
                    start + links.index(p) for p in pads
                ]
            active.append(a)
            carry.append(c)
            required.append(req)
            origin.append(phase_origin)

        def array(values, dtype, components=1):
            values = np.stack(values, axis=1)
            values = (
                values.reshape(-1, components) if components > 1 else values.ravel()
            )
            return wp.array(values, dtype=dtype, device=device)

        self.active = array(active, wp.float32)
        self.carry = array(carry, wp.float32)
        self.lift_origin = array(origin, wp.float32)
        self.shaping = int(config.load_shaping == "rational_lift")
        self.target_com = array(target_com, wp.vec3, 3)
        self.load = array(load, wp.spatial_vector, 6)
        self.required = array(required, wp.int32)
        self.hand_ids = wp.array(hand_ids, dtype=wp.int32, device=device)
        self.object_ids = wp.array(object_ids, dtype=wp.int32, device=device)
        self.starts = wp.array(starts, dtype=wp.int32, device=device)
        self.counts = wp.array(counts, dtype=wp.int32, device=device)
        self.term_names = base.term_names + (
            "robot_pad_support",
            "robot_grip_stability",
            "robot_carried_load",
            "robot_carried_height",
            "robot_excess_pad_force",
        )
        self.weights = wp.array(
            np.r_[base.weights.numpy(), config.grip_weights],
            dtype=wp.float32,
            device=device,
        )
        self.terms = wp.zeros(
            worlds * len(self.term_names), dtype=wp.float32, device=device
        )
        self.score = wp.zeros(worlds, dtype=wp.float32, device=device)

    def reset(self, mask):
        self.base.reset(mask)

    def set_weights(self, weights):
        if set(weights) != set(self.term_names) or not all(
            np.isfinite(float(v)) for v in weights.values()
        ):
            raise ValueError("Grip curriculum objective mapping differs")
        self.base.set_weights({name: weights[name] for name in self.base.term_names})
        self.weights.assign([weights[name] for name in self.term_names])

    def evaluate(self, state, timestep, terminated):
        self.base.evaluate(state, timestep, terminated)
        wp.launch(
            evaluate_grip,
            dim=self.worlds,
            inputs=[
                timestep,
                self.frames,
                self.bodies,
                self.hands,
                self.contact.layout.slots_per_world,
                self.base.terms,
                len(self.base.term_names),
                self.base.score,
                self.dt,
                state.body_q,
                state.body_qd,
                self.model.body_com,
                self.hand_ids,
                self.object_ids,
                self.starts,
                self.counts,
                self.required,
                self.active,
                self.carry,
                self.lift_origin,
                self.shaping,
                self.target_com,
                self.load,
                self.contact.contact_pos_w,
                self.contact.contact_force_w,
                self.weights,
                self.terms,
                self.score,
            ],
        )
