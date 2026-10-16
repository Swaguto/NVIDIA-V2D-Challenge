"""Bounded simulator-pose feedback for acquiring frozen inferred grasps.

Only robot motor targets change. Source references, object states, support,
geometry, masses, friction and native effort limits remain untouched.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import warp as wp
from flash_chord.utils.quat import (
    quat_inv_xyzw,
    quat_mul_xyzw,
    quat_geodesic_angle,
    quat_to_rotvec,
)
from .grasp_action import GraspActionConfig, GripResidualAction


@wp.func
def smooth_unit(x: float) -> float:
    u = wp.clamp(x, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@wp.func
def blend_rotation(a: wp.quat, b: wp.quat, w: float) -> wp.quat:
    sign = float(1.0)
    if a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] < 0.0:
        sign = -1.0
    return wp.normalize(
        wp.quat(
            (1.0 - w) * a[0] + w * sign * b[0],
            (1.0 - w) * a[1] + w * sign * b[1],
            (1.0 - w) * a[2] + w * sign * b[2],
            (1.0 - w) * a[3] + w * sign * b[3],
        )
    )


@wp.kernel
def reset_feedback(
    mask: wp.array(dtype=wp.int32),
    hands: int,
    phase: wp.array(dtype=wp.int32),
    registered: wp.array(dtype=wp.int32),
):
    world, hand = wp.tid()
    if mask[world] != 0:
        phase[world * hands + hand] = -1
        registered[world * hands + hand] = 0


@wp.kernel
def correct_grasp_targets(
    timestep: wp.array(dtype=wp.int32),
    frames: int,
    hands: int,
    bodies: int,
    dofs: int,
    processed_dim: int,
    body_pose: wp.array(dtype=wp.transform),
    object_body: wp.array(dtype=wp.int32),
    hand_body: wp.array(dtype=wp.int32),
    ref_object: wp.array(dtype=wp.transform),
    ref_hand: wp.array(dtype=wp.transform),
    grasp: wp.array(dtype=wp.transform),
    park: wp.array(dtype=wp.float32),
    weight: wp.array(dtype=wp.float32),
    capture: wp.array(dtype=wp.int32),
    reference_phase: wp.array(dtype=wp.int32),
    processed_starts: wp.array(dtype=wp.int32),
    pos_dofs: wp.array(dtype=wp.int32),
    rot_dofs: wp.array(dtype=wp.int32),
    last_phase: wp.array(dtype=wp.int32),
    registered: wp.array(dtype=wp.int32),
    offset: wp.array(dtype=wp.vec3),
    bias: wp.array(dtype=wp.quat),
    processed: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
    actual_feedback: wp.array(dtype=wp.float32),
):
    world, hand = wp.tid()
    frame = wp.clamp(timestep[world], 0, frames - 1)
    ri = frame * hands + hand
    si = world * hands + hand
    ref = ref_object[ri]
    rp = wp.transform_get_translation(ref)
    rq = wp.transform_get_rotation(ref)
    measured = body_pose[world * bodies + object_body[hand]]
    mp = wp.transform_get_translation(measured)
    mq = wp.transform_get_rotation(measured)
    if last_phase[si] != reference_phase[ri]:
        last_phase[si] = reference_phase[ri]
        registered[si] = 0
        offset[si] = wp.vec3(0.0)
        bias[si] = wp.quat(0.0, 0.0, 0.0, 1.0)
    if capture[ri] != 0 and registered[si] == 0:
        if wp.length(mp - rp) <= 0.03 and quat_geodesic_angle(mq, rq) <= 0.1745329252:
            offset[si] = mp - rp
            bias[si] = quat_mul_xyzw(quat_inv_xyzw(rq), mq)
            registered[si] = 1
    goal_p = rp
    goal_q = rq
    if registered[si] != 0:
        goal_p = rp + offset[si]
        goal_q = quat_mul_xyzw(rq, bias[si])
    distance = wp.length(mp - goal_p)
    angle = quat_geodesic_angle(mq, goal_q)
    w = (
        weight[ri]
        * (1.0 - smooth_unit((distance - 0.03) / 0.03))
        * (1.0 - smooth_unit((angle - 0.1745329252) / 0.1745329252))
    )
    actual_feedback[si] = w
    object_p = goal_p * (1.0 - w) + mp * w
    object_q = blend_rotation(goal_q, mq, w)
    rel = grasp[ri]
    wanted_p = (
        object_p
        + wp.quat_rotate(object_q, wp.transform_get_translation(rel))
        + wp.vec3(0.0, 0.0, park[ri])
    )
    wanted_q = quat_mul_xyzw(object_q, wp.transform_get_rotation(rel))
    pb = world * processed_dim + processed_starts[hand]
    nominal = ref_hand[ri]
    processed_p = wp.vec3(processed[pb], processed[pb + 1], processed[pb + 2])
    processed_q = wp.quat(
        processed[pb + 4], processed[pb + 5], processed[pb + 6], processed[pb + 3]
    )
    residual_q = quat_mul_xyzw(
        quat_inv_xyzw(wp.transform_get_rotation(nominal)), processed_q
    )
    wanted_p = wanted_p + processed_p - wp.transform_get_translation(nominal)
    wanted_q = wp.normalize(quat_mul_xyzw(wanted_q, residual_q))
    motor_p = wanted_p
    if registered[si] != 0:
        actual_p = wp.transform_get_translation(
            body_pose[world * bodies + hand_body[hand]]
        )
        gain = 1.0 / wp.max(0.1, 1.0 - w)
        motor_p = actual_p + gain * (wanted_p - actual_p)
    rotation = quat_to_rotvec(wanted_q)
    for axis in range(3):
        target[world * dofs + pos_dofs[hand * 3 + axis]] = motor_p[axis]
        target[world * dofs + rot_dofs[hand * 3 + axis]] = rotation[axis]
        processed[pb + axis] = wanted_p[axis]
    processed[pb + 3] = wanted_q[3]
    processed[pb + 4] = wanted_q[0]
    processed[pb + 5] = wanted_q[1]
    processed[pb + 6] = wanted_q[2]


@dataclass(frozen=True)
class FeedbackGraspActionConfig(GraspActionConfig):
    def build(self, scene, device=None):
        base = super().build(scene, device)
        root = Path(self.input_root)
        manifest = json.loads((root / "manifest.json").read_text())
        spec = json.loads((root / "task-grips.json").read_text())
        with np.load(root / "targets.npz", allow_pickle=False) as f:
            data = {k: f[k].copy() for k in f.files}
        object_ids = []
        hand_ids = []
        objects = []
        nominal = []
        grasp = []
        park = []
        weight = []
        capture = []
        phase = []
        for side in base.sides:
            si = ("left", "right").index(side)
            assignments = {p["object"] for p in spec["phases"][side]}
            if len(assignments) != 1:
                raise ValueError("Feedback needs one assigned rigid object per hand")
            oi = next(iter(assignments))
            name = manifest["objects"][oi]["name"]
            binding = next(b for b in scene.objects if b.name == name)
            object_ids.append(binding.root.body_id)
            hand_ids.append(scene.layout.hand(side).palm_frame.body_id)
            objects.append(data["object_poses"][:, oi])
            nominal.append(data["robot_qpos"][:, si, :7])
            grasp.append(data["grasp_pose_object_xyzw"][:, si])
            park.append(data["park_height_world_m"][:, si])
            weight.append(data["measured_object_feedback_weight"][:, si])
            capture.append(data["capture_settled_registration"][:, si])
            phase.append(data["phase_index"][:, si])
        result = FeedbackGripAction(base)

        def array(values, dtype):
            return wp.array(
                np.stack(values, axis=1).reshape(-1, 7)
                if dtype == wp.transform
                else np.stack(values, axis=1).ravel(),
                dtype=dtype,
                device=device,
            )

        result.ref_object = array(objects, wp.transform)
        result.ref_hand = array(nominal, wp.transform)
        result.ref_grasp = array(grasp, wp.transform)
        result.park = array(park, wp.float32)
        result.feedback = array(weight, wp.float32)
        result.capture = array(capture, wp.int32)
        result.phase = array(phase, wp.int32)
        result.object_body = wp.array(object_ids, dtype=wp.int32, device=device)
        result.hand_body = wp.array(hand_ids, dtype=wp.int32, device=device)
        result.body_stride = scene.model.body_count // scene.world_count
        total = scene.world_count * len(base.sides)
        result.last_phase = wp.full(total, -1, dtype=wp.int32, device=device)
        result.registered = wp.zeros(total, dtype=wp.int32, device=device)
        result.offset = wp.zeros(total, dtype=wp.vec3, device=device)
        result.bias = wp.zeros(total, dtype=wp.quat, device=device)
        result.actual_feedback = wp.zeros(total, dtype=wp.float32, device=device)
        return result


class FeedbackGripAction(GripResidualAction):
    def __init__(self, base):
        # Reuse the already validated native motor plan and its buffers.
        self.buffers, self.wrist, self.delay = base.buffers, base.wrist, base.delay
        self.grip_feedforward, self.grip_stiffness = (
            base.grip_feedforward,
            base.grip_stiffness,
        )

    def process(self, action, timestep, state=None):
        if state is None:
            raise ValueError("Measured grasp feedback requires live simulator state")
        super().process(action, timestep, state)
        layout = self.buffers.layout
        wp.launch(
            correct_grasp_targets,
            dim=(self.buffers.world_count, layout.num_hands),
            inputs=[
                timestep,
                self.buffers.reference.num_frames,
                layout.num_hands,
                self.body_stride,
                layout.num_joint_dof,
                layout.processed_dim,
                state.body_q,
                self.object_body,
                self.hand_body,
                self.ref_object,
                self.ref_hand,
                self.ref_grasp,
                self.park,
                self.feedback,
                self.capture,
                self.phase,
                self.buffers.hand_processed_starts,
                self.buffers.wrist_pos_dof_ids,
                self.buffers.wrist_ori_dof_ids,
                self.last_phase,
                self.registered,
                self.offset,
                self.bias,
                self.processed_target,
                self.joint_target,
                self.actual_feedback,
            ],
        )

    def reset(self, mask):
        super().reset(mask)
        wp.launch(
            reset_feedback,
            dim=(self.buffers.world_count, self.buffers.layout.num_hands),
            inputs=[
                mask,
                self.buffers.layout.num_hands,
                self.last_phase,
                self.registered,
            ],
        )
