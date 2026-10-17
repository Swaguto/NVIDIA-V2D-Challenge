"""Capped active angular-velocity tracking for the upstream child-frame wrist P loop."""

import numpy as np
import warp as wp
from .angular_trajectory import reference_angular_velocity
from .payload_kernels import reset_payload_clock


@wp.kernel
def reference_angular_effort(
    timestep: wp.array(dtype=wp.int32), frames: int, hands: int, q_stride: int, dof_stride: int,
    q_offsets: wp.array(dtype=wp.int32), dof_offsets: wp.array(dtype=wp.int32),
    reference: wp.array(dtype=wp.vec3), damping: float, caps: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32), joint_qd: wp.array(dtype=wp.float32),
    effort: wp.array(dtype=wp.float32),
):
    world, hand = wp.tid()
    if timestep[world] < 0:
        return
    frame = wp.clamp(timestep[world], 0, frames - 1)
    qb = world * q_stride + q_offsets[hand]
    db = world * dof_stride + dof_offsets[hand]
    current = wp.normalize(wp.quat(joint_q[qb], joint_q[qb + 1], joint_q[qb + 2], joint_q[qb + 3]))
    # Upstream wrist P effort and Newton ball qd are in the child frame.
    desired = wp.quat_rotate_inv(current, reference[frame * hands + hand])
    for axis in range(3):
        effort[db + axis] = wp.clamp(effort[db + axis] + damping * (desired[axis] - joint_qd[db + axis]),
                                     -caps[hand], caps[hand])


class AngularReferenceServo:
    def __init__(self, base, scene, sampled, damping, maximum_speed):
        if not np.isfinite(damping) or not 0 < damping <= 1:
            raise ValueError("Active wrist angular damping must be in (0, 1]")
        self.wrist = base.wrist
        self.worlds = base.buffers.world_count
        self.frames = base.buffers.reference.num_frames
        offsets = self.wrist.qcoord_off.numpy()
        sides = []
        for offset in offsets:
            matches = [h.side for h in scene.layout.hands if h.wrist_orient_q_id == offset]
            if len(matches) != 1:
                raise ValueError("Angular reference wrist/body binding is ambiguous")
            sides.append(matches[0])
        passive = getattr(scene.model.mujoco, "dof_passive_damping", None)
        if passive is None:
            raise ValueError("Need explicit native passive wrist damping")
        values = passive.numpy().reshape(self.worlds, -1)
        dofs = np.concatenate([np.arange(i, i + 3) for i in self.wrist.dof_off.numpy()])
        if not np.allclose(values[:, dofs], 0):
            raise ValueError("Active angular servo requires zero native passive wrist damping")
        omega = np.stack([
            reference_angular_velocity(sampled["timestamps_s"],
                                       sampled["robot_qpos"][:, ("left", "right").index(s), 3:7],
                                       maximum_speed)
            for s in sides
        ], axis=1)
        self.reference = wp.array(omega.reshape(-1, 3), dtype=wp.vec3, device=scene.model.device)
        self.timestep = wp.full(self.worlds, -1, dtype=wp.int32, device=scene.model.device)
        self.damping = damping

    def process(self, timestep):
        wp.copy(self.timestep, timestep)

    def reset(self, mask):
        wp.launch(reset_payload_clock, dim=self.worlds, inputs=[mask, self.timestep])

    def apply(self, state, control):
        wrist = self.wrist
        wp.launch(reference_angular_effort, dim=(self.worlds, wrist.num_wrists),
                  inputs=[self.timestep, self.frames, wrist.num_wrists, wrist.num_joint_q, wrist.num_joint_dof,
                          wrist.qcoord_off, wrist.dof_off, self.reference, self.damping, wrist.tau_max,
                          state.joint_q, state.joint_qd, control.joint_f])
