"""Read-only first-failure snapshots before RL's automatic reset.

This observer changes no states, rewards, failure criteria or reset masks.
It is selected only by the training entry point, never by policy evaluation.
"""

import json
from pathlib import Path

import numpy as np
import warp as wp
from flash_chord.envs.rl import RLEnv

FLAGS = {
    1: "body_pose",
    2: "body_velocity",
    4: "joint_state",
    8: "raw_action",
    16: "objective",
    32: "contact_force",
    64: "tracking_diagnostics",
}


@wp.kernel
def record_first_failure(
    bodies: int,
    nq: int,
    nv: int,
    actions: int,
    terms: int,
    slots: int,
    diagnostics: int,
    poses: wp.array(dtype=wp.transform),
    velocities: wp.array(dtype=wp.spatial_vector),
    q: wp.array(dtype=wp.float32),
    qd: wp.array(dtype=wp.float32),
    action: wp.array(dtype=wp.float32),
    objective: wp.array(dtype=wp.float32),
    score: wp.array(dtype=wp.float32),
    force: wp.array(dtype=wp.vec3),
    tracking: wp.array(dtype=wp.float32),
    timestep: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    flags: wp.array(dtype=wp.int32),
    latest_flags: wp.array(dtype=wp.int32),
    first_frame: wp.array(dtype=wp.int32),
    first_age: wp.array(dtype=wp.int32),
    saved_poses: wp.array(dtype=wp.transform),
    saved_velocities: wp.array(dtype=wp.spatial_vector),
    saved_q: wp.array(dtype=wp.float32),
    saved_qd: wp.array(dtype=wp.float32),
    saved_action: wp.array(dtype=wp.float32),
    saved_objective: wp.array(dtype=wp.float32),
    saved_score: wp.array(dtype=wp.float32),
    saved_force: wp.array(dtype=wp.vec3),
    saved_tracking: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    bad = int(0)
    for body in range(bodies):
        i = world * bodies + body
        p = poses[i]
        v = velocities[i]
        for axis in range(7):
            if not wp.isfinite(p[axis]):
                bad = bad | 1
        for axis in range(6):
            if not wp.isfinite(v[axis]):
                bad = bad | 2
    for axis in range(nq):
        if not wp.isfinite(q[world * nq + axis]):
            bad = bad | 4
    for axis in range(nv):
        if not wp.isfinite(qd[world * nv + axis]):
            bad = bad | 4
    for axis in range(actions):
        if not wp.isfinite(action[world * actions + axis]):
            bad = bad | 8
    for axis in range(terms):
        if not wp.isfinite(objective[world * terms + axis]):
            bad = bad | 16
    if not wp.isfinite(score[world]):
        bad = bad | 16
    for slot in range(slots):
        f = force[world * slots + slot]
        for axis in range(3):
            if not wp.isfinite(f[axis]):
                bad = bad | 32
    for axis in range(diagnostics):
        if not wp.isfinite(tracking[world * diagnostics + axis]):
            bad = bad | 64
    latest_flags[world] = bad
    if bad == 0:
        return
    counts[world] += 1
    if flags[world] != 0:
        return
    flags[world] = bad
    first_frame[world] = timestep[world]
    first_age[world] = episode_step[world]
    for body in range(bodies):
        i = world * bodies + body
        saved_poses[i] = poses[i]
        saved_velocities[i] = velocities[i]
    for axis in range(nq):
        saved_q[world * nq + axis] = q[world * nq + axis]
    for axis in range(nv):
        saved_qd[world * nv + axis] = qd[world * nv + axis]
    for axis in range(actions):
        saved_action[world * actions + axis] = action[world * actions + axis]
    for axis in range(terms):
        saved_objective[world * terms + axis] = objective[world * terms + axis]
    saved_score[world] = score[world]
    for slot in range(slots):
        saved_force[world * slots + slot] = force[world * slots + slot]
    for axis in range(diagnostics):
        saved_tracking[world * diagnostics + axis] = tracking[
            world * diagnostics + axis
        ]


class TransitionTrace:
    def __init__(self, env):
        self.env = env
        worlds = env.world_count
        source = dict(
            body_q=env.state_0.body_q,
            body_qd=env.state_0.body_qd,
            joint_q=env.state_0.joint_q,
            joint_qd=env.state_0.joint_qd,
            raw_action=env.action.buffers.raw_action,
            objective_terms=env.objective.terms,
            objective_score=env.objective.score,
            contact_force=env.contact.contact_force_w,
            tracking_diagnostics=env.termination.packed_diagnostics,
        )
        self.sizes = [
            source[k].size // worlds
            for k in (
                "body_q",
                "joint_q",
                "joint_qd",
                "raw_action",
                "objective_terms",
                "contact_force",
                "tracking_diagnostics",
            )
        ]
        self.arrays = {k: wp.zeros_like(a) for k, a in source.items()}
        self.counts = wp.zeros(worlds, dtype=wp.int32, device=env.device)
        self.flags = wp.zeros_like(self.counts)
        self.latest_flags = wp.zeros_like(self.counts)
        self.first_frame, self.first_age = (
            wp.zeros_like(self.counts),
            wp.zeros_like(self.counts),
        )

    def record(self):
        e = self.env
        wp.launch(
            record_first_failure,
            dim=e.world_count,
            inputs=[
                *self.sizes,
                e.state_0.body_q,
                e.state_0.body_qd,
                e.state_0.joint_q,
                e.state_0.joint_qd,
                e.action.buffers.raw_action,
                e.objective.terms,
                e.objective.score,
                e.contact.contact_force_w,
                e.termination.packed_diagnostics,
                e.timestep,
                e.episode_step,
                self.counts,
                self.flags,
                self.latest_flags,
                self.first_frame,
                self.first_age,
                *self.arrays.values(),
            ],
        )

    def clear(self):
        for array in (
            self.counts,
            self.flags,
            self.latest_flags,
            self.first_frame,
            self.first_age,
            *self.arrays.values(),
        ):
            array.zero_()

    def save(self, output):
        output = Path(output)
        flags, counts = self.flags.numpy(), self.counts.numpy()
        arrays = {k: a.numpy() for k, a in self.arrays.items()}
        arrays.update(
            first_flags=flags,
            event_counts=counts,
            first_source_frame=self.first_frame.numpy(),
            first_episode_step=self.first_age.numpy(),
        )
        with (output / "first-nonfinite-transitions.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report = dict(
            scope="Read-only snapshot before automatic reset; no failed states or rewards repaired",
            nonfinite_world_transition_count=int(counts.sum()),
            worlds=[
                dict(
                    world=int(i),
                    flags=int(flags[i]),
                    categories=[name for bit, name in FLAGS.items() if flags[i] & bit],
                    first_source_frame=int(arrays["first_source_frame"][i]),
                    first_episode_step=int(arrays["first_episode_step"][i]),
                )
                for i in np.flatnonzero(flags)
            ],
        )
        (output / "first-nonfinite-transitions.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return report


class TracedRLEnv(RLEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transition_trace = TransitionTrace(self)

    def _advance_lifecycle(self):
        self.transition_trace.record()
        super()._advance_lifecycle()
