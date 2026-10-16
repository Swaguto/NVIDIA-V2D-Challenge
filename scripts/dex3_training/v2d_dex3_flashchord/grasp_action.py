"""Bounded inferred finger feedforward inside Newton's native capped drives.

The policy keeps upstream residual wrist/finger semantics. Feedforward shifts
only the motor setpoint after residual processing; geometric references and
observations remain unchanged. It never applies an object wrench.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from flash_chord.runtime.actions import ActionConfig, ResidualHandPoseAction
from .sampling import resample_targets


@wp.kernel
def add_grip_motor_bias(
    timestep: wp.array(dtype=wp.int32),
    num_frames: int,
    finger_count: int,
    dofs_per_world: int,
    finger_dofs: wp.array(dtype=wp.int32),
    feedforward: wp.array(dtype=wp.float32),
    stiffness: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
):
    world, finger = wp.tid()
    frame = wp.clamp(timestep[world], 0, num_frames - 1)
    dof = world * dofs_per_world + finger_dofs[finger]
    target[dof] = (
        target[dof] + feedforward[frame * finger_count + finger] / stiffness[dof]
    )


@dataclass(frozen=True)
class GraspActionConfig(ActionConfig):
    input_root: str = "/data"

    def build(self, scene, device=None):
        if self.delay.max_steps != 0:
            raise ValueError("Grasp feedforward currently requires zero action delay")
        root = Path(self.input_root).resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["schema"] != "v2d.dex3-training-task.v3":
            raise ValueError("Need the explicit inferred task reference contract")
        targets = root / "targets.npz"
        if (
            hashlib.sha256(targets.read_bytes()).hexdigest()
            != manifest["files"]["targets.npz"]
        ):
            raise ValueError("Task motor targets changed")
        base = ResidualHandPoseAction.build(scene, config=self, device=device)
        with np.load(targets, allow_pickle=False) as data:
            sampled = resample_targets(
                {k: data[k].copy() for k in data.files}, base.buffers.reference.num_frames
            )
            ff = sampled["finger_feedforward_Nm"]
            expected_kp = sampled["finger_stiffness_Nm_per_rad"]
        if (
            ff.shape != (base.buffers.reference.num_frames, 2, 7)
            or not np.isfinite(ff).all()
            or abs(ff).max() > 0.55
        ):
            raise ValueError("Invalid bounded task finger feedforward")
        arranged = []
        for side in base.sides:
            si = ("left", "right").index(side)
            indices = [
                manifest["joint_names"][side].index(n)
                for n in scene.layout.hand(side).finger_joint_names
            ]
            arranged.append(ff[:, si, indices])
            dofs = np.array(scene.layout.hand(side).finger_dof_ids)
            actual = scene.model.joint_target_ke.numpy().reshape(scene.world_count, -1)[
                :, dofs
            ]
            wanted = expected_kp[:, si, indices]
            if (
                not np.isfinite(wanted).all()
                or not np.allclose(wanted, wanted[0])
                or not np.allclose(actual, wanted[0])
            ):
                raise ValueError(
                    "Native drive stiffness differs from the frozen motor plan"
                )
            caps = scene.model.joint_effort_limit.numpy().reshape(
                scene.world_count, -1
            )[:, dofs]
            if not np.allclose(caps, 0.76):
                raise ValueError("Native finger motor cap differs from 0.76 Nm")
        return GripResidualAction(
            base, np.concatenate(arranged, axis=1), scene.model.joint_target_ke, device
        )


class GripResidualAction(ResidualHandPoseAction):
    def __init__(self, base, feedforward, stiffness, device):
        super().__init__(base.buffers, base.wrist, base.delay)
        self.grip_feedforward = wp.array(
            feedforward.ravel(), dtype=wp.float32, device=device
        )
        self.grip_stiffness = stiffness

    def process(self, action, timestep, state=None):
        super().process(action, timestep, state)
        layout = self.buffers.layout
        wp.launch(
            add_grip_motor_bias,
            dim=(self.buffers.world_count, layout.total_fingers),
            inputs=[
                timestep,
                self.buffers.reference.num_frames,
                layout.total_fingers,
                layout.num_joint_dof,
                self.buffers.finger_dof_ids,
                self.grip_feedforward,
                self.grip_stiffness,
                self.joint_target,
            ],
        )
