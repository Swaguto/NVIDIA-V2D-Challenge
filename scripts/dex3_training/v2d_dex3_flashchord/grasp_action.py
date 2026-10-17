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
from .phase_prior import schedule as prior_schedule, model_signature, control_signature


@wp.kernel
def add_phase_policy_prior(
    timestep: wp.array(dtype=wp.int32),
    frames: int,
    dimensions: int,
    actor: wp.array(dtype=wp.float32),
    prior: wp.array(dtype=wp.float32),
    combined: wp.array(dtype=wp.float32),
):
    world, axis = wp.tid()
    frame = wp.clamp(timestep[world], 0, frames - 1)
    combined[world * dimensions + axis] = (
        actor[world * dimensions + axis] + prior[frame * dimensions + axis]
    )


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


@wp.kernel
def gather_reference_velocity(
    timestep: wp.array(dtype=wp.int32),
    frames: int,
    dofs: int,
    wrist_dofs: wp.array(dtype=wp.int32),
    velocity: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
):
    world, component = wp.tid()
    frame = wp.clamp(timestep[world], 0, frames - 1)
    target[world * dofs + wrist_dofs[component]] = velocity[frame * 6 + component]


@dataclass(frozen=True)
class GraspActionConfig(ActionConfig):
    input_root: str = "/data"
    wrist_reference_velocity: bool = False
    maximum_reference_wrist_speed_m_s: float = 1.0
    wrist_angular_reference_velocity: bool = False
    wrist_angular_kd: float = 0.4
    maximum_reference_wrist_angular_speed_rad_s: float = 10.0
    phase_policy_prior: str | None = None
    phase_policy_physics_substeps: int = 20
    wrist_payload_compensation: bool = False
    wrist_payload_gain: float = 1.0
    wrist_payload_carry_gain: float | None = None

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
                {k: data[k].copy() for k in data.files},
                base.buffers.reference.num_frames,
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
        result = GripResidualAction(
            base, np.concatenate(arranged, axis=1), scene.model.joint_target_ke, device
        )
        result.native_physics_model_sha256 = model_signature(
            scene.model, scene.world_count
        )
        result.policy_control_contract_sha256 = control_signature(self)
        if self.wrist_angular_reference_velocity:
            from .angular_servo import AngularReferenceServo
            result.angular_servo = AngularReferenceServo(
                result, scene, sampled, self.wrist_angular_kd,
                self.maximum_reference_wrist_angular_speed_rad_s,
            )
        if self.wrist_reference_velocity:
            velocity = np.gradient(
                sampled["robot_qpos"][:, :, :3], sampled["timestamps_s"], axis=0
            )
            if (
                not 0 < self.maximum_reference_wrist_speed_m_s <= 2.0
                or not np.isfinite(velocity).all()
                or np.linalg.norm(velocity, axis=-1).max()
                > self.maximum_reference_wrist_speed_m_s
            ):
                raise ValueError(
                    "Reference wrist velocity exceeds the declared motor plan"
                )
            # The final command is a hold; no extrapolated motion after source end.
            velocity[-1] = 0
            result.reference_velocity = wp.array(
                np.concatenate(
                    [velocity[:, ("left", "right").index(s)] for s in base.sides],
                    axis=1,
                ).ravel(),
                dtype=wp.float32,
                device=device,
            )
            result.velocity_dofs = wp.array(
                [d for s in base.sides for d in scene.layout.hand(s).wrist_pos_dof_ids],
                dtype=wp.int32,
                device=device,
            )
            result.motor_velocity_target = wp.zeros_like(result.joint_target)
        if self.phase_policy_prior is not None:
            path = (root / self.phase_policy_prior).resolve()
            if (
                not path.is_relative_to(root)
                or self.phase_policy_prior not in manifest["files"]
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != manifest["files"][self.phase_policy_prior]
            ):
                raise ValueError("Phase-policy prior is not bound to these inputs")
            plan = json.loads(path.read_text())
            if plan["input_manifest_sha256"] != manifest.get(
                "phase_policy_prior_base_manifest_sha256"
            ):
                raise ValueError("Fitted phase policy belongs to other inputs")
            if (
                plan.get("native_physics_model_sha256")
                != model_signature(scene.model, scene.world_count)
                or plan.get("physics_substeps") != self.phase_policy_physics_substeps
            ):
                raise ValueError(
                    "Fitted phase policy physics differs from the loaded scene"
                )
            if (
                plan.get("action_control_contract_sha256")
                != result.policy_control_contract_sha256
            ):
                raise ValueError(
                    "Fitted phase policy controller differs from the loaded action"
                )
            control_fps = 1 / np.median(np.diff(sampled["timestamps_s"]))
            if not np.isclose(plan.get("control_fps", -1), control_fps):
                raise ValueError("Fitted phase policy control clock differs")
            if self.input_mode != "raw" or (
                self.wrist_pos_scale,
                self.wrist_ori_scale,
                self.finger_joint_pos_scale,
            ) != (0.05, 0.15, 0.15):
                raise ValueError(
                    "Fitted phase prior requires the measured residual action scale"
                )
            policy = prior_schedule(
                json.loads((root / "task-grips.json").read_text()),
                plan,
                list(base.sides),
                {s: scene.layout.hand(s).finger_joint_names for s in base.sides},
                sampled["timestamps_s"],
            )
            result.phase_policy_prior = wp.array(
                policy.ravel(), dtype=wp.float32, device=device
            )
            result.combined_actor_action = wp.zeros_like(result.buffers.raw_action)
        return result


class GripResidualAction(ResidualHandPoseAction):
    def __init__(self, base, feedforward, stiffness, device):
        super().__init__(base.buffers, base.wrist, base.delay)
        self.angular_servo = None
        self.grip_feedforward = wp.array(
            feedforward.ravel(), dtype=wp.float32, device=device
        )
        self.grip_stiffness = stiffness
        self.reference_velocity = None
        self.phase_policy_prior = None

    def process(self, action, timestep, state=None):
        if self.phase_policy_prior is not None:
            wp.launch(
                add_phase_policy_prior,
                dim=(self.buffers.world_count, self.action_dim),
                inputs=[
                    timestep,
                    self.buffers.reference.num_frames,
                    self.action_dim,
                    action,
                    self.phase_policy_prior,
                    self.combined_actor_action,
                ],
            )
            action = self.combined_actor_action
        super().process(action, timestep, state)
        if self.angular_servo is not None:
            self.angular_servo.process(timestep)
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
        if self.reference_velocity is not None:
            wp.launch(
                gather_reference_velocity,
                dim=(self.buffers.world_count, 6),
                inputs=[
                    timestep,
                    self.buffers.reference.num_frames,
                    layout.num_joint_dof,
                    self.velocity_dofs,
                    self.reference_velocity,
                    self.motor_velocity_target,
                ],
            )

    def prepare_control(self, control):
        super().prepare_control(control)
        if self.reference_velocity is not None:
            wp.copy(control.joint_target_vel, self.motor_velocity_target)

    def reset(self, mask):
        super().reset(mask)
        if self.angular_servo is not None:
            self.angular_servo.reset(mask)

    def apply_control(self, state, control):
        super().apply_control(state, control)
        if self.angular_servo is not None:
            self.angular_servo.apply(state, control)
