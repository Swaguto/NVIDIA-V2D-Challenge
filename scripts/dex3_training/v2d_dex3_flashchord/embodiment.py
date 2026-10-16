# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two genuine seven-joint Dex3 hands, using upstream wrist drives and binding.

Only the floating-wrist construction is shared with Sharpa. Geometry, joints,
semantic frames, and contact ownership are taken from the Dex3 URDFs.
"""

from dataclasses import dataclass, field
from pathlib import Path
import hashlib

import newton
import warp as wp

from flash_chord.embodiments.base import (
    BodyFrame,
    EmbodimentLayout,
    HandLayout,
    JointControlSpec,
)
from flash_chord.embodiments.frames import capture_body_ids
from flash_chord.embodiments.g1_dex3 import (
    _capture_hand_geometry,
    _fingertip_frame,
    _hand_link_names,
)
from flash_chord.embodiments.sharpa_hands import SharpaHands

SIDES = ("left", "right")
DIGITS = ("thumb", "middle", "index")


@dataclass
class Dex3HandsConfig:
    left_urdf_path: Path
    right_urdf_path: Path
    reference_manifest_sha256: str | None = None
    gravity_compensation: bool = True
    hand_contact_profile: str = "default"
    default_joint: JointControlSpec = field(
        default_factory=lambda: JointControlSpec(
            kp=2.0,
            kd=0.2,
            armature=0.00149,
            effort_limit=0.76,
            velocity_limit=23.0,
        )
    )
    joint_overrides: dict = field(
        default_factory=lambda: {
            r"wrist_[xyz]$": dict(kp=50.0, kd=10.0, effort_limit=60.0),
            r"wrist_rotvec$": dict(mode="effort", kp=12.0, kd=0.1, effort_limit=60.0),
        }
    )


class Dex3Hands(SharpaHands):
    name = "dex3_hands"

    def __init__(self, config: Dex3HandsConfig):
        self.config = config

    def bind_reference(self, builder, layout, reference):
        manifest = Path(reference.metadata.source_path).parent / "manifest.json"
        actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if actual != self.config.reference_manifest_sha256:
            raise ValueError(
                "Dex3 checkpoint/config reference manifest differs from loaded assets and targets"
            )
        return super().bind_reference(builder, layout, reference)

    def _add_dex3(self, builder, side, path):
        parent = -1
        joints = []
        for label, axis in zip(
            "xyz", (newton.Axis.X, newton.Axis.Y, newton.Axis.Z), strict=True
        ):
            body = builder.add_link(mass=1e-4, label=f"{side}_wrist_p{label}_body")
            joints.append(
                builder.add_joint_prismatic(
                    parent=parent,
                    child=body,
                    axis=axis,
                    label=f"{side}_wrist_{label}",
                    friction=0.01,
                    armature=0.01,
                )
            )
            parent = body
        builder.add_articulation(joints, label=f"{side}_wrist_position")
        builder.add_urdf(
            source=str(path),
            xform=wp.transform_identity(),
            parent_body=parent,
            base_joint={
                "joint_type": newton.JointType.BALL,
                "label": f"{side}_wrist_rotvec",
                "angular_axes": [
                    newton.ModelBuilder.JointDofConfig(
                        axis=a,
                        armature=0.01,
                        friction=0.01,
                    )
                    for a in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
                ],
            },
            force_show_colliders=False,
            enable_self_collisions=True,
            ignore_inertial_definitions=False,
            collapse_fixed_joints=False,
            force_position_velocity_actuation=True,
        )
        return capture_body_ids(builder, _hand_link_names(side))

    def build(self, builder):
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        ids = {}
        for side in SIDES:
            path = Path(getattr(self.config, f"{side}_urdf_path"))
            if not path.is_file():
                raise FileNotFoundError(path)
            ids.update(self._add_dex3(builder, side, path))
        geometry = _capture_hand_geometry(builder, ids)
        profile = self.config.hand_contact_profile
        if profile not in {"default", "firm_point", "firm_patch"}:
            raise ValueError("Unknown explicit native hand contact profile")
        if profile != "default":
            hand_bodies = set(ids.values())
            for shape, body in enumerate(builder.shape_body):
                if body not in hand_bodies:
                    continue
                # Newton converts k=1/tau^2, d=2/tau at unit normalized mass
                # to MuJoCo solref=(tau,1). tau=5 ms matches two 400 Hz steps.
                builder.shape_material_ke[shape] = 40000.0
                builder.shape_material_kd[shape] = 400.0
                builder.custom_attributes["mujoco:geom_priority"].values[shape] = 1
                builder.custom_attributes["mujoco:geom_solimp"].values[shape] = (
                    0.99,
                    0.999,
                    0.001,
                    0.5,
                    2.0,
                )
                builder.custom_attributes["mujoco:condim"].values[shape] = (
                    4 if profile == "firm_patch" else 3
                )
                # Existing sliding/torsional/rolling friction coefficients,
                # geometry, inertias and native motor caps stay unchanged.
        self._apply_control_config(builder)
        hands = []
        for side in SIDES:
            pos, orient, finger = [], [], []
            orient_q = None
            names = []
            for j, label in enumerate(builder.joint_label):
                label = label.split("/")[-1]
                if not label.startswith(side + "_"):
                    continue
                dof, q = builder.joint_qd_start[j], builder.joint_q_start[j]
                if builder.joint_type[j] == newton.JointType.PRISMATIC:
                    pos.append((dof, q))
                elif builder.joint_type[j] == newton.JointType.BALL:
                    orient, orient_q = list(range(dof, dof + 3)), q
                elif builder.joint_type[j] == newton.JointType.REVOLUTE:
                    finger.append((dof, q))
                    names.append(label)
            if len(finger) != 7 or len(pos) != 3 or orient_q is None:
                raise ValueError(
                    "Dex3 requires three wrist translations, one ball, seven finger joints"
                )
            dp = tuple(
                BodyFrame(
                    name=f"{side}_hand_{digit}_{2 if digit == 'thumb' else 1}_link",
                    body_id=ids[
                        f"{side}_hand_{digit}_{2 if digit == 'thumb' else 1}_link"
                    ],
                )
                for digit in DIGITS
            )
            hands.append(
                HandLayout(
                    side=side,
                    wrist_pos_dof_ids=tuple(d for d, _ in pos),
                    wrist_pos_q_ids=tuple(q for _, q in pos),
                    wrist_orient_dof_ids=tuple(orient),
                    wrist_orient_q_id=orient_q,
                    finger_dof_ids=tuple(d for d, _ in finger),
                    finger_q_ids=tuple(q for _, q in finger),
                    finger_joint_names=tuple(names),
                    link_geometry=geometry[side],
                    palm_frame=BodyFrame(
                        name=f"{side}_hand_palm_link",
                        body_id=ids[f"{side}_hand_palm_link"],
                    ),
                    dp_frames=dp,
                    fingertip_frames=tuple(
                        _fingertip_frame(side, digit, frame)
                        for digit, frame in zip(DIGITS, dp, strict=True)
                    ),
                )
            )
        return EmbodimentLayout(
            num_joint_q=len(builder.joint_q),
            num_joint_dof=builder.joint_dof_count,
            hands=tuple(hands),
        )
