"""Inspect native hand/object and hand/support contacts at every reference knot.

Both robot and object roots are explicitly reset and robot FK is independently
checked. No physics step, force estimate, command repair or grasp certificate.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from check_native_scene import compile_scene


def canonical_id(names, expected):
    matches = [
        i
        for i, name in enumerate(names)
        if name == expected
        or name.endswith("_" + expected)
        or name.endswith("/" + expected)
    ]
    if len(matches) != 1:
        raise ValueError("Native name is missing or ambiguous: " + expected)
    return matches[0]


def summarize(frames, tolerance_m=0.001):
    if not frames or [r["reference_frame"] for r in frames] != list(range(len(frames))):
        raise ValueError("Need every ordered reference knot, including empty contacts")
    pairs = {}
    for row in frames:
        at_frame = {}
        for c in row["contacts"]:
            gap = c["signed_gap_m"]
            if not np.isfinite(gap):
                raise ValueError("Nonfinite native hand contact distance")
            key = (c["hand"], c["target"])
            at_frame[key] = min(at_frame.get(key, 0), gap)
        for key, gap in at_frame.items():
            pairs.setdefault(key, []).append((row["reference_frame"], max(0, -gap)))
    return [
        dict(
            hand=pair[0],
            target=pair[1],
            frames_over_tolerance=sum(depth > tolerance_m for _, depth in values),
            maximum_penetration_m=max(depth for _, depth in values),
            worst_reference_frame=max(values, key=lambda v: v[1])[0],
        )
        for pair, values in sorted(pairs.items())
    ]


def audit(inputs):
    import mujoco
    from v2d_dex3_flashchord.kinematics import URDFKinematics

    started = time.monotonic()
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        p = (inputs / name).resolve()
        if (
            not p.is_relative_to(inputs)
            or hashlib.sha256(p.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Frozen source closure differs: " + name)
    scene, reference, solver, input_sha, runtime = compile_scene(inputs)
    model = solver.mj_model
    with np.load(inputs / "targets.npz", allow_pickle=False) as f:
        q, frames = f["robot_qpos"], f["frame_indices"]
    if (
        q.shape != (len(frames), 2, 14)
        or not np.isfinite(q).all()
        or not np.array_equal(frames, np.arange(len(frames)))
        or not np.allclose(np.linalg.norm(q[:, :, 3:7], axis=-1), 1, atol=1e-5)
    ):
        raise ValueError("Need complete ordered finite dual-hand reference commands")
    joints = [model.joint(i).name for i in range(model.njnt)]
    bodies = [model.body(i).name for i in range(model.nbody)]
    kin, wrist_joints, fingers, hand_bodies = {}, {}, {}, {}
    for si, side in enumerate(("left", "right")):
        kin[side] = URDFKinematics(inputs / f"robot/urdfs/g1/dex3_{side}.urdf")
        wrist_joints[side] = [
            canonical_id(joints, f"{side}_wrist_{a}") for a in ("x", "y", "z", "rotvec")
        ]
        fingers[side] = [canonical_id(joints, n) for n in manifest["joint_names"][side]]
        if len(fingers[side]) != 7:
            raise ValueError("Need seven explicitly ordered finger joints per hand")
        expected = kin[side].poses(
            q[0, si, :7],
            dict(zip(manifest["joint_names"][side], q[0, si, 7:], strict=True)),
        )
        hand_bodies[side] = {}
        for bid, native_name in enumerate(bodies[1:], start=1):
            matches = [
                name
                for name in expected
                if name.startswith(side + "_")
                and (
                    native_name == name
                    or native_name.endswith("_" + name)
                    or native_name.endswith("/" + name)
                )
            ]
            if len(matches) > 1:
                raise ValueError("Ambiguous native hand body: " + native_name)
            if matches:
                hand_bodies[side][bid] = matches[0]
        for digit, index in (("thumb", 2), ("index", 1), ("middle", 1)):
            required = f"{side}_hand_{digit}_{index}_link"
            if required not in hand_bodies[side].values():
                raise ValueError("Missing native fingertip body: " + required)
        if len(hand_bodies[side]) < 8:
            raise ValueError("Incomplete native robot body binding")
    target_bodies = {}
    for obj in scene.objects:
        if obj.articulations:
            raise ValueError("This audit covers rigid reconstructed scored objects")
        target_bodies.update({b.body_id + 1: obj.name for b in obj.bodies})
    if scene.support is not None:
        target_bodies[scene.support.body_id + 1] = "support"
    position, quaternion = reference.object_body_pos_w(), reference.object_body_quat_w()
    rows, max_fk_error = [], 0.0
    for frame in range(len(frames)):
        state = mujoco.MjData(model)
        for obj in scene.objects:
            bid, rid = obj.root.body_id + 1, obj.root.reference_body_id
            joint = int(model.body_jntadr[bid])
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError("Expected a native free object root")
            a = int(model.jnt_qposadr[joint])
            state.qpos[a : a + 3], state.qpos[a + 3 : a + 7] = (
                position[frame, rid],
                quaternion[frame, rid],
            )
        for si, side in enumerate(("left", "right")):
            target = q[frame, si]
            for col, joint in enumerate(wrist_joints[side][:3]):
                if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_SLIDE:
                    raise ValueError("Expected native wrist slides")
                state.qpos[int(model.jnt_qposadr[joint])] = target[col]
            joint = wrist_joints[side][3]
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_BALL:
                raise ValueError("Expected native wrist quaternion joint")
            a = int(model.jnt_qposadr[joint])
            state.qpos[a : a + 4] = target[[6, 3, 4, 5]]
            for joint, value in zip(fingers[side], target[7:], strict=True):
                if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError("Expected native finger hinges")
                state.qpos[int(model.jnt_qposadr[joint])] = value
        mujoco.mj_forward(model, state)
        for si, side in enumerate(("left", "right")):
            expected = kin[side].poses(
                q[frame, si, :7],
                dict(zip(manifest["joint_names"][side], q[frame, si, 7:], strict=True)),
            )
            for bid, name in hand_bodies[side].items():
                pose = expected[name]
                error = max(
                    float(np.max(np.abs(state.xpos[bid] - pose[:3, 3]))),
                    float(np.max(np.abs(state.xmat[bid].reshape(3, 3) - pose[:3, :3]))),
                )
                if error > 2e-6:
                    raise ValueError(
                        "Native hand FK differs from declared command: " + name
                    )
                max_fk_error = max(max_fk_error, error)
        for obj in scene.objects:
            for body in obj.bodies:
                bid, rid = body.body_id + 1, body.reference_body_id
                np.testing.assert_allclose(
                    state.xpos[bid], position[frame, rid], atol=1e-6, rtol=0
                )
                if (
                    min(
                        np.linalg.norm(state.xquat[bid] - quaternion[frame, rid]),
                        np.linalg.norm(state.xquat[bid] + quaternion[frame, rid]),
                    )
                    > 1e-6
                ):
                    raise ValueError("Native object rotation differs from reference")
        contacts = []
        for c in state.contact[: state.ncon]:
            a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            for side in ("left", "right"):
                if a in hand_bodies[side] and b in target_bodies:
                    hand, target = a, b
                elif b in hand_bodies[side] and a in target_bodies:
                    hand, target = b, a
                else:
                    continue
                contacts.append(
                    dict(
                        hand=side,
                        target=target_bodies[target],
                        native_bodies=[bodies[hand], bodies[target]],
                        signed_gap_m=float(c.dist),
                    )
                )
        rows.append(dict(reference_frame=frame, contacts=contacts))
    return dict(
        schema="v2d.native-reference-hand-contact-audit.v1",
        input_manifest_sha256=input_sha,
        runtime_commit=runtime,
        reference_knots=len(frames),
        reporting_tolerance_m=0.001,
        summary=summarize(rows),
        frames=rows,
        all_native_hand_body_fk_verified=True,
        maximum_native_fk_error=max_fk_error,
        reference_robot_and_object_poses_verified=True,
        frozen_source_closure_verified=True,
        physics_stepped=False,
        contact_forces_assessed=False,
        hand_self_contacts_assessed=False,
        source_targets_modified=False,
        native_mesh_checks_relaxed=False,
        quality_gate_pass_claimed=False,
        physical_task_accepted=False,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wall_seconds=time.monotonic() - started,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.inputs.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        f.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
