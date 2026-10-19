"""Audit both hands' native self-contacts from a hash-bound cached MJB.

No scene compilation, physics steps, force solve, command changes or reuse of
the old hand/object audit. Native model filters remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from audit_hand_reference_contacts import canonical_id
from v2d_dex3_flashchord.self_contact import summarize_self_contacts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(inputs, native_model, native_model_sha256):
    import mujoco
    from v2d_dex3_flashchord.kinematics import URDFKinematics

    start = time.monotonic()
    if sha(native_model) != native_model_sha256:
        raise ValueError('Cached native model hash differs')
    manifest = json.loads((inputs / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (inputs / name).resolve()
        if not path.is_relative_to(inputs.resolve()) or sha(path) != expected:
            raise ValueError('Frozen source closure differs: ' + name)
    with np.load(inputs / 'targets.npz', allow_pickle=False) as z:
        q, frames, objects = z['robot_qpos'].copy(), z['frame_indices'].copy(), z['object_poses'].copy()
    if (q.shape != (len(frames), 2, 14) or not len(frames)
            or not np.array_equal(frames, np.arange(len(frames)))
            or not np.isfinite(q).all()
            or not np.allclose(np.linalg.norm(q[:, :, 3:7], axis=-1), 1, atol=1e-5, rtol=0)
            or objects.shape != (len(frames), len(manifest['objects']), 7)
            or not np.isfinite(objects).all()
            or not np.allclose(np.linalg.norm(objects[:, :, 3:], axis=-1), 1, atol=1e-5, rtol=0)):
        raise ValueError('Need complete finite ordered commands and rigid object poses')
    model = mujoco.MjModel.from_binary_path(str(native_model))
    data = mujoco.MjData(model)
    joints = [model.joint(i).name for i in range(model.njnt)]
    bodies = [model.body(i).name for i in range(model.nbody)]
    registries, body_ids, wrists, fingers, kin = {}, {}, {}, {}, {}
    for side in ('left', 'right'):
        si = 0 if side == 'left' else 1
        kin[side] = URDFKinematics(inputs / f'robot/urdfs/g1/dex3_{side}.urdf')
        expected = kin[side].poses(q[0, si, :7], dict(zip(manifest['joint_names'][side], q[0, si, 7:], strict=True)))
        body_ids[side] = {}
        for bid, native_name in enumerate(bodies):
            matches = [name for name in expected if name.startswith(side + '_') and (
                native_name == name or native_name.endswith('_' + name) or native_name.endswith('/' + name))]
            if len(matches) > 1:
                raise ValueError('Ambiguous native hand body binding')
            if matches:
                body_ids[side][bid] = matches[0]
        if len(body_ids[side]) < 8:
            raise ValueError('Incomplete native hand body binding')
        registries[side] = [bodies[i] for i in body_ids[side]]
        wrists[side] = [canonical_id(joints, f'{side}_wrist_{a}') for a in ('x', 'y', 'z', 'rotvec')]
        fingers[side] = [canonical_id(joints, n) for n in manifest['joint_names'][side]]
        if len(fingers[side]) != 7:
            raise ValueError('Need seven explicitly ordered native finger joints')
        if (not np.all(model.jnt_type[wrists[side][:3]] == mujoco.mjtJoint.mjJNT_SLIDE)
                or model.jnt_type[wrists[side][3]] != mujoco.mjtJoint.mjJNT_BALL
                or not np.all(model.jnt_type[fingers[side]] == mujoco.mjtJoint.mjJNT_HINGE)):
            raise ValueError('Unexpected cached native wrist/finger joint types')
    object_joints = []
    for obj in manifest['objects']:
        bid = canonical_id(bodies, obj['name'] + '_object')
        joint = int(model.body_jntadr[bid])
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Expected rigid native object root')
        object_joints.append(joint)
    rows, max_fk_error = [], 0.
    for frame in range(len(frames)):
        data.qpos[:] = model.qpos0; data.qvel[:] = 0
        for oi, joint in enumerate(object_joints):
            address = int(model.jnt_qposadr[joint]); target = objects[frame, oi]
            data.qpos[address:address + 3] = target[:3]
            data.qpos[address + 3:address + 7] = target[[6, 3, 4, 5]]
        for si, side in enumerate(('left', 'right')):
            target = q[frame, si]; w = wrists[side]
            data.qpos[model.jnt_qposadr[w[:3]]] = target[:3]
            address = int(model.jnt_qposadr[w[3]])
            data.qpos[address:address + 4] = target[[6, 3, 4, 5]]
            data.qpos[model.jnt_qposadr[fingers[side]]] = target[7:]
        mujoco.mj_fwdPosition(model, data)
        for si, side in enumerate(('left', 'right')):
            poses = kin[side].poses(q[frame, si, :7], dict(zip(manifest['joint_names'][side], q[frame, si, 7:], strict=True)))
            for bid, name in body_ids[side].items():
                pose = poses[name]
                error = max(float(abs(data.xpos[bid] - pose[:3, 3]).max()),
                    float(abs(data.xmat[bid].reshape(3, 3) - pose[:3, :3]).max()))
                if error > 2e-6:
                    raise ValueError('Native hand FK differs from declared command: ' + name)
                max_fk_error = max(max_fk_error, error)
        contacts = []
        for c in data.contact[:data.ncon]:
            a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            for side in ('left', 'right'):
                if a in body_ids[side] and b in body_ids[side]:
                    contacts.append(dict(hand=side, native_bodies=[bodies[a], bodies[b]],
                        native_geom_ids=[int(c.geom1), int(c.geom2)], signed_gap_m=float(c.dist)))
        rows.append(dict(reference_frame=frame, contacts=contacts))
    summary = summarize_self_contacts(rows, registries, reference_frame_count=len(frames), tolerance_m=0.001)
    return dict(schema='v2d.cached-native-hand-self-contact-audit.v1',
        input_manifest_sha256=sha(inputs / 'manifest.json'), targets_sha256=sha(inputs / 'targets.npz'),
        native_model_sha256=native_model_sha256, MuJoCo_version=mujoco.__version__,
        producer_sha256=sha(Path(__file__)), hand_bodies=registries, frames=rows, **summary,
        all_native_hand_body_FK_verified=True, maximum_native_FK_error=max_fk_error,
        frozen_source_closure_verified=True, native_contact_filters_unchanged=True,
        prior_hand_object_support_audit_not_repeated=True, native_scene_not_recompiled=True,
        physics_stepped=False, contact_forces_assessed=False, source_targets_modified=False,
        hand_object_support_swept_or_grasp_not_assessed=True,
        training_approved=False, physical_task_accepted=False, wall_seconds=time.monotonic()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--native-model', type=Path, required=True)
    parser.add_argument('--native-model-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.inputs, args.native_model, args.native_model_sha256)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('reference_knots', 'summary_by_hand', 'summary_by_pair', 'hand_self_contact_gate_passed')}, indent=2))
