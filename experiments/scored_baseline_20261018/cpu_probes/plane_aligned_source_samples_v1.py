"""Two-knot static-plane alignment hypothesis; no full trajectory promotion.

Background-derived normal/height alignment changes the inferred world gauge at
these knots. Source pixels, relative projections, support, geometry and flags
are fixed. Depth bias versus camera drift remains unresolved.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0] = [str(R / 'scripts/track3'), str(R / 'scripts/dex3_training'), str(S / 'compute')]
from measured_rigid_pose_probe_v2 import refine
from foundationpose_track import pose_rejection
from camera_prior import transport
from audit_hand_reference_contacts import canonical_id
from v2d_dex3_flashchord.kinematics import URDFKinematics

OUT = S / 'plane-aligned-source-samples-v1'
OUT.mkdir(exist_ok=True)
ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
F = S / 'fast-stereo-full-object-trajectory-v1'
N = S / 'fast-stereo-full-object-native-validation-v2'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
audit_path = S / 'source-support-failure-window-v1/report.json'
audit = json.loads(audit_path.read_text())
assert audit['source_closure_verified_after']
fit_report = json.loads((F / 'report.json').read_text())
with np.load(F / 'inferred-object-trajectory.npz', allow_pickle=False) as z:
    original = z['original_camera_from_object'].copy()
with np.load(ROOT / 'bundles/e037/episode.npz', allow_pickle=False) as z:
    camera, K = z['camera_c2w'].copy(), z['intrinsics'].copy()
    hand_points = z['hand_joints_world'].copy()
    hand_valid = z['hand_valid'].copy()
    observed = z['hand_directly_observed'].copy()
prior_plane = np.asarray(json.loads((ROOT / 'support/e037/report.json').read_text())['plane_world'])
with np.load(ROOT / 'support/e037/inferred-support-sheet.npz', allow_pickle=False) as z:
    sheet = z['vertices_world_m'].copy()
mesh = trimesh.load(ROOT / 'geometry/e037/wooden_spoon/mesh.glb', force='mesh', process=False)
vertices, faces = np.asarray(mesh.vertices)*1.3, np.asarray(mesh.faces)
model = mujoco.MjModel.from_binary_path(str(N / 'native-model.mjb'))
manifest = json.loads((N / 'frozen/manifest.json').read_text())
with np.load(N / 'frozen/targets.npz', allow_pickle=False) as z:
    commands = z['robot_qpos'].copy()
stage = np.asarray(json.loads((N / 'frozen/command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
bodies = [model.body(i).name for i in range(model.nbody)]
joints = [model.joint(i).name for i in range(model.njnt)]
object_bid, support_bid = [canonical_id(bodies, n) for n in ['wooden_spoon_object', 'support']]
object_adr = int(model.jnt_qposadr[model.body_jntadr[object_bid]])
bindings = {}
for si, side in enumerate(['left', 'right']):
    bindings[side] = dict(si=si, wrist=[canonical_id(joints, f'{side}_wrist_{a}') for a in ['x', 'y', 'z', 'rotvec']],
        fingers=[canonical_id(joints, n) for n in manifest['joint_names'][side]],
        bodies={b for b, n in enumerate(bodies) if n.startswith('dex3_'+side+'_')},
        kin=URDFKinematics(N / f'frozen/robot/urdfs/g1/dex3_{side}.urdf'))
report = dict(schema='v2d.plane-aligned-source-samples.v1', episode=37, frames=[],
    producer_sha256=sha(Path(__file__)), background_audit_sha256=sha(audit_path),
    original_fit_report_sha256=sha(F / 'report.json'), native_model_sha256=sha(N / 'native-model.mjb'),
    changed_factor='Background-derived inferred normal/height camera alignment; then same bounded source fitter.',
    camera_vs_depth_bias_cause_proven=False, tangential_translation_or_yaw_observed=False,
    support_changed=False, geometry_changed=False, source_pixels_changed=False,
    source_confidence_changed=False, missing_observations_filled=False,
    all_frame_motion_or_contact_certificate=False, physics_stepped=False, training_steps=0,
    trajectory_promoted=False, physical_task_accepted=False)
save = lambda: (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
started = time.monotonic()
save()
try:
    for frame in [126, 128]:
        measured = next(r for r in audit['rows'] if r['frame'] == frame)
        local = np.asarray(measured['local_plane']['plane_world'])
        cross = np.cross(local[:3], prior_plane[:3])
        angle = np.arctan2(np.linalg.norm(cross), np.dot(local[:3], prior_plane[:3]))
        rotation = Rotation.from_rotvec(cross/max(np.linalg.norm(cross), 1e-15)*angle).as_matrix()
        H = np.eye(4); H[:3, :3] = rotation; H[:3, 3] = prior_plane[:3]*(local[3]-prior_plane[3])
        mapped = np.r_[H[:3, :3]@local[:3], local[3]-np.dot(prior_plane[:3], H[:3, 3])]
        np.testing.assert_allclose(mapped, prior_plane, atol=1e-10, rtol=0)
        corrected_camera = H@camera[frame]
        plane_camera = np.r_[corrected_camera[:3, :3].T@prior_plane[:3],
                             prior_plane[3]+np.dot(prior_plane[:3], corrected_camera[:3, 3])]
        footprint = np.einsum('ni,ij->nj', sheet-corrected_camera[:3, 3], corrected_camera[:3, :3], optimize=False)
        with np.load(S / f'fast-stereo-cpu-full-e037-v1/{frame:06d}-depth.npz', allow_pickle=False) as z:
            depth, valid = z['depth_m'].copy(), z['left_right_consistent'].copy()
        mask = cv2.imread(str(ROOT / f'source/e037/masks/wooden_spoon/{frame:06d}.png'), 0) > 127
        pose, fit = refine(vertices, faces, original[frame], depth, valid, mask, K, plane_camera, footprint)
        prior = transport(original[frame-1], np.linalg.inv(camera[frame-1]), np.linalg.inv(camera[frame]))
        rejection, support = pose_rejection(pose, depth, valid, (mask*255).astype(np.uint8), K,
                                            np.stack([vertices.min(0), vertices.max(0)]), True, prior)
        stage_pose = np.eye(4); stage_pose[:3, :3] = stage
        T = stage_pose@corrected_camera@pose
        data = mujoco.MjData(model)
        data.qpos[object_adr:object_adr+3] = T[:3, 3]
        data.qpos[object_adr+3:object_adr+7] = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
        aligned_commands = []
        for side, b in bindings.items():
            command = commands[frame, b['si']].copy()
            Rstage = stage@H[:3, :3]@stage.T
            command[:3] = Rstage@command[:3]+stage@H[:3, 3]
            command[3:7] = Rotation.from_matrix(Rstage@Rotation.from_quat(command[3:7]).as_matrix()).as_quat()
            for col, j in enumerate(b['wrist'][:3]):
                data.qpos[int(model.jnt_qposadr[j])] = command[col]
            adr = int(model.jnt_qposadr[b['wrist'][3]])
            data.qpos[adr:adr+4] = command[[6, 3, 4, 5]]
            for j, value in zip(b['fingers'], command[7:], strict=True):
                data.qpos[int(model.jnt_qposadr[j])] = value
            aligned_commands.append(command)
        mujoco.mj_forward(model, data)
        max_fk = 0.
        for side, b in bindings.items():
            command = aligned_commands[b['si']]
            poses = b['kin'].poses(command[:7], dict(zip(manifest['joint_names'][side], command[7:], strict=True)))
            for bid in b['bodies']:
                match = [n for n in poses if bodies[bid] == n or bodies[bid].endswith('_'+n)]
                assert len(match) == 1
                p = poses[match[0]]
                error = max(float(abs(data.xpos[bid]-p[:3, 3]).max()), float(abs(data.xmat[bid].reshape(3, 3)-p[:3, :3]).max()))
                assert error < 2e-6; max_fk = max(max_fk, error)
        penetration = dict(object_support=0., left_object=0., left_support=0., right_object=0., right_support=0.)
        for contact in data.contact[:data.ncon]:
            a, b = [int(model.geom_bodyid[g]) for g in [contact.geom1, contact.geom2]]
            depth_m = max(0., -float(contact.dist))
            if {a, b} == {object_bid, support_bid}:
                penetration['object_support'] = max(penetration['object_support'], depth_m)
            for side, hand in bindings.items():
                for bid, label in [(object_bid, 'object'), (support_bid, 'support')]:
                    if (a in hand['bodies'] and b == bid) or (b in hand['bodies'] and a == bid):
                        penetration[f'{side}_{label}'] = max(penetration[f'{side}_{label}'], depth_m)
        point_projection_delta = 0.
        for si in range(2):
            if not (hand_valid[frame, si] and observed[frame, si]):
                continue
            points = hand_points[frame, si]
            transformed = np.einsum('ni,ji->nj', points, H[:3, :3], optimize=False)+H[:3, 3]
            before = np.einsum('ni,ij->nj', points-camera[frame, :3, 3], camera[frame, :3, :3], optimize=False)
            after = np.einsum('ni,ij->nj', transformed-corrected_camera[:3, 3], corrected_camera[:3, :3], optimize=False)
            finite = np.isfinite(before).all(1) & np.isfinite(after).all(1)
            if finite.any():
                buv = before[finite, :2]/before[finite, 2:3]
                auv = after[finite, :2]/after[finite, 2:3]
                point_projection_delta = max(point_projection_delta, float(abs((auv-buv)*np.array([K[0, 0], K[1, 1]])).max()))
        assert point_projection_delta < 1e-8
        report['frames'].append(dict(frame=frame, inferred_world_alignment=H.tolist(),
            rotation_correction_rad=float(angle), alignment_translation_norm_m=float(np.linalg.norm(H[:3, 3])),
            camera_center_displacement_m=float(np.linalg.norm(corrected_camera[:3, 3]-camera[frame, :3, 3])),
            source_fit=fit, source_depth_gate_rejection=rejection, source_depth_support=support,
            candidate_camera_from_object=pose.tolist(), candidate_world_from_object=(corrected_camera@pose).tolist(),
            maximum_source_hand_projection_change_px=point_projection_delta,
            maximum_hand_fk_error=max_fk, native_penetration_by_pair_m=penetration,
            sample_source_and_native_gate_passed=bool(fit['accepted'] and rejection is None and max(penetration.values()) <= .001)))
        save()
    assert sha(audit_path) == report['background_audit_sha256']
    for name, h in fit_report['source_files_sha256'].items():
        assert sha(S / name) == h
    report['frozen_source_closure_verified_after'] = True
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['wall_seconds'] = time.monotonic()-started
    save()
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2), flush=True)
