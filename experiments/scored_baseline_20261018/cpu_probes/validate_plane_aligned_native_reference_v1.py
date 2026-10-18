"""Fresh native full-reference audit of a separately frozen object diagnostic.

The new geometry and all object poses invalidate the prior hand certificate.
Both unchanged native auditors and the committed 1 mm full-clock gate run;
neither failed fits nor penetrating frames are excluded.  No physics step.
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0] = [str(S / 'cpu-native-runtime-v1/src'),
                str(R / 'scripts/dex3_training'), str(R / 'scripts/track3')]
from check_native_scene import compile_scene
from v2d_dex3_flashchord.kinematics import URDFKinematics
import audit_hand_reference_contacts as hands
import audit_reference_contacts as objects
from reference_contact_gate import assess

P = S / 'native-scaled-spoon-samples-v2/frozen'
FIT = S / 'plane-aligned-pose-temporal-path-v4'
OUT = S / 'plane-aligned-native-reference-v1'
OUT.mkdir(exist_ok=True)
inputs = OUT / 'frozen'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
fit = json.loads((FIT / 'report.json').read_text())
guard = json.loads((FIT / 'guard.json').read_text())
assert fit['all_frames_complete'] and fit['all_source_fits_qualified'] and fit['motion_quality_certified']
assert fit['frozen_source_and_depth_closures_verified_after']
assert guard['worker_exit_code'] == 0
assert sha(FIT / 'inferred-object-trajectory.npz') == fit['candidate_sha256']
manifest = json.loads((P / 'manifest.json').read_text())
for name, h in manifest['files'].items():
    assert sha(P / name) == h
with np.load(P / 'targets.npz', allow_pickle=False) as z:
    original = {k: z[k].copy() for k in z.files}
with np.load(FIT / 'inferred-object-trajectory.npz', allow_pickle=False) as z:
    world = z['candidate_world_from_object'].copy()
    alignment = z['world_alignments'].copy()
    corrected_camera = z['candidate_camera_c2w'].copy()
    plane_qualified = z['source_plane_qualified'].copy()
    np.testing.assert_array_equal(z['frame_indices'], original['frame_indices'])
    np.testing.assert_array_equal(z['timestamps_s'], original['timestamps_s'])
    qualified = z['estimated_source_fit_qualified'].copy()
assert world.shape == (151, 4, 4) and np.isfinite(world).all()
stage = np.asarray(json.loads((P / 'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
positions = np.einsum('ij,tj->ti', stage, world[:, :3, 3], optimize=False)
rotations = np.einsum('ij,tjk->tik', stage, world[:, :3, :3], optimize=False)
shutil.copytree(P, inputs)
new = {k: v.copy() for k, v in original.items()}
new['object_poses'][:, 0, :3] = positions
new['object_poses'][:, 0, 3:7] = Rotation.from_matrix(rotations).as_quat()
# Apply the same inferred world transform to BOTH hand roots and all FK frame
# positions. Finger commands and every original confidence/contact flag stay.
aligned_rotation = np.einsum('ij,tjk,lk->til', stage, alignment[:, :3, :3], stage, optimize=False)
aligned_translation = np.einsum('ij,tj->ti', stage, alignment[:, :3, 3], optimize=False)
def aligned_robot(commands):
    result = commands.copy()
    result[:, :, :3] = np.einsum('tij,tsj->tsi', aligned_rotation, commands[:, :, :3], optimize=False)+aligned_translation[:, None]
    for si in range(2):
        rotations = np.einsum('tij,tjk->tik', aligned_rotation, Rotation.from_quat(commands[:, si, 3:7]).as_matrix(), optimize=False)
        result[:, si, 3:7] = Rotation.from_matrix(rotations).as_quat()
    return result
new['robot_qpos'] = aligned_robot(original['robot_qpos'])
kin = {side:URDFKinematics(inputs / f'robot/urdfs/g1/dex3_{side}.urdf') for side in ['left','right']}
for si, side in enumerate(['left','right']):
    new[f'{side}_frames'] = np.stack([np.stack([poses[name][:3,3] for name in manifest['frame_names'][side]])
        for command in new['robot_qpos'][:,si]
        for poses in [kin[side].poses(command[:7],dict(zip(manifest['joint_names'][side],command[7:],strict=True)))]] )
# Preserve a source-aligned ORIGINAL robot baseline for a fresh retargeter,
# preventing cumulative correction bounds relative to an already repaired hand.
B = S / 'scored-cohort-14-results/policy-e037/frozen'
with np.load(B / 'targets.npz', allow_pickle=False) as z:
    original_robot_baseline = z['robot_qpos'].copy()
source_bundle = S / 'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(source_bundle, allow_pickle=False) as z:
    source = {k:z[k].copy() for k in z.files}
aligned_joints = np.einsum('tij,tsnj->tsni',alignment[:,:3,:3],source['hand_joints_world'],optimize=False)+alignment[:,None,None,:3,3]
reference_path = inputs / 'aligned-source-reference.npz'
np.savez_compressed(reference_path,robot_qpos=aligned_robot(original_robot_baseline),
    camera_c2w=corrected_camera,hand_joints_world=aligned_joints,
    hand_valid=source['hand_valid'],hand_directly_observed=source['hand_directly_observed'],hand_metric_valid=source['hand_metric_valid'],
    source_plane_qualified=plane_qualified,alignment_inferred_only=~plane_qualified,
    frame_indices=source['frame_indices'],timestamps_s=source['timestamps_s'])
manifest['aligned_source_reference'] = dict(path=reference_path.name,sha256=sha(reference_path),
    raw_bundle_sha256=sha(source_bundle),original_robot_manifest_sha256=sha(B/'manifest.json'),
    alignment_fit_report_sha256=fit['alignment_report_sha256'],alignment_npz_sha256=fit['alignment_npz_sha256'],
    source_plane_inferred_only_frames=np.flatnonzero(~plane_qualified).tolist(),source_confidence_upgraded=False,
    source_world_frames_consistently_transformed=True,physical_task_accepted=False)
manifest['files'][reference_path.name] = sha(reference_path)
np.savez_compressed(inputs / 'targets.npz', **new)
manifest['files']['targets.npz'] = sha(inputs / 'targets.npz')
manifest['object_source_reconstruction_diagnostic'] = dict(
    source_fit_report_sha256=sha(FIT / 'report.json'),
    candidate_sha256=fit['candidate_sha256'],
    producer_sha256=sha(Path(__file__)),
    copied_geometry_manifest_sha256=sha(P / 'manifest.json'),
    source_observations_changed=False, source_confidence_changed=False,
    object_reference_estimates_changed=True, original_support_changed=False,
    prior_hand_contact_certificate_reused=False, full_trajectory_qualified=False,
    physical_task_accepted=False, training_approved=False)
(inputs / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
report = dict(schema='v2d.plane-aligned-full-reference-native.v1',
    episode=37, producer_sha256=sha(Path(__file__)),
    input_manifest_sha256=sha(inputs / 'manifest.json'),
    source_fit_report_sha256=sha(FIT / 'report.json'),
    all_reference_knots_included=True, source_fit_qualified_frames=int(qualified.sum()),
    source_fit_failed_frames=np.flatnonzero(~qualified).tolist(),
    source_confidence_modified=False, original_support_changed=False,
    prior_hand_certificate_reused=False, physics_stepped=False, training_steps=0,
    trajectory_promoted=False, training_approved=False, physical_task_accepted=False)
save = lambda: (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
started = time.monotonic()
save()
try:
    compiled = compile_scene(inputs)
    # Share only the exact fresh compiler product, at the exact frozen path.
    def exact_compiled(path):
        if path.resolve() != inputs.resolve():
            raise ValueError('Unexpected input path for fresh native audit')
        return compiled
    hands.compile_scene = exact_compiled
    objects.compile_scene = exact_compiled
    hand = hands.audit(inputs.resolve())
    obj = objects.audit(inputs.resolve())
    for name, value in [('hand-reference-contacts.json', hand),
                        ('object-reference-contacts.json', obj)]:
        (OUT / name).write_text(json.dumps(value, indent=2)+'\n')
    gate = assess(inputs / 'manifest.json', manifest, obj, hand)
    (OUT / 'complete-reference-contact-gate.json').write_text(json.dumps(gate, indent=2)+'\n')
    report['native_contact_gate'] = gate
    report['native_hand_body_fk_verified'] = hand['all_native_hand_body_fk_verified']
    report['object_support_native_stage_passed'] = gate['object_knots_over_tolerance'] == 0
    report['complete_reference_native_gate_passed'] = gate['native_penetration_gate_passed']
    report['all_source_fits_qualified'] = bool(qualified.all())
    report['motion_quality_certified'] = False
    report['object_source_motion_stage_passed'] = True
    report['hand_retarget_motion_revalidation_required'] = True
    report['remaining_motion_review'] = fit['motion']
    report['unchanged_auditor_sha256'] = {name: sha(R / 'scripts/dex3_training' / name)
        for name in ['audit_hand_reference_contacts.py', 'audit_reference_contacts.py', 'check_native_scene.py']}
    report['unchanged_contact_gate_sha256'] = sha(R / 'scripts/track3/reference_contact_gate.py')
    import mujoco
    mujoco.mj_saveModel(compiled[2].mj_model, str(OUT / 'native-model.mjb'), None)
    # Geometry was already matched to the unchanged upstream import.  Verify
    # that this fresh full reference still imports the same native geometry.
    control = mujoco.MjModel.from_binary_path(str(S / 'native-scaled-spoon-samples-v2/native-model.mjb'))
    current = compiled[2].mj_model
    checks = []
    for key in ['geom_type', 'geom_bodyid', 'geom_dataid', 'geom_pos', 'geom_quat',
                'geom_size', 'mesh_vert', 'mesh_pos', 'mesh_quat']:
        a, b = np.asarray(getattr(control, key)), np.asarray(getattr(current, key))
        assert a.shape == b.shape
        error = float(abs(a-b).max(initial=0))
        checks.append(dict(array=key, maximum_absolute_error=error, passed=error < 1e-10))
    report['fresh_geometry_checks'] = checks
    assert all(row['passed'] for row in checks)
    for key in original.keys() - {'object_poses','robot_qpos','left_frames','right_frames'}:
        np.testing.assert_array_equal(new[key], original[key])
    for name, h in manifest['files'].items():
        assert sha(inputs / name) == h
    old = json.loads((P / 'manifest.json').read_text())
    for name, h in old['files'].items():
        assert sha(P / name) == h
    for name, h in fit['source_files_sha256'].items():
        assert sha(S / name) == h
    report['all_frozen_closures_and_unchanged_non_pose_targets_verified_after'] = True
    report['source_plane_inferred_only_frames'] = np.flatnonzero(~plane_qualified).tolist()
    report['both_hand_roots_and_FK_frames_aligned_consistently'] = True
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['wall_seconds'] = time.monotonic()-started
    save()
    print(json.dumps({k: v for k, v in report.items()
                     if k not in ['remaining_motion_review', 'fresh_geometry_checks']}, indent=2), flush=True)
