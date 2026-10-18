"""Complete e037 source-fit diagnostic, with every failed knot retained.

This consumes the sealed CPU depth pass and reuses the seven prior fits.  It
does not fill source gaps, change confidence, promote a bundle, or train a
policy.  Corrected consecutive poses are checked independently of the original
pose priors used by the seven-knot experiment.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0] = [str(R / 'scripts/track3'), str(S / 'compute')]
from measured_rigid_pose_probe_v2 import refine
from measured_pose_refinement import finite_support_overlap
from foundationpose_track import pose_rejection
from camera_prior import transport

ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
DEPTH = S / 'fast-stereo-cpu-full-e037-v1'
OUT = S / 'fast-stereo-full-object-trajectory-v1'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
depth_report_path = DEPTH / 'depth-report.json'
inference = json.loads(depth_report_path.read_text())
guard = json.loads((DEPTH / 'guard.json').read_text())
verification_path = DEPTH / 'verification-report.json'
verification = json.loads(verification_path.read_text())
assert verification['passed'] and verification['source_frames'] == 151
assert verification['depth_report_sha256'] == sha(depth_report_path)
assert inference['all_frames_complete']
assert inference['frozen_source_closure_verified_after']
assert guard['worker_exit_code'] == 0 and 'completed_utc' in guard
assert [r['frame'] for r in inference['complete']] == list(range(151))
for row in inference['complete']:
    assert sha(DEPTH / row['path']) == row['sha256']
for name, h in inference['source_files_sha256'].items():
    assert sha(S / name) == h

bundle = ROOT / 'bundles/e037/episode.npz'
with np.load(bundle, allow_pickle=False) as z:
    camera = z['camera_c2w'].copy()
    K = z['intrinsics'].copy()
    axis = z['frame_indices'].copy()
    timestamps = z['timestamps_s'].copy()
    object_observed = z['object_0_valid'].copy()
np.testing.assert_array_equal(axis, np.arange(151))
np.testing.assert_allclose(np.diff(timestamps), .05, atol=1e-8, rtol=0)
support_report = ROOT / 'support/e037/report.json'
support_sheet = ROOT / 'support/e037/inferred-support-sheet.npz'
world_plane = np.asarray(json.loads(support_report.read_text())['plane_world'])
with np.load(support_sheet, allow_pickle=False) as z:
    sheet = z['vertices_world_m'].copy()
mesh_path = ROOT / 'geometry/e037/wooden_spoon/mesh.glb'
mesh = trimesh.load(mesh_path, force='mesh', process=False)
vertices = np.asarray(mesh.vertices) * 1.3
faces = np.asarray(mesh.faces)
bounds = np.stack([vertices.min(0), vertices.max(0)])
poses = ROOT / 'source/e037/poses/e037/ego_object_poses/e037/wooden_spoon/object_to_cam/ego_cam_c'
masks = ROOT / 'source/e037/masks/wooden_spoon'
source_paths = [bundle, support_report, support_sheet, mesh_path]
source_paths += [p for frame in axis for p in [poses / f'{frame:06d}.json', masks / f'{frame:06d}.png']]
source_hashes = {str(p.relative_to(S)): sha(p) for p in source_paths}

def load_pose(frame):
    j = json.loads((poses / f'{frame:06d}.json').read_text())
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(np.asarray(j['rotation'])[[1, 2, 3, 0]]).as_matrix()
    pose[:3, 3] = j['translation']
    return pose

original = np.stack([load_pose(frame) for frame in axis])
candidate = original.copy()
fit_accepted = np.zeros(len(axis), dtype=bool)
source_qualified = np.zeros(len(axis), dtype=bool)
seven_path = S / 'fast-stereo-scaled-seven-pose-probe-v1/report.json'
seven = json.loads(seven_path.read_text())
reused = {r['frame']: r for r in seven['rows']}
assert sorted(reused) == [0, 25, 50, 75, 100, 125, 150]
report = dict(
    schema='v2d.fast-stereo-full-object-source-diagnostic.v1', episode=37,
    producer_sha256=sha(Path(__file__)),
    refiner_sha256=sha(S / 'compute/measured_rigid_pose_probe_v2.py'),
    depth_report_sha256=sha(depth_report_path),
    depth_closure_verification_sha256=sha(verification_path),
    reused_seven_fit_report_sha256=sha(seven_path),
    source_files_sha256=source_hashes, source_frames=len(axis), rows=[],
    geometry_uniform_scale=1.3, estimated_depth_is_not_ground_truth=True,
    original_source_files_modified=False, source_confidence_modified=False,
    original_support_changed=False, observations_filled=False,
    acceptance_logic_relaxed=False, all_frames_complete=False,
    corrected_temporal_checks_complete=False, motion_quality_certified=False,
    original_sgbm_comparison_available_only_at_seven_samples=True,
    retained_conflicting_evidence=dict(
        textured_object_census_cost='New depth worse at six of seven samples, tied at one.',
        source_fit='New scaled f125 fit improves on learned depth but regresses on retained SGBM.',
        depth_accuracy_ground_truth_available=False),
    physics_stepped=False, training_steps=0, full_native_contact_gate_completed=False,
    trajectory_promoted=False, physical_task_accepted=False)
save = lambda: (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
started = time.monotonic()
save()
try:
    for frame in axis:
        frame = int(frame)
        assert time.monotonic() - started < 2100, 'Bounded full source-fit CPU job'
        path = DEPTH / f'{frame:06d}-depth.npz'
        with np.load(path, allow_pickle=False) as z:
            depth, valid = z['depth_m'].copy(), z['left_right_consistent'].copy()
            assert int(z['frame_index']) == frame
            np.testing.assert_array_equal(z['timestamp_s'], timestamps[frame])
        mask = cv2.imread(str(masks / f'{frame:06d}.png'), 0)
        assert mask is not None and mask.shape == depth.shape
        mask = mask > 127
        C = camera[frame]
        plane = np.r_[np.einsum('ji,j->i', C[:3, :3], world_plane[:3], optimize=False),
                      world_plane[3] + np.dot(world_plane[:3], C[:3, 3])]
        footprint = np.einsum('ni,ij->nj', sheet - C[:3, 3], C[:3, :3], optimize=False)
        tick = time.monotonic()
        if frame in reused:
            fit = reused[frame]['fit']
            pose = np.asarray(reused[frame]['candidate_camera_from_object'])
        else:
            pose, fit = refine(vertices, faces, original[frame], depth, valid, mask,
                               K, plane, footprint)
        candidate[frame] = pose
        fit_accepted[frame] = bool(fit['accepted'])
        original_prior = (transport(original[frame-1], np.linalg.inv(camera[frame-1]),
                                    np.linalg.inv(C)) if frame else None)
        corrected_prior = (transport(candidate[frame-1], np.linalg.inv(camera[frame-1]),
                                     np.linalg.inv(C)) if frame else None)
        rejection, support = pose_rejection(pose, depth, valid, (mask*255).astype(np.uint8),
                                            K, bounds, True, original_prior)
        corrected_rejection, _ = pose_rejection(pose, depth, valid, (mask*255).astype(np.uint8),
                                                K, bounds, True, corrected_prior)
        source_qualified[frame] = bool(fit_accepted[frame] and rejection is None
                                      and corrected_rejection is None and object_observed[frame])
        report['rows'].append(dict(
            frame=frame, reused_exact_fit=frame in reused,
            processing_seconds=time.monotonic()-tick, fit=fit,
            original_object_observed_flag=bool(object_observed[frame]),
            learned_foreground_valid_pixels=int((valid & mask).sum()),
            learned_depth_original_prior_gate_rejection=rejection,
            learned_depth_corrected_prior_gate_rejection=corrected_rejection,
            learned_depth_support=support,
            finite_overlap_after=finite_support_overlap(vertices, faces, pose, plane, footprint),
            candidate_camera_from_object=pose.tolist(),
            source_fit_qualified=bool(source_qualified[frame])))
        save()
        print('Source fit', frame, 'accepted', bool(fit_accepted[frame]),
              'qualified', bool(source_qualified[frame]),
              'reused', frame in reused, 'seconds', time.monotonic()-tick, flush=True)
    # These measure the actual new consecutive sequence in the fixed world frame.
    # The old 15 cm / 40 degree jump rejection is applied above; smoothness is
    # reported separately and cannot silently qualify a noisy sequence.
    world = np.einsum('tij,tjk->tik', camera, candidate, optimize=False)
    original_world = np.einsum('tij,tjk->tik', camera, original, optimize=False)
    def motion(poses_world):
        dt = np.diff(timestamps)
        rotations = Rotation.from_matrix(poses_world[:, :3, :3])
        angular = (rotations[:-1].inv() * rotations[1:]).magnitude()
        delta = np.diff(poses_world[:, :3, 3], axis=0)
        acceleration = np.diff(delta / dt[:, None], axis=0) / ((dt[1:] + dt[:-1])[:, None] / 2)
        return dict(maximum_step_translation_m=float(np.linalg.norm(delta, axis=1).max()),
                    maximum_step_rotation_rad=float(angular.max()),
                    maximum_linear_speed_m_s=float(np.linalg.norm(delta / dt[:, None], axis=1).max()),
                    maximum_angular_speed_rad_s=float((angular / dt).max()),
                    maximum_linear_acceleration_m_s2=float(np.linalg.norm(acceleration, axis=1).max()),
                    p95_linear_speed_m_s=float(np.quantile(np.linalg.norm(delta / dt[:, None], axis=1), .95)),
                    p95_angular_speed_rad_s=float(np.quantile(angular / dt, .95)))
    correction = world[:, :3, 3] - original_world[:, :3, 3]
    report['motion'] = dict(original=motion(original_world), candidate=motion(world),
        maximum_added_translation_m=float(np.linalg.norm(correction, axis=1).max()),
        maximum_added_adjacent_translation_m=float(np.linalg.norm(np.diff(correction, axis=0), axis=1).max()),
        maximum_added_second_translation_m=float(np.linalg.norm(np.diff(correction, n=2, axis=0), axis=1).max()))
    report['corrected_temporal_checks_complete'] = True
    report['all_source_fits_qualified'] = bool(source_qualified.all())
    report['source_fit_failed_frames'] = np.flatnonzero(~source_qualified).tolist()
    report['corrected_prior_jump_failed_frames'] = [r['frame'] for r in report['rows']
        if r['learned_depth_corrected_prior_gate_rejection'] is not None]
    np.savez_compressed(OUT / 'inferred-object-trajectory.npz',
        candidate_camera_from_object=candidate, candidate_world_from_object=world,
        original_camera_from_object=original, frame_indices=axis, timestamps_s=timestamps,
        original_object_observed_flags=object_observed,
        estimated_source_fit_qualified=source_qualified, estimated_fit_accepted=fit_accepted)
    report['candidate_sha256'] = sha(OUT / 'inferred-object-trajectory.npz')
    for name, h in source_hashes.items():
        assert sha(S / name) == h
    for row in inference['complete']:
        assert sha(DEPTH / row['path']) == row['sha256']
    assert sha(depth_report_path) == report['depth_report_sha256']
    assert sha(seven_path) == report['reused_seven_fit_report_sha256']
    report['frozen_source_and_depth_closures_verified_after'] = True
    report['all_frames_complete'] = True
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['wall_seconds'] = time.monotonic() - started
    save()
