"""Choose and independently source-check a motion-bounded complete pose path."""
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0] = [str(R / 'scripts/track3')]
from temporal_pose_path import MotionLimits, motion_summary, select_path
from select_pose_hypothesis import observed_fit
from foundationpose_track import pose_rejection
from camera_prior import transport

P = S / 'plane-aligned-pose-pools-v6'
F = S / 'fast-stereo-full-object-trajectory-v1'
D = S / 'fast-stereo-cpu-full-e037-v1'
ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
OUT = S / 'plane-aligned-pose-temporal-path-v3'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pool = json.loads((P / 'report.json').read_text())
assert pool['all_frames_complete'] and pool['frozen_input_closures_verified_after']
assert json.loads((P / 'guard.json').read_text())['worker_exit_code'] == 0
old = json.loads((F / 'report.json').read_text())
assert sha(F / 'report.json') == pool['original_fit_report_sha256']
with np.load(F / 'inferred-object-trajectory.npz', allow_pickle=False) as z:
    original = z['original_camera_from_object'].copy()
    axis, timestamps = z['frame_indices'].copy(), z['timestamps_s'].copy()
    flags = z['original_object_observed_flags'].copy()
with np.load(ROOT / 'bundles/e037/episode.npz', allow_pickle=False) as z:
    original_camera, K = z['camera_c2w'].copy(), z['intrinsics'].copy()
alignment_path = S / 'smooth-plane-camera-alignment-v2/inferred-world-alignment.npz'
assert sha(alignment_path) == pool['alignment_npz_sha256']
with np.load(alignment_path, allow_pickle=False) as z:
    camera = z['candidate_camera_c2w'].copy()
    alignments = z['world_alignments'].copy()
    plane_qualified = z['source_plane_qualified'].copy()
report = dict(schema='v2d.plane-aligned-pose-temporal-path.v3', episode=37,
    producer_sha256=sha(Path(__file__)), selector_sha256=sha(R / 'scripts/track3/temporal_pose_path.py'),
    candidate_pool_report_sha256=sha(P / 'report.json'), original_fit_report_sha256=sha(F / 'report.json'),
    alignment_report_sha256=pool['alignment_report_sha256'], alignment_npz_sha256=pool['alignment_npz_sha256'],
    source_camera_estimates_changed=True, source_plane_inferred_only_frames=np.flatnonzero(~plane_qualified).tolist(),
    source_files_sha256=pool['source_files_sha256'], depth_report_sha256=pool['depth_report_sha256'],
    source_confidence_modified=False, observations_filled=False,
    estimated_depth_is_not_ground_truth=True,
    retained_conflicting_evidence=old['retained_conflicting_evidence'],
    original_support_changed=False, acceptance_gates_relaxed=False,
    motion_limits_chosen_before_selection='All four maxima of the untouched original trajectory, at exact source timestamps; p95 speed must also not regress.',
    all_frames_complete=False, all_source_fits_qualified=False, motion_quality_certified=False,
    full_native_contact_gate_completed=False, physics_stepped=False, training_steps=0,
    trajectory_promoted=False, physical_task_accepted=False)
save = lambda: (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
started = time.monotonic()
save()
try:
    original_world = np.einsum('tij,tjk->tik', original_camera, original, optimize=False)
    before_motion = motion_summary(original_world, timestamps)
    limits = MotionLimits(before_motion['maximum_linear_speed_m_s'], before_motion['maximum_angular_speed_rad_s'],
                          before_motion['maximum_linear_acceleration_m_s2'], before_motion['maximum_angular_acceleration_rad_s2'])
    world_pools, source_costs = [], []
    for frame in pool['frames']:
        index = frame['frame']
        poses = np.asarray([p['camera_from_object'] for p in frame['retained_candidates']])
        if len(poses):
            poses = np.einsum('ij,tjk->tik', camera[index], poses, optimize=False)
            qualities = np.asarray([p['source_fit']['quality'] for p in frame['retained_candidates']])
            costs = 1.-qualities/qualities.max()
        else:
            costs = []
        world_pools.append(poses)
        source_costs.append(costs)
    selected, temporal = select_path(world_pools, source_costs, timestamps, limits)
    report['temporal_selection'] = temporal
    report['motion'] = dict(original=before_motion, candidate=temporal.get('selected_motion'))
    if selected is None:
        report['rejected_without_fabricating_pose'] = True
        report['selection_complete'] = True
        save()
        print(json.dumps(temporal, indent=2), flush=True)
        raise SystemExit(0)
    candidate = np.stack([np.asarray(pool['frames'][f]['retained_candidates'][i]['camera_from_object'])
                          for f, i in enumerate(selected)])
    world = np.einsum('tij,tjk->tik', camera, candidate, optimize=False)
    report['selected_pool_indices'] = selected
    after_motion = motion_summary(world, timestamps)
    report['motion']['candidate'] = after_motion
    report['motion_quality_certified'] = bool(
        all(after_motion[k] <= before_motion[k]+1e-9 for k in before_motion))
    mesh = trimesh.load(ROOT / 'geometry/e037/wooden_spoon/mesh.glb', force='mesh', process=False)
    vertices, faces = np.asarray(mesh.vertices)*1.3, np.asarray(mesh.faces)
    tree = cKDTree(vertices)
    bounds = np.stack([vertices.min(0), vertices.max(0)])
    inference = json.loads((D / 'depth-report.json').read_text())
    assert sha(D / 'depth-report.json') == report['depth_report_sha256']
    rows = []
    for frame in axis:
        frame = int(frame)
        assert time.monotonic()-started < 150, 'Bounded independent source validation'
        path = D / f'{frame:06d}-depth.npz'
        assert sha(path) == inference['complete'][frame]['sha256']
        with np.load(path, allow_pickle=False) as z:
            depth, valid = z['depth_m'].copy(), z['left_right_consistent'].copy()
        mask = cv2.imread(str(ROOT / f'source/e037/masks/wooden_spoon/{frame:06d}.png'), 0) > 127
        measured = observed_fit(vertices, faces, tree, candidate[frame], depth, valid, mask, K)
        before = old['rows'][frame]['fit']['before']
        prior = transport(candidate[frame-1], np.linalg.inv(camera[frame-1]), np.linalg.inv(camera[frame])) if frame else None
        rejection, _ = pose_rejection(candidate[frame], depth, valid, (mask*255).astype(np.uint8), K, bounds, True, prior)
        passed = bool(flags[frame] and rejection is None and measured['quality'] > before['quality']
                      and measured['silhouette_iou'] >= before['silhouette_iou']
                      and measured['median_measured_vertex_distance_m'] < before['median_measured_vertex_distance_m'])
        stored = pool['frames'][frame]['retained_candidates'][selected[frame]]['source_fit']
        for key in ['quality', 'silhouette_iou', 'median_measured_vertex_distance_m']:
            assert abs(measured[key]-stored[key]) < 1e-10
        rows.append(dict(frame=frame, source_fit_before=before, source_fit_after=measured,
                         original_object_observed_flag=bool(flags[frame]),
                         corrected_prior_gate_rejection=rejection, source_fit_qualified=passed))
    report['rows'] = rows
    report['source_fit_failed_frames'] = [r['frame'] for r in rows if not r['source_fit_qualified']]
    report['all_source_fits_qualified'] = not report['source_fit_failed_frames']
    np.savez_compressed(OUT / 'inferred-object-trajectory.npz',
        candidate_camera_from_object=candidate, candidate_world_from_object=world,
        original_camera_from_object=original, frame_indices=axis, timestamps_s=timestamps,
        original_object_observed_flags=flags,
        estimated_source_fit_qualified=np.asarray([r['source_fit_qualified'] for r in rows]),
        candidate_camera_c2w=camera, world_alignments=alignments, source_plane_qualified=plane_qualified)
    report['candidate_sha256'] = sha(OUT / 'inferred-object-trajectory.npz')
    for name, h in report['source_files_sha256'].items():
        assert sha(S / name) == h
    assert sha(P / 'report.json') == report['candidate_pool_report_sha256']
    assert sha(alignment_path) == report['alignment_npz_sha256']
    report['frozen_source_and_depth_closures_verified_after'] = True
    report['all_frames_complete'] = True
except SystemExit:
    raise
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['wall_seconds'] = time.monotonic()-started
    save()
