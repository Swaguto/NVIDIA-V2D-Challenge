"""Finite source-pixel neighbourhood search, retaining all rejected controls.

Only the eleven failed knots receive new pose evaluations. Converged previous
attempts provide seeds; none of their failed outcomes is re-labelled accepted.
Every new candidate must pass the same source/bounds/finite-support/depth gates.
This makes a pool for a later temporal solve, not a qualified reference.
"""
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0] = [str(R / 'scripts/track3')]
from select_pose_hypothesis import observed_fit
from foundationpose_track import pose_rejection
from camera_prior import transport

F = S / 'fast-stereo-full-object-trajectory-v1'
D = S / 'fast-stereo-cpu-full-e037-v1'
ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
OUT = S / 'source-pose-pool-repair-v1'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
previous = json.loads((F / 'report.json').read_text())
assert previous['all_frames_complete'] and previous['frozen_source_and_depth_closures_verified_after']
assert sha(F / 'inferred-object-trajectory.npz') == previous['candidate_sha256']
for name, h in previous['source_files_sha256'].items():
    assert sha(S / name) == h
depth_report = json.loads((D / 'depth-report.json').read_text())
assert sha(D / 'depth-report.json') == previous['depth_report_sha256']
with np.load(F / 'inferred-object-trajectory.npz', allow_pickle=False) as z:
    original = z['original_camera_from_object'].copy()
    axis = z['frame_indices'].copy()
    timestamps = z['timestamps_s'].copy()
    flags = z['original_object_observed_flags'].copy()
with np.load(ROOT / 'bundles/e037/episode.npz', allow_pickle=False) as z:
    camera, K = z['camera_c2w'].copy(), z['intrinsics'].copy()
world_plane = np.asarray(json.loads((ROOT / 'support/e037/report.json').read_text())['plane_world'])
with np.load(ROOT / 'support/e037/inferred-support-sheet.npz', allow_pickle=False) as z:
    sheet = z['vertices_world_m'].copy()
mesh = trimesh.load(ROOT / 'geometry/e037/wooden_spoon/mesh.glb', force='mesh', process=False)
vertices, faces = np.asarray(mesh.vertices)*1.3, np.asarray(mesh.faces)
tree = cKDTree(vertices)
bounds = np.stack([vertices.min(0), vertices.max(0)])
report = dict(schema='v2d.source-pose-pool-repair.v1', episode=37,
    producer_sha256=sha(Path(__file__)), original_fit_report_sha256=sha(F / 'report.json'),
    source_files_sha256=previous['source_files_sha256'], depth_report_sha256=sha(D / 'depth-report.json'),
    failed_control_frames=previous['source_fit_failed_frames'], frames=[],
    search=dict(alpha=[.25, .5, .75, 1.], pixel_offsets=[0., -1., 1., -2., 2.],
                depth_offsets_m=[0., -.001, .001, -.003, .003], retained_pool_per_frame=12),
    source_confidence_modified=False, original_support_changed=False,
    source_observations_filled=False, acceptance_gates_relaxed=False,
    failed_optimizer_runs_promoted=False, all_frames_complete=False,
    full_native_contact_gate_completed=False, motion_quality_certified=False,
    physics_stepped=False, training_steps=0, trajectory_promoted=False, physical_task_accepted=False)
save = lambda: (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
started = time.monotonic()
save()

def deadline():
    if time.monotonic()-started > 840:
        raise TimeoutError('Fourteen-minute bounded source-pose neighbourhood search')

try:
    for row in previous['rows']:
        deadline()
        frame = row['frame']
        p = D / f'{frame:06d}-depth.npz'
        assert sha(p) == depth_report['complete'][frame]['sha256']
        with np.load(p, allow_pickle=False) as z:
            depth, valid = z['depth_m'].copy(), z['left_right_consistent'].copy()
        mask = cv2.imread(str(ROOT / f'source/e037/masks/wooden_spoon/{frame:06d}.png'), 0) > 127
        C = camera[frame]
        normal = np.einsum('ji,j->i', C[:3, :3], world_plane[:3], optimize=False)
        plane_offset = world_plane[3] + np.dot(world_plane[:3], C[:3, 3])
        footprint = np.einsum('ni,ij->nj', sheet-C[:3, 3], C[:3, :3], optimize=False)
        a = np.eye(3)[np.argmin(abs(normal))]
        u = np.cross(normal, a); u /= np.linalg.norm(u)
        basis = np.stack([u, np.cross(normal, u)], axis=1)
        hull = ConvexHull(np.einsum('ni,ij->nj', footprint, basis, optimize=False))
        prior = transport(original[frame-1], np.linalg.inv(camera[frame-1]), np.linalg.inv(C)) if frame else None
        before = row['fit']['before']
        pool, seen = [], set()
        evaluated = 0
        counters = dict(source_failed=0, bounds_failed=0, tracker_failed=0)
        tick = time.monotonic()

        def evaluate(pose, *, support_applied, provenance, recorded_score=None):
            key = np.asarray(pose).round(11).tobytes()
            if key in seen:
                return
            seen.add(key)
            if (np.linalg.norm(pose[:3, 3]-original[frame, :3, 3]) > .06+1e-7 or
                (Rotation.from_matrix(pose[:3, :3])*Rotation.from_matrix(original[frame, :3, :3]).inv()).magnitude() > np.deg2rad(20)+1e-7):
                counters['bounds_failed'] += 1
                return
            transformed = np.einsum('ni,ji->nj', vertices, pose[:3, :3], optimize=False)+pose[:3, 3]
            if support_applied:
                height = float((np.einsum('ni,i->n', transformed, normal, optimize=False)+plane_offset).min())
                projected = np.einsum('ni,ij->nj', transformed, basis, optimize=False)
                side = np.einsum('ni,ki->nk', projected, hull.equations[:, :2], optimize=False).max(0)+hull.equations[:, 2]
                if height < .002-1e-7 or (-.001-side).min() < -1e-7:
                    counters['support_failed'] = counters.get('support_failed', 0)+1
                    return
            score = recorded_score if recorded_score is not None else observed_fit(vertices, faces, tree, pose, depth, valid, mask, K)
            if not (score['quality'] > before['quality'] and
                    score.get('silhouette_iou', -1) >= before.get('silhouette_iou', 0) and
                    score.get('median_measured_vertex_distance_m', np.inf) < before.get('median_measured_vertex_distance_m', 0)):
                counters['source_failed'] += 1
                return
            rejection, _ = pose_rejection(pose, depth, valid, (mask*255).astype(np.uint8), K, bounds, True, prior)
            if rejection is not None or not flags[frame]:
                counters['tracker_failed'] += 1
                return
            pool.append(dict(camera_from_object=np.asarray(pose).tolist(), source_fit=score,
                             finite_support_applied=bool(support_applied), provenance=provenance))

        attempts = row['fit'].get('attempts', [])
        for index, attempt in enumerate(attempts):
            if attempt['accepted'] and attempt['optimizer_success']:
                evaluate(np.asarray(attempt['candidate_camera_from_object']),
                         support_applied=attempt['finite_support_applied'],
                         provenance=dict(method='exact-reused-qualified-SLSQP-attempt', attempt=index),
                         recorded_score=attempt['after'])
        if not row['source_fit_qualified']:
            # Finite search completion is recorded independently. Old failed
            # SLSQP statuses remain failed; only converged seeds are considered.
            for index, attempt in enumerate(attempts):
                if not attempt['optimizer_success']:
                    continue
                seed = np.asarray(attempt['candidate_camera_from_object'])
                delta = (Rotation.from_matrix(seed[:3, :3])*Rotation.from_matrix(original[frame, :3, :3]).inv()).as_rotvec()
                for alpha in report['search']['alpha']:
                    center = original[frame].copy()
                    center[:3, 3] += alpha*(seed[:3, 3]-center[:3, 3])
                    center[:3, :3] = np.einsum('ij,jk->ik', Rotation.from_rotvec(alpha*delta).as_matrix(), original[frame, :3, :3], optimize=False)
                    for dx, dy, dz in itertools.product(report['search']['pixel_offsets'], report['search']['pixel_offsets'], report['search']['depth_offsets_m']):
                        deadline()
                        pose = center.copy()
                        pose[:3, 3] += [dx*center[2, 3]/K[0, 0], dy*center[2, 3]/K[1, 1], dz]
                        evaluated += 1
                        evaluate(pose, support_applied=attempt['finite_support_applied'],
                                 provenance=dict(method='completed-finite-source-neighbourhood-trial', converged_seed_attempt=index,
                                                 alpha=alpha, pixel_offset=[dx, dy], depth_offset_m=dz))
        # Keep high-fit alternatives and a spread of small corrections. They are
        # still proposals, never a source-observation confidence update.
        ranked = sorted(pool, key=lambda p: -p['source_fit']['quality'])
        retained = ranked[:6]
        for entry in sorted(pool, key=lambda p: np.linalg.norm(np.asarray(p['camera_from_object'])[:3, 3]-original[frame, :3, 3])):
            if entry not in retained:
                retained.append(entry)
            if len(retained) >= 12:
                break
        report['frames'].append(dict(frame=frame, prior_selected_fit_qualified=row['source_fit_qualified'],
            original_object_observed_flag=bool(flags[frame]),
            new_grid_trials=evaluated, distinct_pose_trials=len(seen), rejection_counts=counters,
            qualified_candidate_count=len(pool), retained_candidates=retained,
            processing_seconds=time.monotonic()-tick))
        save()
        print('Pose pool', frame, 'new grid trials', evaluated, 'qualified', len(pool), 'retained', len(retained), 'seconds', time.monotonic()-tick, flush=True)
    report['empty_pool_frames'] = [r['frame'] for r in report['frames'] if not r['retained_candidates']]
    report['all_frame_pools_nonempty'] = not report['empty_pool_frames']
    for name, h in previous['source_files_sha256'].items():
        assert sha(S / name) == h
    for row in depth_report['complete']:
        assert sha(D / row['path']) == row['sha256']
    assert sha(F / 'report.json') == report['original_fit_report_sha256']
    report['frozen_input_closures_verified_after'] = True
    report['all_frames_complete'] = True
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['wall_seconds'] = time.monotonic()-started
    save()
