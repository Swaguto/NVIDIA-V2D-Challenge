"""Reuse source-qualified pools under the separately checked smooth alignment.

Only pools emptied by the corrected support relation are fitted again. All
source/image, 6 cm/20 degree, 2 mm finite-support and depth gates are retained.
Camera/world alignment remains an inferred hypothesis, including 14 unobserved
plane frames; this does not upgrade source confidence or qualify training.
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
sys.path[:0] = [str(R / 'scripts/track3'), str(S / 'compute')]
from measured_rigid_pose_probe_v2 import refine
from select_pose_hypothesis import observed_fit
from foundationpose_track import pose_rejection
from camera_prior import transport

P = S / 'source-pose-pool-repair-v1'
F = S / 'fast-stereo-full-object-trajectory-v1'
A = S / 'smooth-plane-camera-alignment-v2'
D = S / 'fast-stereo-cpu-full-e037-v1'
ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
OUT = S / 'plane-aligned-pose-pools-v1'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior_pool = json.loads((P / 'report.json').read_text())
old = json.loads((F / 'report.json').read_text())
alignment = json.loads((A / 'report.json').read_text())
assert prior_pool['all_frames_complete'] and alignment['source_alignment_stage_passed']
assert alignment['source_closures_verified_after']
assert sha(A / 'inferred-world-alignment.npz') == alignment['candidate_sha256']
with np.load(A / 'inferred-world-alignment.npz', allow_pickle=False) as z:
    camera = z['candidate_camera_c2w'].copy()
    original_camera = z['original_camera_c2w'].copy()
    plane_qualified = z['source_plane_qualified'].copy()
with np.load(F / 'inferred-object-trajectory.npz', allow_pickle=False) as z:
    original, flags = z['original_camera_from_object'].copy(), z['original_object_observed_flags'].copy()
with np.load(ROOT / 'bundles/e037/episode.npz', allow_pickle=False) as z:
    K = z['intrinsics'].copy()
plane = np.asarray(json.loads((ROOT / 'support/e037/report.json').read_text())['plane_world'])
with np.load(ROOT / 'support/e037/inferred-support-sheet.npz', allow_pickle=False) as z:
    sheet = z['vertices_world_m'].copy()
mesh = trimesh.load(ROOT / 'geometry/e037/wooden_spoon/mesh.glb', force='mesh', process=False)
vertices, faces = np.asarray(mesh.vertices)*1.3, np.asarray(mesh.faces)
tree = cKDTree(vertices)
bounds = np.stack([vertices.min(0), vertices.max(0)])
inference = json.loads((D / 'depth-report.json').read_text())
samples = json.loads((S / 'plane-aligned-source-samples-v1/report.json').read_text())
sample_by_frame = {r['frame']: r for r in samples['frames']}
report = dict(schema='v2d.plane-aligned-source-pose-pools.v1', episode=37, frames=[],
    producer_sha256=sha(Path(__file__)), original_fit_report_sha256=sha(F / 'report.json'),
    reused_pool_report_sha256=sha(P / 'report.json'), source_files_sha256=old['source_files_sha256'],
    depth_report_sha256=sha(D / 'depth-report.json'),
    alignment_report_sha256=sha(A / 'report.json'), alignment_npz_sha256=sha(A / 'inferred-world-alignment.npz'),
    source_camera_estimates_changed=True, source_pixel_or_confidence_changed=False,
    source_plane_inferred_only_frames=np.flatnonzero(~plane_qualified).tolist(),
    estimated_depth_is_not_ground_truth=True, retained_conflicting_evidence=old['retained_conflicting_evidence'],
    support_changed=False, geometry_uniform_scale=1.3, observations_filled=False,
    acceptance_gates_relaxed=False, failed_optimizer_runs_promoted=False,
    all_frames_complete=False, motion_quality_certified=False, full_native_contact_gate_completed=False,
    physics_stepped=False, training_steps=0, trajectory_promoted=False, physical_task_accepted=False)
save = lambda: (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
started = time.monotonic()
save()
try:
    for frame in range(151):
        assert time.monotonic()-started < 1500, 'Bounded aligned source-pool repair'
        tick = time.monotonic()
        depth_path = D / f'{frame:06d}-depth.npz'
        assert sha(depth_path) == inference['complete'][frame]['sha256']
        with np.load(depth_path, allow_pickle=False) as z:
            depth, valid = z['depth_m'].copy(), z['left_right_consistent'].copy()
        mask = cv2.imread(str(ROOT / f'source/e037/masks/wooden_spoon/{frame:06d}.png'), 0) > 127
        C = camera[frame]
        normal = np.einsum('ji,j->i', C[:3, :3], plane[:3], optimize=False)
        offset = plane[3]+np.dot(plane[:3], C[:3, 3])
        plane_camera = np.r_[normal, offset]
        footprint = np.einsum('ni,ij->nj', sheet-C[:3, 3], C[:3, :3], optimize=False)
        u = np.cross(normal, np.eye(3)[np.argmin(abs(normal))]); u /= np.linalg.norm(u)
        basis = np.stack([u, np.cross(normal, u)], axis=1)
        hull = ConvexHull(np.einsum('ni,ij->nj', footprint, basis, optimize=False))
        prior = transport(original[frame-1], np.linalg.inv(original_camera[frame-1]), np.linalg.inv(original_camera[frame])) if frame else None
        before = old['rows'][frame]['fit']['before']
        pool, seen = [], set()

        def evaluate(pose, support_applied, provenance, score=None):
            key = np.asarray(pose).round(11).tobytes()
            if key in seen:
                return
            seen.add(key)
            if np.linalg.norm(pose[:3, 3]-original[frame, :3, 3]) > .06+1e-7:
                return
            angle = (Rotation.from_matrix(pose[:3, :3])*Rotation.from_matrix(original[frame, :3, :3]).inv()).magnitude()
            if angle > np.deg2rad(20)+1e-7:
                return
            transformed = np.einsum('ni,ji->nj', vertices, pose[:3, :3], optimize=False)+pose[:3, 3]
            if support_applied:
                h = np.einsum('ni,i->n', transformed, normal, optimize=False)+offset
                projected = np.einsum('ni,ij->nj', transformed, basis, optimize=False)
                side = np.einsum('ni,ki->nk', projected, hull.equations[:, :2], optimize=False).max(0)+hull.equations[:, 2]
                if h.min() < .002-1e-7 or (-.001-side).min() < -1e-7:
                    return
            score = observed_fit(vertices, faces, tree, pose, depth, valid, mask, K) if score is None else score
            if not (score['quality'] > before['quality'] and score['silhouette_iou'] >= before['silhouette_iou']
                    and score['median_measured_vertex_distance_m'] < before['median_measured_vertex_distance_m']):
                return
            rejection, _ = pose_rejection(pose, depth, valid, (mask*255).astype(np.uint8), K, bounds, True, prior)
            if rejection is None and flags[frame]:
                pool.append(dict(camera_from_object=np.asarray(pose).tolist(), source_fit=score,
                    finite_support_applied=bool(support_applied), provenance=provenance))

        for index, entry in enumerate(prior_pool['frames'][frame]['retained_candidates']):
            evaluate(np.asarray(entry['camera_from_object']), entry['finite_support_applied'],
                     dict(method='reused-source-qualified-pool-with-new-support-recheck', candidate=index), entry['source_fit'])
        if frame in sample_by_frame:
            sample = sample_by_frame[frame]
            fit = sample['source_fit']
            if fit['accepted']:
                attempt = fit['attempts'][fit['selected_attempt']]
                evaluate(np.asarray(sample['candidate_camera_from_object']), attempt['finite_support_applied'],
                         dict(method='reused-two-knot-alignment-source-fit'), attempt['after'])
        reused_count = len(pool)
        repair = None
        if not pool:
            candidate, repair = refine(vertices, faces, original[frame], depth, valid, mask, K, plane_camera, footprint)
            for index, attempt in enumerate(repair.get('attempts', [])):
                if attempt['accepted'] and attempt['optimizer_success']:
                    evaluate(np.asarray(attempt['candidate_camera_from_object']), attempt['finite_support_applied'],
                             dict(method='new-bounded-source-fit-after-alignment', attempt=index), attempt['after'])
            if not pool:
                for index, attempt in enumerate(repair.get('attempts', [])):
                    if not attempt['optimizer_success']:
                        continue
                    seed = np.asarray(attempt['candidate_camera_from_object'])
                    delta = (Rotation.from_matrix(seed[:3, :3])*Rotation.from_matrix(original[frame, :3, :3]).inv()).as_rotvec()
                    for alpha in [.25, .5, .75, 1.]:
                        center = original[frame].copy()
                        center[:3, 3] += alpha*(seed[:3, 3]-center[:3, 3])
                        center[:3, :3] = np.einsum('ij,jk->ik', Rotation.from_rotvec(alpha*delta).as_matrix(), original[frame, :3, :3], optimize=False)
                        for dx, dy, dz in itertools.product([0., -1., 1., -2., 2.], [0., -1., 1., -2., 2.], [0., -.001, .001, -.003, .003]):
                            assert time.monotonic()-started < 1500
                            pose = center.copy()
                            pose[:3, 3] += [dx*center[2, 3]/K[0, 0], dy*center[2, 3]/K[1, 1], dz]
                            evaluate(pose, attempt['finite_support_applied'], dict(method='finite-source-neighbourhood-after-alignment', seed_attempt=index, alpha=alpha, pixel_offset=[dx, dy], depth_offset_m=dz))
        ranked = sorted(pool, key=lambda p: -p['source_fit']['quality'])
        retained = ranked[:6]
        for entry in sorted(pool, key=lambda p: np.linalg.norm(np.asarray(p['camera_from_object'])[:3, 3]-original[frame, :3, 3])):
            if entry not in retained:
                retained.append(entry)
            if len(retained) >= 12:
                break
        report['frames'].append(dict(frame=frame, original_object_observed_flag=bool(flags[frame]),
            plane_directly_qualified=bool(plane_qualified[frame]), reused_qualified_count=reused_count,
            qualified_candidate_count=len(pool), retained_candidates=retained,
            new_fit=repair, processing_seconds=time.monotonic()-tick))
        save()
        print('Aligned pool', frame, 'reused', reused_count, 'qualified', len(pool), 'retained', len(retained), 'seconds', time.monotonic()-tick, flush=True)
    report['empty_pool_frames'] = [r['frame'] for r in report['frames'] if not r['retained_candidates']]
    report['all_frame_pools_nonempty'] = not report['empty_pool_frames']
    for name, h in report['source_files_sha256'].items():
        assert sha(S / name) == h
    assert sha(A / 'report.json') == report['alignment_report_sha256']
    assert sha(P / 'report.json') == report['reused_pool_report_sha256']
    report['frozen_input_closures_verified_after'] = True
    report['all_frames_complete'] = True
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['wall_seconds'] = time.monotonic()-started
    save()
