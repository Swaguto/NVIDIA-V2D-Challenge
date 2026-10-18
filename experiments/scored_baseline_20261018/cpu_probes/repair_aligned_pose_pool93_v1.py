"""Narrow source-grid repair at the only empty aligned pool; reuse other150."""
import copy
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
sys.path.insert(0, str(R / 'scripts/track3'))
from select_pose_hypothesis import observed_fit
from foundationpose_track import pose_rejection
from camera_prior import transport

P = S / 'plane-aligned-pose-pools-v1'
OUT = S / 'plane-aligned-pose-pools-v2'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = copy.deepcopy(json.loads((P / 'report.json').read_text()))
assert report['all_frames_complete'] and report['empty_pool_frames'] == [93]
assert json.loads((P / 'guard.json').read_text())['worker_exit_code'] == 0
report.update(schema='v2d.plane-aligned-source-pose-pools.v2',
    producer_sha256=sha(Path(__file__)), prior_failed_pool_report_sha256=sha(P / 'report.json'),
    all_frames_complete=False, frozen_input_closures_verified_after=False)
save = lambda: (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
with np.load(S / 'smooth-plane-camera-alignment-v2/inferred-world-alignment.npz', allow_pickle=False) as z:
    camera, original_camera = z['candidate_camera_c2w'].copy(), z['original_camera_c2w'].copy()
with np.load(S / 'fast-stereo-full-object-trajectory-v1/inferred-object-trajectory.npz', allow_pickle=False) as z:
    original = z['original_camera_from_object'].copy()
with np.load(ROOT / 'bundles/e037/episode.npz', allow_pickle=False) as z:
    K = z['intrinsics'].copy()
with np.load(ROOT / 'support/e037/inferred-support-sheet.npz', allow_pickle=False) as z:
    sheet = z['vertices_world_m'].copy()
plane = np.asarray(json.loads((ROOT / 'support/e037/report.json').read_text())['plane_world'])
frame = 93; C = camera[frame]
n = np.einsum('ji,j->i', C[:3, :3], plane[:3], optimize=False)
d = plane[3]+np.dot(plane[:3], C[:3, 3])
footprint = np.einsum('ni,ij->nj', sheet-C[:3, 3], C[:3, :3], optimize=False)
u = np.cross(n, np.eye(3)[np.argmin(abs(n))]); u /= np.linalg.norm(u)
basis = np.stack([u, np.cross(n, u)], axis=1)
hull = ConvexHull(np.einsum('ni,ij->nj', footprint, basis, optimize=False))
mesh = trimesh.load(ROOT / 'geometry/e037/wooden_spoon/mesh.glb', force='mesh', process=False)
v, faces = np.asarray(mesh.vertices)*1.3, np.asarray(mesh.faces)
tree = cKDTree(v)
with np.load(S / f'fast-stereo-cpu-full-e037-v1/{frame:06d}-depth.npz', allow_pickle=False) as z:
    depth, valid = z['depth_m'].copy(), z['left_right_consistent'].copy()
mask = cv2.imread(str(ROOT / f'source/e037/masks/wooden_spoon/{frame:06d}.png'), 0) > 127
prior = transport(original[frame-1], np.linalg.inv(original_camera[frame-1]), np.linalg.inv(original_camera[frame]))
before = report['frames'][frame]['new_fit']['before']
seeds = []
for i, attempt in enumerate(report['frames'][frame]['new_fit']['attempts']):
    if attempt['optimizer_success']:
        seeds.append((np.asarray(attempt['candidate_camera_from_object']), dict(method='converged-new-fit', attempt=i)))
old_pool = json.loads((S / 'source-pose-pool-repair-v1/report.json').read_text())['frames'][frame]['retained_candidates']
for index, entry in enumerate(old_pool):
    pose = np.asarray(entry['camera_from_object'])
    seeds.append((pose, dict(method='reused-source-qualified-old-pool', candidate=index)))
    vertices = np.einsum('ni,ji->nj', v, pose[:3, :3], optimize=False)+pose[:3, 3]
    missing = max(0., .002-(np.einsum('ni,i->n', vertices, n, optimize=False)+d).min())
    ray = -pose[:3, 3]/np.linalg.norm(pose[:3, 3])
    normal_component = float(np.dot(n, ray))
    if normal_component > .1:
        lifted = pose.copy(); lifted[:3, 3] += ray*missing/normal_component
        seeds.append((lifted, dict(method='analytic-centroid-view-ray-support-seed', candidate=index,
                                  minimum_lift_m=missing/normal_component)))
started = time.monotonic()
pool, seen = [], set()
trials = 0
save()
try:
    for seed, provenance in seeds:
        for dx, dy, dz, roll in itertools.product([0., -.5, .5, -1., 1.], [0., -.5, .5, -1., 1.], [0., -.0005, .0005], [0., -.01, .01]):
            assert time.monotonic()-started < 240, 'Four-minute narrow pose-grid cap'
            pose = seed.copy()
            pose[:3, 3] += [dx*seed[2, 3]/K[0, 0], dy*seed[2, 3]/K[1, 1], dz]
            pose[:3, :3] = np.einsum('ij,jk->ik', Rotation.from_rotvec([0, 0, roll]).as_matrix(), seed[:3, :3], optimize=False)
            key = pose.round(11).tobytes()
            if key in seen:
                continue
            seen.add(key); trials += 1
            if np.linalg.norm(pose[:3, 3]-original[frame, :3, 3]) > .06+1e-7:
                continue
            if (Rotation.from_matrix(pose[:3, :3])*Rotation.from_matrix(original[frame, :3, :3]).inv()).magnitude() > np.deg2rad(20)+1e-7:
                continue
            vertices = np.einsum('ni,ji->nj', v, pose[:3, :3], optimize=False)+pose[:3, 3]
            projected = np.einsum('ni,ij->nj', vertices, basis, optimize=False)
            side = np.einsum('ni,ki->nk', projected, hull.equations[:, :2], optimize=False).max(0)+hull.equations[:, 2]
            if (np.einsum('ni,i->n', vertices, n, optimize=False)+d).min() < .002-1e-7 or (-.001-side).min() < -1e-7:
                continue
            score = observed_fit(v, faces, tree, pose, depth, valid, mask, K)
            if not(score['quality'] > before['quality'] and score['silhouette_iou'] >= before['silhouette_iou'] and score['median_measured_vertex_distance_m'] < before['median_measured_vertex_distance_m']):
                continue
            rejection, _ = pose_rejection(pose, depth, valid, (mask*255).astype(np.uint8), K, np.stack([v.min(0), v.max(0)]), True, prior)
            if rejection is None:
                pool.append(dict(camera_from_object=pose.tolist(), source_fit=score,
                    finite_support_applied=True, provenance=dict(seed=provenance, pixel_offset=[dx, dy], depth_offset_m=dz, optical_axis_rotation_rad=roll)))
    ranked = sorted(pool, key=lambda x: -x['source_fit']['quality'])
    retained = ranked[:6]
    for entry in sorted(pool, key=lambda x: np.linalg.norm(np.asarray(x['camera_from_object'])[:3, 3]-original[frame, :3, 3])):
        if entry not in retained:
            retained.append(entry)
        if len(retained) >= 12:
            break
    row = report['frames'][frame]
    row['retained_candidates'] = retained
    row['qualified_candidate_count'] = len(pool)
    row['narrow_grid_repair'] = dict(trials=trials, qualified=len(pool), old_failed_fit_retained=True,
                                    other150frames_reused_exactly=True, seconds=time.monotonic()-started)
    report['empty_pool_frames'] = [r['frame'] for r in report['frames'] if not r['retained_candidates']]
    report['all_frame_pools_nonempty'] = not report['empty_pool_frames']
    for name, h in report['source_files_sha256'].items():
        assert sha(S / name) == h
    assert sha(P / 'report.json') == report['prior_failed_pool_report_sha256']
    report['frozen_input_closures_verified_after'] = True
    report['all_frames_complete'] = True
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['narrow_repair_wall_seconds'] = time.monotonic()-started
    save()
    print('Frame93 narrow search', trials, 'qualified', len(pool), flush=True)
