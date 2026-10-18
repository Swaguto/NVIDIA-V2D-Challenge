"""Independently seal completed additions from an intentionally stopped search.

The source-qualified base covers every frame. Only completed added candidates
are revalidated and copied; the unfinished expansion is never labelled complete.
"""
import copy
import hashlib
import json
import sys
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

B = S / 'plane-aligned-pose-pools-v2'
P = S / 'plane-aligned-pose-pools-v4'
OUT = S / 'plane-aligned-pose-pools-v5'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
base = json.loads((B / 'report.json').read_text())
partial = json.loads((P / 'report.json').read_text())
guard = json.loads((P / 'guard.json').read_text())
assert guard['worker_exit_code'] == -15 and 'completed_utc' in guard
assert partial['finite_checked_explicit_contractions'] and not partial['all_frames_complete']
assert base['all_frames_complete'] and base['all_frame_pools_nonempty']
frames = [f for f in partial['changed_frames'] if 'temporal_expansion' in partial['frames'][f]]
assert frames == [60, 61, 62, 63]
ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
with np.load(S / 'smooth-plane-camera-alignment-v2/inferred-world-alignment.npz', allow_pickle=False) as z:
    camera, original_camera = z['candidate_camera_c2w'].copy(), z['original_camera_c2w'].copy()
with np.load(S / 'fast-stereo-full-object-trajectory-v1/inferred-object-trajectory.npz', allow_pickle=False) as z:
    original = z['original_camera_from_object'].copy()
with np.load(ROOT / 'bundles/e037/episode.npz', allow_pickle=False) as z:
    K = z['intrinsics'].copy(); flags = z['object_0_valid'].copy()
with np.load(ROOT / 'support/e037/inferred-support-sheet.npz', allow_pickle=False) as z:
    sheet = z['vertices_world_m'].copy()
plane = np.asarray(json.loads((ROOT / 'support/e037/report.json').read_text())['plane_world'])
mesh = trimesh.load(ROOT / 'geometry/e037/wooden_spoon/mesh.glb', force='mesh', process=False)
v, faces = np.asarray(mesh.vertices)*1.3, np.asarray(mesh.faces)
tree = cKDTree(v)
old = json.loads((S / 'fast-stereo-full-object-trajectory-v1/report.json').read_text())
report = copy.deepcopy(base)
report.update(schema='v2d.independently-sealed-temporal-source-pools.v1',
    producer_sha256=sha(Path(__file__)), base_source_pool_report_sha256=sha(B / 'report.json'),
    partial_expansion_report_sha256=sha(P / 'report.json'), partial_expansion_guard_sha256=sha(P / 'guard.json'),
    original_expansion_job_completed=False, original_expansion_intentionally_stopped=True,
    reason_for_early_stop='Completed additions cover the bottleneck; test the path before generating more candidates.',
    independently_revalidated_added_frames=frames, independent_new_candidate_checks=[],
    all_frames_complete=False, frozen_input_closures_verified_after=False)
for frame in frames:
    assert flags[frame]
    row = partial['frames'][frame]
    previous = base['frames'][frame]['retained_candidates']
    assert row['retained_candidates'][:len(previous)] == previous
    C = camera[frame]
    normal = np.einsum('ji,j->i', C[:3, :3], plane[:3], optimize=False)
    offset = plane[3]+np.dot(plane[:3], C[:3, 3])
    footprint = np.einsum('ni,ij->nj', sheet-C[:3, 3], C[:3, :3], optimize=False)
    u = np.cross(normal, np.eye(3)[np.argmin(abs(normal))]); u /= np.linalg.norm(u)
    basis = np.stack([u, np.cross(normal, u)], axis=1)
    hull = ConvexHull(np.einsum('ni,ij->nj', footprint, basis, optimize=False))
    with np.load(S / f'fast-stereo-cpu-full-e037-v1/{frame:06d}-depth.npz', allow_pickle=False) as z:
        depth, valid = z['depth_m'].copy(), z['left_right_consistent'].copy()
    mask = cv2.imread(str(ROOT / f'source/e037/masks/wooden_spoon/{frame:06d}.png'), 0) > 127
    prior = transport(original[frame-1], np.linalg.inv(original_camera[frame-1]), np.linalg.inv(original_camera[frame]))
    before = old['rows'][frame]['fit']['before']
    for entry in row['retained_candidates'][len(previous):]:
        pose = np.asarray(entry['camera_from_object'])
        assert np.isfinite(pose).all()
        assert np.linalg.norm(pose[:3, 3]-original[frame, :3, 3]) <= .06+1e-7
        assert (Rotation.from_matrix(pose[:3, :3])*Rotation.from_matrix(original[frame, :3, :3]).inv()).magnitude() <= np.deg2rad(20)+1e-7
        vertices = np.einsum('ni,ji->nj', v, pose[:3, :3], optimize=False)+pose[:3, 3]
        projected = np.einsum('ni,ij->nj', vertices, basis, optimize=False)
        side = np.einsum('ni,ki->nk', projected, hull.equations[:, :2], optimize=False).max(0)+hull.equations[:, 2]
        heights = np.einsum('ni,i->n', vertices, normal, optimize=False)+offset
        assert np.isfinite(side).all() and np.isfinite(heights).all()
        assert heights.min() >= .002-1e-7 and (-.001-side).min() >= -1e-7
        score = observed_fit(v, faces, tree, pose, depth, valid, mask, K)
        assert score['quality'] > before['quality'] and score['silhouette_iou'] >= before['silhouette_iou']
        assert score['median_measured_vertex_distance_m'] < before['median_measured_vertex_distance_m']
        for key in ['quality', 'silhouette_iou', 'median_measured_vertex_distance_m']:
            assert abs(score[key]-entry['source_fit'][key]) < 1e-10
        rejection, _ = pose_rejection(pose, depth, valid, (mask*255).astype(np.uint8), K, np.stack([v.min(0), v.max(0)]), True, prior)
        assert rejection is None
    report['frames'][frame] = copy.deepcopy(row)
    report['independent_new_candidate_checks'].append(dict(frame=frame, checked_additions=len(row['retained_candidates'])-len(previous), passed=True))
for name, h in report['source_files_sha256'].items():
    assert sha(S / name) == h
assert sha(P / 'report.json') == report['partial_expansion_report_sha256']
assert all(r['retained_candidates'] for r in report['frames'])
report['frozen_input_closures_verified_after'] = True
report['all_frames_complete'] = True
(OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
print('Sealed151source-frame pools; independently checked',sum(r['checked_additions'] for r in report['independent_new_candidate_checks']),'new candidates. Original expansion remains partial.')
