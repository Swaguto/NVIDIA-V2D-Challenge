"""Propose smoother poses near the first motion bottleneck; verify current pixels.

Neighbouring inferred poses propose a finite family, not observations. Every
retained addition must independently pass the current frame's unchanged source,
depth, support and bounds gates. No missing source flag is filled or upgraded.
"""
import copy
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--pool', type=Path, required=True)
parser.add_argument('--failure-report', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
P = args.pool.resolve()
OUT = args.out.resolve()
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
base = json.loads((P / 'report.json').read_text())
assert base['all_frames_complete'] and base['all_frame_pools_nonempty']
failure = json.loads(args.failure_report.read_text())
assert failure['candidate_pool_report_sha256'] == sha(P / 'report.json')
assert not failure['temporal_selection']['feasible']
bottleneck = int(failure['temporal_selection']['failed_frame'])
window = list(range(max(1, bottleneck-1), min(150, bottleneck+2)))
report = copy.deepcopy(base)
report.update(schema='v2d.adaptive-temporal-source-pose-pools.v1', producer_sha256=sha(Path(__file__)),
    prior_source_pool_report_sha256=sha(P / 'report.json'), all_frames_complete=False,
    frozen_input_closures_verified_after=False, changed_frames=window, failure_report_sha256=sha(args.failure_report), candidate_seed_limit_per_frame=4,
    selection_reason=f'First unchanged motion-cap failure at{bottleneck}; only local source-qualified candidates expanded, no cap changes.',
    interpolation_is_only_a_proposal_not_an_observation=True,
    numerical_control='Earlier fixed-window invalid-matmul and partial controls retained; this worker uses explicit finite checks.',
    finite_checked_explicit_contractions=True)
save = lambda: (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
with np.load(S / 'smooth-plane-camera-alignment-v2/inferred-world-alignment.npz', allow_pickle=False) as z:
    camera, original_camera = z['candidate_camera_c2w'].copy(), z['original_camera_c2w'].copy()
with np.load(S / 'fast-stereo-full-object-trajectory-v1/inferred-object-trajectory.npz', allow_pickle=False) as z:
    original, timestamps = z['original_camera_from_object'].copy(), z['timestamps_s'].copy()
with np.load(ROOT / 'bundles/e037/episode.npz', allow_pickle=False) as z:
    K = z['intrinsics'].copy(); flags = z['object_0_valid'].copy()
with np.load(ROOT / 'support/e037/inferred-support-sheet.npz', allow_pickle=False) as z:
    sheet = z['vertices_world_m'].copy()
plane = np.asarray(json.loads((ROOT / 'support/e037/report.json').read_text())['plane_world'])
mesh = trimesh.load(ROOT / 'geometry/e037/wooden_spoon/mesh.glb', force='mesh', process=False)
v, faces = np.asarray(mesh.vertices)*1.3, np.asarray(mesh.faces)
tree = cKDTree(v)
def mm(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    if a.ndim == 2 and b.ndim == 2:
        out = np.einsum('ij,jk->ik', a, b, optimize=False)
    elif a.ndim == 2 and b.ndim == 1:
        out = np.einsum('ij,j->i', a, b, optimize=False)
    else:
        raise ValueError('Unsupported finite contraction')
    assert np.isfinite(out).all()
    return out

world_pools = [np.stack([mm(camera[f], np.asarray(e['camera_from_object'])) for e in row['retained_candidates']]) for f, row in enumerate(base['frames'])]
old = json.loads((S / 'fast-stereo-full-object-trajectory-v1/report.json').read_text())
started = time.monotonic()
save()
try:
    def thin(values):
        if len(values) <= 4:
            return values
        ids = np.linspace(0, len(values)-1, 4, dtype=int)
        return [values[i] for i in ids]

    for frame in report['changed_frames']:
        assert flags[frame]
        C = camera[frame]
        normal = mm(C[:3, :3].T, plane[:3])
        offset = plane[3]+np.dot(plane[:3], C[:3, 3])
        footprint = np.einsum('ni,ij->nj', sheet-C[:3, 3], C[:3, :3], optimize=False)
        u = np.cross(normal, np.eye(3)[np.argmin(abs(normal))]); u /= np.linalg.norm(u)
        basis = np.stack([u, np.cross(normal, u)], axis=1)
        hull = ConvexHull(np.einsum('ni,ij->nj', footprint, basis, optimize=False))
        with np.load(S / f'fast-stereo-cpu-full-e037-v1/{frame:06d}-depth.npz', allow_pickle=False) as z:
            depth, valid = z['depth_m'].copy(), z['left_right_consistent'].copy()
        mask = cv2.imread(str(ROOT / f'source/e037/masks/wooden_spoon/{frame:06d}.png'), 0) > 127
        before = old['rows'][frame]['fit']['before']
        prior = transport(original[frame-1], np.linalg.inv(original_camera[frame-1]), np.linalg.inv(original_camera[frame]))
        alpha = float((timestamps[frame]-timestamps[frame-1])/(timestamps[frame+1]-timestamps[frame-1]))
        proposals = []
        for left, right in itertools.product(thin(world_pools[frame-1]), thin(world_pools[frame+1])):
            relative = Rotation.from_matrix(right[:3, :3])*Rotation.from_matrix(left[:3, :3]).inv()
            world = np.eye(4)
            world[:3, :3] = mm(Rotation.from_rotvec(alpha*relative.as_rotvec()).as_matrix(), left[:3, :3])
            world[:3, 3] = (1-alpha)*left[:3, 3]+alpha*right[:3, 3]
            proposed = mm(np.linalg.inv(C), world)
            proposals.append((proposed, 'both-world-components'))
            for entry in thin(base['frames'][frame]['retained_candidates']):
                current = np.asarray(entry['camera_from_object'])
                only_rotation = proposed.copy(); only_rotation[:3, 3] = current[:3, 3]
                proposals.append((only_rotation, 'world-rotation-current-position'))
                only_position = current.copy(); only_position[:3, 3] = proposed[:3, 3]
                proposals.append((only_position, 'world-position-current-rotation'))
        found, seen = [], set()
        trials = 0
        for seed, method in proposals:
            for dx, dy, dz in itertools.product([0., -.5, .5], [0., -.5, .5], [0., -.0005, .0005]):
                assert time.monotonic()-started < 240, 'Four-minute bounded adaptive source expansion'
                pose = seed.copy(); pose[:3, 3] += [dx*seed[2, 3]/K[0, 0], dy*seed[2, 3]/K[1, 1], dz]
                assert np.isfinite(pose).all()
                key = pose.round(10).tobytes()
                if key in seen:
                    continue
                seen.add(key); trials += 1
                if np.linalg.norm(pose[:3, 3]-original[frame, :3, 3]) > .06+1e-7:
                    continue
                if (Rotation.from_matrix(pose[:3, :3])*Rotation.from_matrix(original[frame, :3, :3]).inv()).magnitude() > np.deg2rad(20)+1e-7:
                    continue
                vertices = np.einsum('ni,ji->nj', v, pose[:3, :3], optimize=False)+pose[:3, 3]
                side = np.einsum('ni,ki->nk', np.einsum('ni,ij->nj', vertices, basis, optimize=False), hull.equations[:, :2], optimize=False).max(0)+hull.equations[:, 2]
                heights = np.einsum('ni,i->n', vertices, normal, optimize=False)+offset
                assert np.isfinite(side).all() and np.isfinite(heights).all()
                if heights.min() < .002-1e-7 or (-.001-side).min() < -1e-7:
                    continue
                score = observed_fit(v, faces, tree, pose, depth, valid, mask, K)
                if not(score['quality'] > before['quality'] and score['silhouette_iou'] >= before['silhouette_iou'] and score['median_measured_vertex_distance_m'] < before['median_measured_vertex_distance_m']):
                    continue
                rejection, _ = pose_rejection(pose, depth, valid, (mask*255).astype(np.uint8), K, np.stack([v.min(0), v.max(0)]), True, prior)
                if rejection is None:
                    new_world = mm(C, pose)
                    center_position = (1-alpha)*world_pools[frame-1][:, :3, 3].mean(0)+alpha*world_pools[frame+1][:, :3, 3].mean(0)
                    left_rotation = Rotation.from_matrix(world_pools[frame-1][:, :3, :3]).mean()
                    right_rotation = Rotation.from_matrix(world_pools[frame+1][:, :3, :3]).mean()
                    center_rotation = Rotation.from_rotvec(alpha*(right_rotation*left_rotation.inv()).as_rotvec())*left_rotation
                    angular_difference = float((Rotation.from_matrix(new_world[:3, :3])*center_rotation.inv()).magnitude())
                    smoothness = float(np.hypot(np.linalg.norm(new_world[:3, 3]-center_position)/.01, angular_difference/.1))
                    found.append(dict(camera_from_object=pose.tolist(), source_fit=score, finite_support_applied=True,
                        provenance=dict(method='source-validated-temporal-proposal', components=method, pixel_offset=[dx, dy], depth_offset_m=dz),
                        proposed_center_motion_score=smoothness))
        # Preserve every existing source-qualified choice, then add a bounded
        # spread of smoother proposals and high-quality alternatives.
        retained = list(base['frames'][frame]['retained_candidates'])
        additions = sorted(found, key=lambda x: (x['proposed_center_motion_score'], -x['source_fit']['quality']))[:12]
        for entry in sorted(found, key=lambda x: -x['source_fit']['quality']):
            if entry not in additions:
                additions.append(entry)
            if len(additions) >= 20:
                break
        retained.extend(additions)
        report['frames'][frame]['retained_candidates'] = retained
        report['frames'][frame]['temporal_expansion'] = dict(trials=trials, qualified_additions=len(found), retained_additions=len(additions), motion_caps_relaxed=False)
        save()
        print('Temporal source pool', frame, 'trials', trials, 'qualified additions', len(found), 'retained additions', len(additions), flush=True)
    for name, h in report['source_files_sha256'].items():
        assert sha(S / name) == h
    assert sha(P / 'report.json') == report['prior_source_pool_report_sha256']
    report['frozen_input_closures_verified_after'] = True
    report['all_frames_complete'] = True
    report['current_expansion_job_completed'] = True
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['temporal_expansion_wall_seconds'] = time.monotonic()-started
    save()
