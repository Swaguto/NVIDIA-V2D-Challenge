"""Propose smoother poses near the first motion bottleneck; verify current pixels.

Neighbouring inferred poses propose a finite family, not observations. Every
retained addition must independently pass the current frame's unchanged source,
depth, support and bounds gates. No missing source flag is filled or upgraded.
"""
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

P = S / 'plane-aligned-pose-pools-v2'
OUT = S / 'plane-aligned-pose-pools-v3'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
base = json.loads((P / 'report.json').read_text())
assert base['all_frames_complete'] and base['all_frame_pools_nonempty']
report = copy.deepcopy(base)
report.update(schema='v2d.plane-aligned-source-pose-pools.v3', producer_sha256=sha(Path(__file__)),
    prior_source_pool_report_sha256=sha(P / 'report.json'), all_frames_complete=False,
    frozen_input_closures_verified_after=False, changed_frames=[60, 61, 62, 63, 64],
    selection_reason='First unchanged motion-cap failure at62; local candidate expansion, no cap changes.',
    interpolation_is_only_a_proposal_not_an_observation=True)
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
world_pools = [np.stack([camera[f]@np.asarray(e['camera_from_object']) for e in row['retained_candidates']]) for f, row in enumerate(base['frames'])]
old = json.loads((S / 'fast-stereo-full-object-trajectory-v1/report.json').read_text())
started = time.monotonic()
save()
try:
    for frame in report['changed_frames']:
        assert flags[frame]
        C = camera[frame]
        normal = C[:3, :3].T@plane[:3]
        offset = plane[3]+np.dot(plane[:3], C[:3, 3])
        footprint = (sheet-C[:3, 3])@C[:3, :3]
        u = np.cross(normal, np.eye(3)[np.argmin(abs(normal))]); u /= np.linalg.norm(u)
        basis = np.stack([u, np.cross(normal, u)], axis=1)
        hull = ConvexHull(footprint@basis)
        with np.load(S / f'fast-stereo-cpu-full-e037-v1/{frame:06d}-depth.npz', allow_pickle=False) as z:
            depth, valid = z['depth_m'].copy(), z['left_right_consistent'].copy()
        mask = cv2.imread(str(ROOT / f'source/e037/masks/wooden_spoon/{frame:06d}.png'), 0) > 127
        before = old['rows'][frame]['fit']['before']
        prior = transport(original[frame-1], np.linalg.inv(original_camera[frame-1]), np.linalg.inv(original_camera[frame]))
        alpha = float((timestamps[frame]-timestamps[frame-1])/(timestamps[frame+1]-timestamps[frame-1]))
        proposals = []
        for left, right in itertools.product(world_pools[frame-1], world_pools[frame+1]):
            relative = Rotation.from_matrix(right[:3, :3])*Rotation.from_matrix(left[:3, :3]).inv()
            world = np.eye(4)
            world[:3, :3] = Rotation.from_rotvec(alpha*relative.as_rotvec()).as_matrix()@left[:3, :3]
            world[:3, 3] = (1-alpha)*left[:3, 3]+alpha*right[:3, 3]
            proposed = np.linalg.inv(C)@world
            proposals.append((proposed, 'both-world-components'))
            for entry in base['frames'][frame]['retained_candidates']:
                current = np.asarray(entry['camera_from_object'])
                only_rotation = proposed.copy(); only_rotation[:3, 3] = current[:3, 3]
                proposals.append((only_rotation, 'world-rotation-current-position'))
                only_position = current.copy(); only_position[:3, 3] = proposed[:3, 3]
                proposals.append((only_position, 'world-position-current-rotation'))
        found, seen = [], set()
        trials = 0
        for seed, method in proposals:
            for dx, dy, dz in itertools.product([0., -.5, .5], [0., -.5, .5], [0., -.0005, .0005]):
                assert time.monotonic()-started < 300, 'Five-minute temporal pool expansion cap'
                pose = seed.copy(); pose[:3, 3] += [dx*seed[2, 3]/K[0, 0], dy*seed[2, 3]/K[1, 1], dz]
                key = pose.round(10).tobytes()
                if key in seen:
                    continue
                seen.add(key); trials += 1
                if np.linalg.norm(pose[:3, 3]-original[frame, :3, 3]) > .06+1e-7:
                    continue
                if (Rotation.from_matrix(pose[:3, :3])*Rotation.from_matrix(original[frame, :3, :3]).inv()).magnitude() > np.deg2rad(20)+1e-7:
                    continue
                vertices = np.einsum('ni,ji->nj', v, pose[:3, :3], optimize=False)+pose[:3, 3]
                side = np.einsum('ni,ki->nk', vertices@basis, hull.equations[:, :2], optimize=False).max(0)+hull.equations[:, 2]
                if (vertices@normal+offset).min() < .002-1e-7 or (-.001-side).min() < -1e-7:
                    continue
                score = observed_fit(v, faces, tree, pose, depth, valid, mask, K)
                if not(score['quality'] > before['quality'] and score['silhouette_iou'] >= before['silhouette_iou'] and score['median_measured_vertex_distance_m'] < before['median_measured_vertex_distance_m']):
                    continue
                rejection, _ = pose_rejection(pose, depth, valid, (mask*255).astype(np.uint8), K, np.stack([v.min(0), v.max(0)]), True, prior)
                if rejection is None:
                    new_world = C@pose
                    center_position = .5*(world_pools[frame-1][:, :3, 3].mean(0)+world_pools[frame+1][:, :3, 3].mean(0))
                    smoothness = float(np.linalg.norm(new_world[:3, 3]-center_position))
                    found.append(dict(camera_from_object=pose.tolist(), source_fit=score, finite_support_applied=True,
                        provenance=dict(method='source-validated-temporal-proposal', components=method, pixel_offset=[dx, dy], depth_offset_m=dz),
                        proposed_center_position_difference_m=smoothness))
        # Preserve every existing source-qualified choice, then add a bounded
        # spread of smoother proposals and high-quality alternatives.
        retained = list(base['frames'][frame]['retained_candidates'])
        additions = sorted(found, key=lambda x: (x['proposed_center_position_difference_m'], -x['source_fit']['quality']))[:12]
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
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['temporal_expansion_wall_seconds'] = time.monotonic()-started
    save()
