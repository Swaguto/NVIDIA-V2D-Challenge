"""Observe source-derived table planes across the clock; no camera repair yet.

Unavailable/unsupported observations stay NaN with false flags. A projected
directly-observed hand exclusion reduces contamination, but background identity
and depth accuracy remain uncertain. This never replaces the static support.
"""
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

S = Path(__file__).resolve().parents[1]
ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
D = S / 'fast-stereo-cpu-full-e037-v1'
OUT = S / 'full-static-support-observations-v1'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
helper = S / 'compute/spoon_source_support_audit_v1.py'
nodes = [n for n in ast.parse(helper.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in {'stats', 'points', 'local_plane'}]
namespace = dict(np=np)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(helper), 'exec'), namespace)
stats, points, local_plane = [namespace[k] for k in ['stats', 'points', 'local_plane']]
bundle = ROOT / 'bundles/e037/episode.npz'
with np.load(bundle, allow_pickle=False) as z:
    camera, K = z['camera_c2w'].copy(), z['intrinsics'].copy()
    joints = z['hand_joints_world'].copy()
    hand_valid, hand_direct = z['hand_valid'].copy(), z['hand_directly_observed'].copy()
    axis, timestamps = z['frame_indices'].copy(), z['timestamps_s'].copy()
namespace['K'] = K
prior_path = ROOT / 'support/e037/report.json'
prior = np.asarray(json.loads(prior_path.read_text())['plane_world'])
inference = json.loads((D / 'depth-report.json').read_text())
assert inference['all_frames_complete']
planes = np.full((151, 4), np.nan)
observed = np.zeros(151, bool)
qualified = np.zeros(151, bool)
train_points = np.full((151, 256, 3), np.nan)
heldout_points = np.full((151, 256, 3), np.nan)
train_counts, heldout_counts = np.zeros(151, int), np.zeros(151, int)
report = dict(schema='v2d.full-source-static-support-observations.v1', episode=37, rows=[],
    producer_sha256=sha(Path(__file__)), helper_sha256=sha(helper),
    source_bundle_sha256=sha(bundle), original_support_sha256=sha(prior_path),
    depth_report_sha256=sha(D / 'depth-report.json'),
    background_membership_uncertain=True, depth_truth_verified=False,
    projected_hand_exclusion_is_not_a_measured_hand_mask=True,
    fit_split='Fixed alternating8px checkerboard; withheld cells never used for fit.',
    qualification=dict(minimum_fit_inliers=150, maximum_median_heldout_residual_m=.002,
        minimum_heldout_fraction_inside_4mm=.5, heldout_median_must_improve_original=True),
    source_confidence_modified=False, missing_observations_filled=False,
    source_cameras_changed=False, support_replaced=False, object_targets_changed=False,
    all_frames_complete=False, physics_stepped=False, training_steps=0,
    trajectory_promoted=False, physical_task_accepted=False)
save = lambda: (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
started = time.monotonic()
save()
try:
    for frame in axis:
        frame = int(frame)
        assert time.monotonic()-started < 300, 'Five-minute source-plane observation cap'
        path = D / f'{frame:06d}-depth.npz'
        assert sha(path) == inference['complete'][frame]['sha256']
        with np.load(path, allow_pickle=False) as z:
            depth, valid = z['depth_m'].copy(), z['left_right_consistent'].copy()
        mask_path = ROOT / f'source/e037/masks/wooden_spoon/{frame:06d}.png'
        mask = cv2.imread(str(mask_path), 0) > 127
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        hand_exclusion = np.zeros(mask.shape, np.uint8)
        excluded_sides = []
        C = camera[frame]
        for si in range(2):
            if not (hand_valid[frame, si] and hand_direct[frame, si]):
                continue
            p = np.einsum('ni,ij->nj', joints[frame, si]-C[:3, 3], C[:3, :3], optimize=False)
            visible = np.isfinite(p).all(1) & (p[:, 2] > .05)
            if visible.sum() < 4:
                continue
            p = p[visible]
            uv = np.c_[K[0, 0]*p[:, 0]/p[:, 2]+K[0, 2], K[1, 1]*p[:, 1]/p[:, 2]+K[1, 2]]
            hull = cv2.convexHull(np.rint(np.clip(uv, -2000, 3000)).astype(np.int32))
            cv2.fillConvexPoly(hand_exclusion, hull, 1)
            excluded_sides.append(si)
        hand_exclusion = cv2.dilate(hand_exclusion, np.ones((21, 21), np.uint8)) > 0
        metric = valid & np.isfinite(depth) & (depth > .15) & (depth < 1.4)
        p, yy, xx = points(depth, metric)
        world = np.einsum('ni,ji->nj', p, C[:3, :3], optimize=False)+C[:3, 3]
        h = np.einsum('ni,i->n', world, prior[:3], optimize=False)+prior[3]
        near = (distance[yy, xx] > 26) & (distance[yy, xx] < 100) & (abs(h) < .035) & ~hand_exclusion[yy, xx]
        split = ((xx//8+yy//8) % 2) == 0
        ids = np.flatnonzero(near & split)[::3]
        held = np.flatnonzero(near & ~split)
        fitted = local_plane(world[ids], prior)
        row = dict(frame=frame, depth_sha256=inference['complete'][frame]['sha256'], mask_sha256=sha(mask_path),
            projected_hand_exclusion_sides=excluded_sides, fit_points=len(ids), heldout_points=len(held),
            original_heldout_abs_residual=stats(abs(h[held])), plane_observed=False, plane_qualified=False)
        for src, dest, counts in [(ids, train_points, train_counts), (held, heldout_points, heldout_counts)]:
            if len(src):
                choose = src[np.linspace(0, len(src)-1, min(256, len(src)), dtype=int)]
                dest[frame, :len(choose)] = world[choose]
                counts[frame] = len(choose)
        if fitted is not None and len(held) >= 150:
            plane, inliers = fitted
            residual = abs(np.einsum('ni,i->n', world[held], plane[:3], optimize=False)+plane[3])
            observed[frame] = True; planes[frame] = plane
            qualified[frame] = bool(np.median(residual) <= .002 and (residual < .004).mean() >= .5
                                    and np.median(residual) < np.median(abs(h[held])))
            row.update(plane_observed=True, plane_qualified=bool(qualified[frame]),
                plane_world=plane.tolist(), fit_inliers=int(inliers.sum()),
                heldout_abs_residual=stats(residual), heldout_inlier_fraction_4mm=float((residual < .004).mean()))
        report['rows'].append(row)
        save()
        print('Support observation', frame, 'observed', bool(observed[frame]), 'qualified', bool(qualified[frame]), flush=True)
    np.savez_compressed(OUT / 'background-plane-observations.npz', plane_world=planes,
        plane_observed=observed, plane_qualified=qualified,
        training_points_world=train_points, heldout_points_world=heldout_points,
        training_point_counts=train_counts, heldout_point_counts=heldout_counts,
        frame_indices=axis, timestamps_s=timestamps)
    report['output_sha256'] = sha(OUT / 'background-plane-observations.npz')
    report['observed_plane_frames'] = int(observed.sum())
    report['qualified_plane_frames'] = int(qualified.sum())
    report['missing_or_rejected_plane_frames'] = np.flatnonzero(~qualified).tolist()
    assert sha(bundle) == report['source_bundle_sha256']
    assert sha(prior_path) == report['original_support_sha256']
    assert sha(D / 'depth-report.json') == report['depth_report_sha256']
    report['source_closures_verified_after'] = True
    report['all_frames_complete'] = True
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['wall_seconds'] = time.monotonic()-started
    save()
