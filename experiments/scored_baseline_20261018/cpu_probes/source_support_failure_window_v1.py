"""Inspect failed knots and their neighbours without replacing the support."""
import ast
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0, str(R / 'reconstruction/modules/v2d_stereo_depth/lib'))
from stereo_depth import load_rig

ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
D = S / 'fast-stereo-cpu-full-e037-v1'
OUT = S / 'source-support-failure-window-v1'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
producer = S / 'compute/spoon_source_support_audit_v1.py'
nodes = [n for n in ast.parse(producer.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in {'stats', 'points', 'local_plane'}]
assert len(nodes) == 3
namespace = dict(np=np)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(producer), 'exec'), namespace)
stats, points, local_plane = [namespace[n] for n in ['stats', 'points', 'local_plane']]
with np.load(ROOT / 'bundles/e037/episode.npz', allow_pickle=False) as z:
    camera, K = z['camera_c2w'].copy(), z['intrinsics'].copy()
namespace['K'] = K
plane = np.asarray(json.loads((ROOT / 'support/e037/report.json').read_text())['plane_world'])
inference = json.loads((D / 'depth-report.json').read_text())
assert inference['all_frames_complete']
old = json.loads((S / 'fast-stereo-source-comparison-seven-v1/report.json').read_text())
calib = S / 'dataset/track_3/public/meta/camera_calibration.json'
rig = load_rig(str(calib))
videos = [S / f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{cam}/episode_000037.mp4' for cam in [rig.left, rig.right]]
for p in videos:
    assert sha(p) == inference['source_files_sha256'][str(p.relative_to(S))]
caps = [cv2.VideoCapture(str(p)) for p in videos]
frames = list(range(124, 131))
report = dict(schema='v2d.source-support-failure-window.v1', episode=37, frames=frames, rows=[],
    producer_sha256=sha(Path(__file__)), audit_functions_sha256=sha(producer),
    full_depth_report_sha256=sha(D / 'depth-report.json'),
    existing_seven_control_report_sha256=sha(S / 'fast-stereo-source-comparison-seven-v1/report.json'),
    selection_reason='First remaining source-fit failure126 and neighbouring128; contiguous window124..130, no reference labels.',
    source_inputs_modified=False, background_membership_uncertain=True, hand_masks_available=False,
    depth_estimates_are_not_ground_truth=True, support_replaced=False,
    original_flags_changed=False, trajectory_promoted=False, physical_task_accepted=False,
    physics_stepped=False, training_steps=0)
panels = []
try:
    for frame in range(131):
        raw = []
        for cap in caps:
            ok, image = cap.read(); assert ok; raw.append(image)
        if frame not in frames:
            continue
        source, _ = rig.rectify_pair(*raw)
        path = D / f'{frame:06d}-depth.npz'
        assert sha(path) == inference['complete'][frame]['sha256']
        with np.load(path, allow_pickle=False) as z:
            depth, valid = z['depth_m'].copy(), z['left_right_consistent'].copy()
        mask_path = ROOT / f'source/e037/masks/wooden_spoon/{frame:06d}.png'
        mask = cv2.imread(str(mask_path), 0) > 127
        interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        metric = valid & np.isfinite(depth) & (depth > .15) & (depth < 1.4)
        p, yy, xx = points(depth, metric)
        C = camera[frame]
        world = np.einsum('ni,ji->nj', p, C[:3, :3], optimize=False)+C[:3, 3]
        h = np.einsum('ni,i->n', world, plane[:3], optimize=False)+plane[3]
        fg, eroded = mask[yy, xx], interior[yy, xx]
        near = (distance[yy, xx] > 26) & (distance[yy, xx] < 100) & (abs(h) < .035)
        split = ((xx//8+yy//8) % 2) == 0
        ids = np.flatnonzero(near & split)[::3]
        row = dict(frame=frame, mask_sha256=sha(mask_path), estimated_foreground_valid_coverage=float((metric & mask).sum()/mask.sum()),
                   foreground_global_height=stats(h[fg]), interior_global_height=stats(h[eroded]),
                   nearby_prior_height=stats(h[near]), nearby_prior_heldout_abs_residual=stats(abs(h[near & ~split])))
        if frame == 125:
            prior = next(r for r in old['rows'] if r['frame'] == frame)
            lp = np.asarray(prior['backends'][1]['local_plane']['plane_world'])
            fitted = lp, np.ones(len(ids), dtype=bool)
            reused = True
        else:
            fitted = local_plane(world[ids], plane)
            reused = False
        if fitted is not None:
            lp, inliers = fitted
            local = np.einsum('ni,i->n', world, lp[:3], optimize=False)+lp[3]
            row['local_plane'] = dict(plane_world=lp.tolist(), reused_verified_prior_control=reused,
                fit_points=len(ids), fit_inliers=None if reused else int(inliers.sum()),
                normal_angle_to_prior_deg=float(np.rad2deg(np.arccos(np.clip(np.dot(lp[:3], plane[:3]), -1, 1)))),
                heldout_background_abs_residual=stats(abs(local[near & ~split])),
                foreground_height=stats(local[fg]), interior_height=stats(local[eroded]))
        display = source.copy()
        cv2.drawContours(display, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 2)
        color = cv2.applyColorMap(np.clip((h+.035)/.07*255, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
        heat = source//3; heat[yy[fg], xx[fg]] = color[fg]
        for panel, text in [(display, 'rectified source + automatic mask'), (heat, 'estimated foreground height to unchanged table, +/-35mm')]:
            cv2.rectangle(panel, (0, 0), (1280, 35), (0, 0, 0), -1)
            cv2.putText(panel, f'e037 f{frame}: {text}', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        collage = np.concatenate([cv2.resize(im, (640, 400)) for im in [display, heat]], axis=1)
        cv2.imwrite(str(OUT / f'frame-{frame:03d}.jpg'), collage)
        panels.append(cv2.resize(collage, (960, 300)))
        report['rows'].append(row)
        (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    cv2.imwrite(str(OUT / 'window.jpg'), np.concatenate(panels, axis=0))
    for p in videos:
        assert sha(p) == inference['source_files_sha256'][str(p.relative_to(S))]
    report['source_closure_verified_after'] = True
finally:
    for cap in caps:
        cap.release()
    (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({r['frame']:dict(global_median=r['interior_global_height'].get('quantiles_m', [None]*5)[2],
    local_median=r.get('local_plane', {}).get('interior_height', {}).get('quantiles_m', [None]*5)[2],
    prior_heldout_residual=r['nearby_prior_heldout_abs_residual'].get('quantiles_m', [None]*5)[2],
    local_heldout_residual=r.get('local_plane', {}).get('heldout_background_abs_residual', {}).get('quantiles_m', [None]*5)[2]) for r in report['rows']}, indent=2))
