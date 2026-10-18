"""Private CPU diagnosis of measured foreground/background support consistency.

No source, pose, mesh, support, confidence or training artifact is changed.
Local plane estimates are diagnostic hypotheses, not replacements for support.
"""
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
from stereo_depth import load_rig, stereo_depth

ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
OUT = S / 'spoon-source-support-audit-v1'
OUT.mkdir()
rig = load_rig(str(S / 'dataset/track_3/public/meta/camera_calibration.json'))
K = rig.P1[:3, :3]
with np.load(ROOT / 'bundles/e037/episode.npz', allow_pickle=False) as z:
    camera = z['camera_c2w'].copy()
    np.testing.assert_allclose(K, z['intrinsics'], atol=1e-6)
plane = np.asarray(json.loads((ROOT / 'support/e037/report.json').read_text())['plane_world'])
caps = [cv2.VideoCapture(str(S / f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{cam}/episode_000037.mp4')) for cam in (rig.left, rig.right)]
report = dict(schema='v2d.source-support-consistency-diagnostic.v1', episode=37,
              rows=[], source_inputs_modified=False, hypothesis_promoted=False,
              native_contact_claimed=False, new_gpu_lease_started=False,
              producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              selection='Measured points near the prior plane, object mask excluded; no hand masks available, so background membership is uncertain.')


def stats(values):
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), quantiles_m=np.quantile(values, [.05, .25, .5, .75, .95]).tolist(),
                below_minus_1mm_fraction=float((values < -.001).mean()))


def points(depth, selection):
    y, x = np.nonzero(selection)
    z = depth[y, x].astype(float)
    p = np.c_[(x-K[0, 2])*z/K[0, 0], (y-K[1, 2])*z/K[1, 1], z]
    return p, y, x


def local_plane(p, prior):
    """Robust fit to a measured nearby surface, constrained near prior normal."""
    if len(p) < 150:
        return None
    rng = np.random.default_rng(20261018)
    best = None
    for _ in range(300):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b-a, c-a)
        length = np.linalg.norm(n)
        if length < 1e-8:
            continue
        n /= length
        if np.dot(n, prior[:3]) < 0:
            n = -n
        if np.dot(n, prior[:3]) < np.cos(np.deg2rad(15)):
            continue
        d = -np.dot(n, a)
        inside = abs(np.einsum('ni,i->n', p, n, optimize=False)+d) < .004
        if best is None or inside.sum() > best.sum():
            best = inside
    if best is None or best.sum() < 150:
        return None
    for _ in range(3):
        center = p[best].mean(0)
        n = np.linalg.svd(p[best]-center, full_matrices=False)[2][-1]
        if np.dot(n, prior[:3]) < 0:
            n = -n
        d = -np.dot(n, center)
        updated = abs(np.einsum('ni,i->n', p, n, optimize=False)+d) < .004
        if updated.sum() < 150:
            return None
        best = updated
    if np.dot(n, prior[:3]) < np.cos(np.deg2rad(15)):
        return None
    return np.r_[n, d], best


for frame in [0, 25, 50, 75, 100, 125, 150]:
    raw = []
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        assert ok
        raw.append(image)
    depth, valid = stereo_depth(rig, *raw)
    image = rig.rectify_pair(*raw)[0]
    mask = cv2.imread(str(ROOT / f'source/e037/masks/wooden_spoon/{frame:06d}.png'), 0) > 127
    interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    metric = valid & np.isfinite(depth) & (depth > .15) & (depth < 1.4)
    p, yy, xx = points(depth, metric)
    P = camera[frame]
    world = np.einsum('ni,ji->nj', p, P[:3, :3], optimize=False)+P[:3, 3]
    h = np.einsum('ni,i->n', world, plane[:3], optimize=False)+plane[3]
    fg, inside = mask[yy, xx], interior[yy, xx]
    # Grid split is fixed before fitting, not selected by fit quality.
    near = (distance[yy, xx] > 26) & (distance[yy, xx] < 100) & (abs(h) < .035)
    split = ((xx//8 + yy//8) % 2) == 0
    ids = np.flatnonzero(near & split)[::3]
    fitted = local_plane(world[ids], plane)
    row = dict(frame=frame, masked_measured_heights=stats(h[fg]),
               eroded_mask_measured_heights=stats(h[inside]),
               nearby_background_prior_heights=stats(h[near]),
               mask_pixels=int(mask.sum()), measured_mask_pixels=int(fg.sum()))
    local_h = None
    if fitted is not None:
        lp, inliers = fitted
        local_h = np.einsum('ni,i->n', world, lp[:3], optimize=False)+lp[3]
        held = near & ~split
        row['local_plane'] = dict(plane_world=lp.tolist(), fit_points=len(ids),
                                 fit_inliers=int(inliers.sum()),
                                 prior_normal_angle_deg=float(np.rad2deg(np.arccos(np.clip(np.dot(lp[:3], plane[:3]), -1, 1)))),
                                 heldout_measured_background_residual=stats(abs(local_h[held])),
                                 foreground_signed_height=stats(local_h[fg]),
                                 eroded_foreground_signed_height=stats(local_h[inside]),
                                 native_support_replacement_accepted=False)
    # Diagnostic source panels: original mask, measured plane height, foreground.
    left = image.copy()
    cv2.drawContours(left, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 0), 2)
    hue = np.clip((h+.035)/.07*255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(hue[:, None], cv2.COLORMAP_TURBO)[:, 0]
    heat = image//3
    heat[yy, xx] = colors
    heat[~(cv2.dilate(mask.astype(np.uint8), np.ones((201, 201), np.uint8)) > 0)] //= 3
    contour = image//3
    contour[yy[fg], xx[fg]] = colors[fg]
    panels = [left, heat, contour]
    for panel, label in zip(panels, ['Rectified source + automatic mask', 'Measured height to global plane: blue -35mm / red +35mm', 'Same heights inside measured mask only']):
        cv2.rectangle(panel, (0, 0), (1280, 36), (0, 0, 0), -1)
        cv2.putText(panel, f'e037 f{frame}: {label}', (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    collage = np.concatenate([cv2.resize(panel, (640, 400)) for panel in panels], axis=1)
    cv2.imwrite(str(OUT / f'frame-{frame:03d}.jpg'), collage)
    report['rows'].append(row)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(row), flush=True)
for cap in caps:
    cap.release()
