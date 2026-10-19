"""Cross-modal diagnostic of retained own-object poses; no fitting or promotion."""
from pathlib import Path
import datetime
import hashlib
import json
import sys

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0] = [str(S / 'mediapipe-cpu-deps-v1'), str(R / 'scripts/track3')]
import cv2
import numpy as np
import trimesh
from source_hand_depth import front_surface_depth

cv2.setNumThreads(1)
O = S / 'e037-retained-pose-grip-audit-v2'
O.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pool_path = S / 'plane-aligned-pose-pools-v7/report.json'
selected_path = S / 'plane-aligned-pose-temporal-path-v4/report.json'
pool, selected = [json.loads(p.read_text()) for p in (pool_path, selected_path)]
assert sha(pool_path) == selected['candidate_pool_report_sha256']
frame = 50
retained = pool['frames'][frame]
assert retained['frame'] == frame
picked = selected['selected_pool_indices'][frame]
P = S / 'plane-aligned-native-reference-v1/frozen'
manifest = json.loads((P / 'manifest.json').read_text())
mesh_path = P / manifest['objects'][0]['mesh']
assert sha(mesh_path) == manifest['files'][manifest['objects'][0]['mesh']]
mesh = trimesh.load(mesh_path, force='mesh', process=False)
assert isinstance(mesh, trimesh.Trimesh)
B = S / 'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:
    K = z['intrinsics'].copy()
    original_hand_flags = {k: z[k][frame].tolist() for k in z.files if k in ('hand_valid', 'hand_direct_observed', 'hand_metric_observed')}
hand_path = S / 'e037-source-hand-object-registration-v1/registered-proxy-and-source-samples.npz'
with np.load(hand_path) as z:
    joints = z['hand_joints_C'].copy()
tip_ids = [4, 8, 12, 16, 20]
tips = joints[tip_ids]
depth_path = S / 'fast-stereo-cpu-full-e037-v1/000050-depth.npz'
with np.load(depth_path) as z:
    depth, valid_depth = z['depth_m'].copy(), z['left_right_consistent'].copy()
mask_path = S / 'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/000050.png'
source_mask = cv2.imread(str(mask_path), 0) > 0
image_path = S / 'e037-fresh-source-hand-request-v1/images/000050.png'
source = cv2.imread(str(image_path))
valid = source_mask & valid_depth & np.isfinite(depth) & (depth > 0)
pixels = np.argwhere(valid)[:, ::-1]
assert len(pixels) >= 256
# Fixed, diagnostic source pixels shared across all poses; never an optimization set.
sampled = pixels[np.linspace(0, len(pixels) - 1, 256).astype(int)]
estimated_z = depth[sampled[:, 1], sampled[:, 0]]

def project(v):
    return np.c_[K[0, 0] * v[:, 0] / v[:, 2] + K[0, 2], K[1, 1] * v[:, 1] / v[:, 2] + K[1, 2]]

rows, panels = [], []
poses = []
for index, candidate in enumerate(retained['retained_candidates']):
    pose = np.asarray(candidate['camera_from_object'], dtype=float)
    assert np.max(abs(pose[3] - [0, 0, 0, 1])) < 1e-10
    assert np.max(abs(pose[:3, :3].T @ pose[:3, :3] - np.eye(3))) < 1e-8
    assert abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-8
    vertices = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
    assert (vertices[:, 2] > 0).all()
    cam_mesh = trimesh.Trimesh(vertices, mesh.faces, process=False)
    _, distances, _ = trimesh.proximity.closest_point_naive(cam_mesh, tips)
    front = front_surface_depth(vertices, mesh.faces, sampled, K)
    hit = np.isfinite(front)
    uv = project(vertices)
    raster = np.zeros(source_mask.shape, np.uint8)
    for tri in mesh.faces:
        cv2.fillConvexPoly(raster, np.rint(uv[tri]).astype(int), 255)
    mask = raster > 0
    row = dict(
        retained_pool_index=index,
        selected_in_frozen_path=(index == picked),
        source_fit_reused_unchanged=candidate['source_fit'],
        inferred_tip_distances_to_own_visual_mesh_m=dict(zip(map(str, tip_ids), distances.tolist())),
        inferred_tip_min_distance_m=float(distances.min()),
        inferred_tip_median_distance_m=float(np.median(distances)),
        inferred_tip_max_distance_m=float(distances.max()),
        raster_source_inferred_mask_coverage=float((mask & source_mask).sum() / source_mask.sum()),
        raster_source_inferred_mask_IoU=float((mask & source_mask).sum() / (mask | source_mask).sum()),
        cached_estimated_visible_depth_sample_count=len(sampled),
        visible_depth_model_hits=int(hit.sum()),
        visible_depth_model_misses=int((~hit).sum()),
        hit_only_estimated_surface_depth_median_abs_error_m=float(np.median(abs(front[hit] - estimated_z[hit]))) if hit.any() else None,
        hit_only_estimated_surface_depth_p95_abs_error_m=float(np.percentile(abs(front[hit] - estimated_z[hit]), 95)) if hit.any() else None,
        silhouette_depth_misses_not_removed_from_report=True,
    )
    rows.append(row)
    poses.append(pose)
    panel = source.copy()
    contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, (0, 200, 255), 1)
    for tip, p in zip(tip_ids, project(tips)):
        cv2.circle(panel, tuple(np.rint(p).astype(int)), 3, (0, 0, 255), 1)
        cv2.putText(panel, str(tip), tuple(np.rint(p + [3, -3]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)
    panel = panel[160:390, 350:900]
    label = f'pool {index}, selected={index == picked}, inferred tip median={np.median(distances)*1000:.1f}mm'
    cv2.putText(panel, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 20, 240), 1)
    panels.append(panel)
assert len(rows) == retained['qualified_candidate_count']
np.savez_compressed(O / 'fixed-diagnostic-inputs.npz', pixels_C=sampled, estimated_surface_depth_m=estimated_z, inferred_hand_tip_ids=np.asarray(tip_ids), inferred_hand_tip_positions_C=tips, retained_camera_from_object=np.asarray(poses))
cv2.imwrite(str(O / 'retained-source-poses-and-inferred-tips.png'), np.vstack(panels))
report = dict(
    schema='v2d.retained-own-object-pose-grip-diagnostic.v1',
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    producer_sha256=sha(Path(__file__)),
    frame=frame,
    source_hashes={str(p.relative_to(S)): sha(p) for p in [pool_path, selected_path, P / 'manifest.json', mesh_path, B, hand_path, depth_path, mask_path, image_path]},
    retained_candidates=len(rows),
    selected_pool_index=picked,
    rows=rows,
    original_source_hand_flags=original_hand_flags,
    source_hand_estimated_and_still_unqualified=True,
    tip_points_are_not_contact_labels=True,
    source_mask_and_depth_are_estimates=True,
    limited_scope='Existing four retained frame50 poses only; this cannot exclude other poses or distinguish source hand versus shape errors.',
    existing_fitters_or_temporal_selection_rerun=False,
    numerical_or_observation_gates_relaxed=False,
    source_or_native_assets_modified=False,
    new_candidate_promoted=False,
    native_contact_certificate_invalidated_by_any_future_mutation=True,
    required_native_reference_contact_gate_m=.001,
    physical_task_accepted=False,
    new_GPU_work=False,
    new_PPO_steps=0,
)
for rel, expected in report['source_hashes'].items():
    assert sha(S / rel) == expected, rel
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
