"""CPU sensitivity probe: separate in-plane size from inferred back-surface thickness.

Not a production reconstruction method; existing poses, masks and metric flags
remain unchanged. No candidate is installed and no physics is stepped.
"""

def _explicit_product(a, b):
    a, b = (np.asarray(a), np.asarray(b))
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite matrix input')
    patterns = {(1, 1): 'i,i->', (2, 1): 'ij,j->i', (1, 2): 'i,ij->j', (2, 2): 'ij,jk->ik'}
    result = np.einsum(patterns[a.ndim, b.ndim], a, b, optimize=False)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite explicit matrix result')
    return result
from pathlib import Path
import hashlib, json, sys, time
import cv2, numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
OUT = S / 'spoon-thickness-grid-v2'
OUT.mkdir()
sys.path[:0] = [str(R / 'scripts/track3'), str(R / 'reconstruction/modules/v2d_stereo_depth/lib')]
from stereo_depth import load_rig, stereo_depth
from select_pose_hypothesis import observed_fit
from measured_pose_refinement import finite_support_overlap
from foundationpose_track import pose_rejection
from camera_prior import transport
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
geo = ROOT / 'geometry/e037/wooden_spoon'
mesh = trimesh.load(geo / 'mesh.glb', force='mesh', process=False)
vertices = np.asarray(mesh.vertices)
faces = np.asarray(mesh.faces)
input_hashes = {p.name: sha(p) for p in geo.iterdir() if p.is_file()}
bundle = ROOT / 'bundles/e037'
rig = load_rig(str(S / 'dataset/track_3/public/meta/camera_calibration.json'))
K = rig.P1[:3, :3]
with np.load(bundle / 'episode.npz', allow_pickle=False) as f:
    camera = f['camera_c2w'].copy()
    np.testing.assert_allclose(K, f['intrinsics'], atol=1e-06)
world_plane = np.asarray(json.loads((ROOT / 'support/e037/report.json').read_text())['plane_world'])
with np.load(ROOT / 'support/e037/inferred-support-sheet.npz', allow_pickle=False) as f:
    sheet = f['vertices_world_m'].copy()
pose_root = ROOT / 'source/e037/poses/e037/ego_object_poses/e037/wooden_spoon/object_to_cam/ego_cam_c'

def pose(frame):
    p = pose_root / f'{frame:06d}.json'
    if not p.is_file():
        return None
    j = json.loads(p.read_text())
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(j['rotation'])[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = j['translation']
    return T

def plane_camera(frame):
    P = camera[frame]
    return np.r_[_explicit_product(P[:3, :3].T, world_plane[:3]), world_plane[3] + _explicit_product(world_plane[:3], P[:3, 3])]
initial = pose(0)
pc = plane_camera(0)
normal = _explicit_product(initial[:3, :3].T, pc[:3])
normal /= np.linalg.norm(normal)
projected = vertices - vertices.mean(0)
projected -= np.outer(_explicit_product(projected, normal), normal)
u = np.linalg.eigh(_explicit_product(projected.T, projected))[1][:, -1]
u -= normal * np.dot(u, normal)
u /= np.linalg.norm(u)
B = np.column_stack([u, np.cross(normal, u), normal])
assert np.allclose(_explicit_product(B.T, B), np.eye(3)) and np.linalg.det(B) > 0
arms = {}
for size in [1.0, 1.3, 1.6]:
    for thickness in [1.0, 0.8, 0.6, 0.4, 0.2]:
        v = _explicit_product(_explicit_product(vertices, B) * np.array([size, size, thickness]), B.T)
        seed_height = float(np.min(_explicit_product(_explicit_product(v, initial[:3, :3].T) + initial[:3, 3], pc[:3]) + pc[3]))
        shift = normal * max(0, 0.002 - seed_height)
        for correction, offset in [('none', np.zeros(3)), ('seed-normal-offset', shift)]:
            name = f's{size:.1f}-t{thickness:.1f}-{correction}'
            arms[name] = dict(vertices=v, tree=cKDTree(v), offset=offset, size=size, thickness=thickness, rows=[])
caps = [cv2.VideoCapture(str(S / f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{cam}/episode_000037.mp4')) for cam in [rig.left, rig.right]]
started = time.monotonic()
for frame in [0, 25, 50, 75, 100, 125, 150]:
    if time.monotonic() - started > 240:
        raise TimeoutError('CPU geometry sensitivity deadline')
    T = pose(frame)
    assert T is not None
    pc = plane_camera(frame)
    P = camera[frame]
    footprint = _explicit_product(sheet - P[:3, 3], P[:3, :3])
    images = []
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        assert ok
        images.append(image)
    depth, valid = stereo_depth(rig, *images)
    mask = cv2.imread(str(ROOT / f'source/e037/masks/wooden_spoon/{frame:06d}.png'), 0)
    previous = pose(frame - 1) if frame else None
    prior = transport(previous, np.linalg.inv(camera[frame - 1]), np.linalg.inv(camera[frame])) if previous is not None else None
    for name, arm in arms.items():
        correction = np.eye(4)
        correction[:3, 3] = arm['offset']
        current = _explicit_product(T, correction)
        v = arm['vertices']
        fit = observed_fit(v, faces, arm['tree'], current, depth, valid, mask > 127, K)
        overlap = finite_support_overlap(v, faces, current, pc, footprint)
        bounds = np.stack([v.min(0), v.max(0)])
        rejection, _ = pose_rejection(current, depth, valid, mask, K, bounds, True, None if prior is None else _explicit_product(prior, correction))
        arm['rows'].append(dict(frame=frame, fit=fit, finite_support_overlap=overlap, original_depth_and_jump_rejection=rejection))
    print('Scored source frame', frame, flush=True)
for cap in caps:
    cap.release()
rows = []
for name, arm in arms.items():
    rows.append(dict(arm=name, in_plane_scale=arm['size'], normal_scale=arm['thickness'], fixed_body_offset_m=arm['offset'].tolist(), mean_source_quality=float(np.mean([x['fit']['quality'] for x in arm['rows']])), mean_iou=float(np.mean([x['fit']['silhouette_iou'] for x in arm['rows']])), mean_depth_fit_m=float(np.mean([x['fit']['median_measured_vertex_distance_m'] for x in arm['rows']])), source_gate_passes=sum((x['original_depth_and_jump_rejection'] is None for x in arm['rows'])), sampled_visual_support_intersections=sum((x['finite_support_overlap'] is not None and x['finite_support_overlap']['minimum_height_m'] < -0.001 for x in arm['rows'])), rows=arm['rows']))
assert input_hashes == {p.name: sha(p) for p in geo.iterdir() if p.is_file()}
r = dict(schema='v2d.spoon-thickness-source-sensitivity.v2', episode=37, body_basis=B.tolist(), rows=rows, source_frames=[0, 25, 50, 75, 100, 125, 150], source_geometry_files_unchanged=True, source_poses_masks_or_confidence_modified=False, body_origin_changed=False, physics_stepped=False, native_contacts_assessed=False, candidate_promoted=False, physical_task_accepted=False, new_gpu_lease_started=False, producer_sha256=sha(Path(__file__)), wall_seconds=time.monotonic() - started)
(OUT / 'report.json').write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps([{k: v for k, v in x.items() if k != 'rows'} for x in sorted(rows, key=lambda x: x['mean_source_quality'], reverse=True)[:8]], indent=2), flush=True)
