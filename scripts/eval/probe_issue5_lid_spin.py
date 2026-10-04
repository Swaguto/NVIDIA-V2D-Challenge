"""GT-free, fixed-translation/axis-tilt spin sweep of the textured lid CAD.

Optional appearance diagnostic on frozen observations, not a tracking default.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
for name in ('data', 'observations', 'masks', 'depth', 'out'):
    parser.add_argument('--' + name, type=Path, required=True)
parser.add_argument('--score', choices=['highpass_ncc', 'gradient_ncc'], default='gradient_ncc')
parser.add_argument('--minimum-gap', type=float, default=.1)
parser.add_argument('--maximum-cost', type=float, default=.75)
args = parser.parse_args()
if args.out.exists():
    parser.error('fresh output required')
repo = Path(__file__).resolve().parents[2]
os.environ['V2D_PUBLIC'] = str(args.data)
os.environ['V2D_CALIB'] = str(args.data / 'meta/camera_calibration.json')
sys.path.insert(0, str(repo / 'scripts/track3'))
import cv2
import numpy as np
import torch
import scipy
from scipy.spatial.transform import Rotation
import register_poses as reg
from lid_spin import build_template, geometric_residual_function, spin_rotation, sweep_appearance
from v2d.world_calib.vm.fit_ego_rig import load_mesh


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


started = time.perf_counter()
raw = args.observations.read_bytes()
baseline = json.loads(raw)
if (baseline['gt_use'] != 'none' or baseline['coordinate_frame'] != 'rectified_camera'
        or baseline['camera_frame'] != 'ego_cam_c'
        or baseline['objects'] != ['white_pot', 'white_pot_lid']):
    raise ValueError('GT-free rectified-left pot/lid observations required')
if baseline['params'].get('cloud_filter') != {
        'method': 'local_density', 'mad_threshold': 3.5, 'iterations': 1,
        'minimum_spacing_ratio': 2.0, 'neighbors': 11}:
    raise ValueError('expected frozen local-density baseline')
if (baseline['params']['scale'] != 1 or baseline['params']['cad_samples'] != 6000
        or baseline['params']['seed'] != 0):
    raise ValueError('expected scale-one, seed-zero, 6000-CAD-sample baseline')
for item in baseline['inputs']:
    if sha(args.data / item['path']) != item['sha256']:
        raise ValueError('RGB/CAD input changed')
if sha(args.data / 'meta/camera_calibration.json') != baseline['calibration_sha256']:
    raise ValueError('calibration changed')
rig = reg.load_rig()
mesh = load_mesh(args.data / 'mesh/white_pot_lid/white_pot_lid.glb')
cad = reg.load_cad('white_pot_lid', 6000, 0)
template, texture, template_info = build_template(
    mesh.vertices, mesh.faces, mesh.visual.to_color().vertex_colors)
args.out.mkdir(parents=True)
cv2.imwrite(str(args.out / 'canonical_texture.png'), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(args.out / 'canonical_roi.png'), template.roi.astype('uint8') * 255)
np.savez_compressed(args.out / 'canonical_template.npz', gray=template.gray, roi=template.roi,
                    axis=template.axis, tangent1=template.tangent1,
                    tangent2=template.tangent2, origin=template.origin, scale=template.scale)
report = copy.deepcopy(baseline)
report.update(frames=[], producer='fixed-axis textured-lid spin appearance diagnostic',
              observation_input_sha256=hashlib.sha256(raw).hexdigest(),
              worker_sha256_lf=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),
              lid_spin_source_sha256_lf=hashlib.sha256((repo / 'scripts/track3/lid_spin.py').read_text().encode()).hexdigest(),
              native_source_sha256_lf=hashlib.sha256((repo / 'scripts/track3/register_poses.py').read_text().encode()).hexdigest(),
              parent_elapsed_s=baseline.get('elapsed_s'),
              parent_environment=baseline.get('environment'),
              environment={'python': platform.python_version(), 'torch': torch.__version__,
                           'numpy': np.__version__, 'scipy': scipy.__version__,
                           'opencv': cv2.__version__, 'device': 'cpu'},
              models_loaded_by_this_worker=[])
report['params']['lid_spin'] = {
    **template_info, 'coarse_step_deg': 5, 'fine_step_deg': .5,
    'translation_fixed': True, 'axis_tilt_fixed': True, 'score': args.score,
    'highpass_sigma_px': 3, 'gradient_sigma_px': 1,
    'minimum_separated_peak_gap': args.minimum_gap,
    'maximum_appearance_cost': args.maximum_cost,
    'gt_use': 'none', 'template_sha256': sha(args.out / 'canonical_template.npz'),
    'geometric_sample': 'native fine CAD indices; symmetric trimmed 0.6 CPU cKDTree',
    'proposal_display': 'observed/proposed/applied panels with labels'}
report['limits'] = [*baseline['limits'],
    'Planar CAD annulus approximation; front-face +X convention is specific to this scanned lid.',
    'Appearance confidence is experimental and is not independently calibrated.',
    'Sparse independent frames; no temporal tracking or world-camera trajectory is estimated.']
for original in baseline['frames']:
    episode, frame = original['episode'], original['frame_index']
    row = copy.deepcopy(original)
    fit = row['fits']['white_pot_lid']
    if fit['status'] != 'observed':
        row['lid_spin_diagnostic'] = {'update_applied': False, 'uncertainty_reason': 'baseline_not_observed'}
        report['frames'].append(row)
        continue
    R0 = Rotation.from_quat(np.asarray(fit['rotation_wxyz'])[[1, 2, 3, 0]]).as_matrix()
    translation = np.asarray(fit['translation'])
    images = [reg.read_frame(args.data / f'videos/chunk-000/observation.images.{cam}/episode_{episode:06d}.mp4', frame)
              for cam in (rig.left, rig.right)]
    image = rig.rectify_pair(*images)[0]
    mp = args.masks / f'e{episode:03d}/{frame:06d}/white_pot_lid_mask.png'
    if sha(mp) != row['views'][rig.left]['objects']['white_pot_lid']['mask_sha256']:
        raise ValueError('mask changed')
    mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0
    dp = args.depth / f'e{episode:03d}/{frame:06d}/depth.npz'
    if sha(dp) != row['depth_sha256']:
        raise ValueError('frozen depth changed')
    with np.load(dp) as loaded:
        depth, valid = loaded['depth'], loaded['valid']
    cloud = reg.trim_cloud(reg.voxel_downsample(
        reg.backproject_full(depth, valid, rig.P1)[mask.ravel() & valid.ravel()], .004))
    generator = torch.Generator(device='cpu').manual_seed(0)
    torch.randperm(len(cad), generator=generator)
    if len(cloud) > 900:
        torch.randperm(len(cloud), generator=generator)
    fine_cad = cad[torch.randperm(len(cad), generator=generator)[:4000].numpy()]
    geometry = geometric_residual_function(fine_cad, cloud, translation)
    diagnostic, proposed_R, applied_R, panels = sweep_appearance(
        template, image, mask, rig.P1[:3, :3], R0, translation, method=args.score,
        minimum_gap=args.minimum_gap, maximum_cost=args.maximum_cost)
    baseline_geometry, proposed_geometry = geometry(R0), geometry(proposed_R)
    for sample in diagnostic['coarse_curve']:
        sample['geometric_residual_m'] = geometry(spin_rotation(R0, template.axis, sample['angle_deg']))
    if diagnostic['update_applied'] and proposed_geometry > .05:
        diagnostic.update(update_applied=False, uncertainty_reason='proposed_geometric_residual_gate')
        applied_R = R0
    diagnostic.update(recomputed_baseline_residual_m=baseline_geometry,
                      proposed_residual_m=proposed_geometry,
                      geometric_cost_method='same native fine CAD indices / symmetric trimmed 0.6 / CPU cKDTree',
                      native_baseline_residual_difference_m=abs(baseline_geometry - fit['residual_m']))
    fit['baseline_geometric_residual_m'] = fit['residual_m']
    fit['rotation_wxyz'] = reg._mat_to_wxyz(applied_R).tolist()
    fit['appearance_spin_deg'] = diagnostic['selected']['angle_deg'] % 360 if diagnostic['update_applied'] else 0.
    fit['spin_observability'] = 'appearance_supported' if diagnostic['update_applied'] else 'uncertain_baseline_retained'
    fit['residual_m'] = proposed_geometry if diagnostic['update_applied'] else baseline_geometry
    if fit['residual_m'] > .05:
        fit.update(status='rejected', reason='post_spin_residual_gate')
    row['lid_spin_diagnostic'] = diagnostic
    crop, proposed_image, applied_image = panels
    # A geometrically rejected proposal also uses the baseline in the applied panel.
    if diagnostic['uncertainty_reason'] == 'proposed_geometric_residual_gate':
        _, _, _, baseline_panels = sweep_appearance(template, image, mask, rig.P1[:3, :3], R0,
            translation, method=args.score, minimum_gap=3, maximum_cost=0)
        applied_image = baseline_panels[2]
    preview = []
    for label, panel in [('observed', crop), ('proposed', proposed_image), ('applied' if diagnostic['update_applied'] else 'baseline retained', applied_image)]:
        if panel.ndim == 2:
            panel = cv2.cvtColor(np.uint8(np.clip(panel, 0, 255)), cv2.COLOR_GRAY2BGR)
        panel = cv2.copyMakeBorder(panel, 24, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(panel, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        preview.append(panel)
    cv2.imwrite(str(args.out / f'e{episode:03d}_{frame:06d}_appearance.jpg'), np.concatenate(preview, axis=1))
    report['frames'].append(row)
    (args.out / 'probe_results.partial.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'episode': episode, 'frame': frame, 'selected': diagnostic['selected'],
                      'gap': diagnostic['separated_peak_cost_gap'], 'update_applied': diagnostic['update_applied'],
                      'uncertainty_reason': diagnostic['uncertainty_reason'], 'residual_m': fit['residual_m']}), flush=True)
report['elapsed_s'] = time.perf_counter() - started
(args.out / 'probe_results.json').write_text(json.dumps(report, indent=2))
if args.observations.read_bytes() != raw:
    raise ValueError('predictions changed during inference')
print('Frozen appearance spin diagnostic; no pose GT read.', flush=True)
