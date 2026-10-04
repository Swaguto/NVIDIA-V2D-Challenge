"""GT-free candidate-selection diagnostic on frozen pot/lid observations.

This is an experiment, not a production default. It retains every refined PCA
candidate and compares geometric ranking with projected visible-mask support.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--observations', type=Path, required=True)
parser.add_argument('--masks', type=Path, required=True)
parser.add_argument('--depth', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--frames', default='all', help='all frozen frames, or comma-separated source IDs')
args = parser.parse_args()
args.repo = Path(__file__).resolve().parents[2]
if args.out.exists():
    parser.error('fresh output required')
os.environ['V2D_PUBLIC'] = str(args.data)
os.environ['V2D_CALIB'] = str(args.data / 'meta/camera_calibration.json')
sys.path.insert(0, str(args.repo / 'scripts/track3'))
import cv2
import numpy as np
import torch
import register_poses as reg
from scipy.spatial import cKDTree
from pose_support import projected_support

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

raw = args.observations.read_bytes()
baseline = json.loads(raw)
if (baseline['gt_use'] != 'none' or baseline['coordinate_frame'] != 'rectified_camera'
        or baseline['camera_frame'] != 'ego_cam_c'
        or set(baseline['objects']) != {'white_pot', 'white_pot_lid'}):
    raise ValueError('expected GT-free rectified-left pot/lid observations')
if sha(args.data / 'meta/camera_calibration.json') != baseline['calibration_sha256']:
    raise ValueError('calibration changed')
if baseline['params'].get('cloud_filter') != {
        'method': 'local_density', 'mad_threshold': 3.5, 'iterations': 1,
        'minimum_spacing_ratio': 2.0, 'neighbors': 11}:
    raise ValueError('expected frozen local-density baseline with boundary floor')
for item in baseline['inputs']:
    if sha(args.data / item['path']) != item['sha256']:
        raise ValueError('RGB/CAD input changed')
torch.set_num_threads(4)
rig = reg.load_rig()
cad = torch.tensor(reg.load_cad('white_pot', 6000, 0), dtype=torch.float32, device='cuda')
report = copy.deepcopy(baseline)
report.update(frames=[], producer='all-PCA candidate visible-mask support experiment',
              observation_input_sha256=hashlib.sha256(raw).hexdigest(),
              native_source_sha256_lf=hashlib.sha256((args.repo/'scripts/track3/register_poses.py').read_text().encode()).hexdigest(),
              worker_sha256_lf=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest())
report['params']['candidate_selection'] = {
    'all_coarse_seeds_refined': True, 'selection': 'projected_support',
    'occluder_depth_margin_m': .03, 'pixel_tolerance_px': 3,
    'pixel_distance_clip_px': 20,
    'gt_use': 'none', 'objects_refit': ['white_pot'],
    'remaining_object_fits': 'frozen baseline',
    'frame_selection': args.frames,
    'mask_exclusion_policy': 'other object only if not clearly behind CAD',
    'support_source_sha256_lf': hashlib.sha256((args.repo/'scripts/track3/pose_support.py').read_text().encode()).hexdigest()}
args.out.mkdir(parents=True)

for original in baseline['frames']:
    frame = original['frame_index']
    if args.frames != 'all' and frame not in {int(x) for x in args.frames.split(',')}:
        continue
    episode = original['episode']
    folder = f'e{episode:03d}/{frame:06d}'
    dp = args.depth/folder/'depth.npz'
    if sha(dp) != original['depth_sha256']:
        raise ValueError('frozen depth changed')
    loaded = np.load(dp)
    depth, valid = loaded['depth'], loaded['valid']
    masks = {}
    for name in baseline['objects']:
        mp = args.masks/folder/f'{name}_mask.png'
        if sha(mp) != original['views'][rig.left]['objects'][name]['mask_sha256']:
            raise ValueError('frozen mask changed')
        masks[name] = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0
    mask = masks['white_pot']
    points = reg.backproject_full(depth, valid, rig.P1)
    cloud = reg.trim_cloud(reg.voxel_downsample(points[mask.ravel() & valid.ravel()], .004))
    dst = torch.tensor(cloud, dtype=torch.float32, device='cuda')
    cc, dc = cad.mean(0), dst.mean(0)
    centered_cad, centered_dst = cad-cc, dst-dc
    generator = torch.Generator(device='cpu').manual_seed(0)
    def sub(x, n):
        if len(x) <= n:
            return x
        return x[torch.randperm(len(x), generator=generator)[:n].to(x.device)]
    coarse_cad, coarse_dst = sub(centered_cad, 900), sub(centered_dst, 900)
    zero = torch.zeros(3, dtype=cad.dtype, device=cad.device)
    seeds = []
    for R in reg._init_rotations(coarse_cad, coarse_dst):
        seeds.append(reg._icp(coarse_cad, coarse_dst, R, zero, iters=10))
    seeds.sort(key=lambda result: result[3])
    fine_cad = sub(centered_cad, 4000)
    candidates = []
    for rank, (seed_scale, R0, t0, coarse_residual) in enumerate(seeds):
        scale, R, t, residual = reg._icp(fine_cad, centered_dst, R0, t0, iters=50)
        t = dc + t - scale * (R @ cc)
        moved = (scale*(cad@R.T)+t).cpu().numpy()
        support = projected_support(moved, rig.P1[:3,:3], mask, depth, valid, masks['white_pot_lid'])
        row = {'coarse_rank':rank, 'coarse_residual_m':float(coarse_residual),
               'residual_m':float(residual), 'scale':float(scale),
               'rotation_wxyz':reg._mat_to_wxyz(R.cpu().numpy()).tolist(),
               'translation':t.cpu().numpy().tolist(),
               **support,
               'cloud_to_cad_p90_m':float(np.percentile(cKDTree(moved).query(cloud)[0],90))}
        candidates.append(row)
        print(json.dumps({'frame':frame,'candidate':row}),flush=True)
    native = min(candidates[:3], key=lambda x:x['residual_m'])
    geometric = min(candidates, key=lambda x:x['residual_m'])
    accepted = [c for c in candidates if np.isfinite(c['residual_m']) and c['residual_m']<=.05]
    selected = min(accepted, key=lambda x:x['projected_support_score'])
    row = copy.deepcopy(original)
    row['candidate_diagnostics'] = {'white_pot':candidates,
        'native_selected_coarse_rank':native['coarse_rank'],
        'all_geometry_selected_coarse_rank':geometric['coarse_rank'],
        'projected_support_selected_coarse_rank':selected['coarse_rank'],
        'native_replay_translation_error_m':float(np.linalg.norm(np.asarray(native['translation'])-np.asarray(original['fits']['white_pot']['translation'])))}
    row['fits']['white_pot'] = {**selected,'status':'observed','reason':None,'cloud_points':len(cloud)}
    report['frames'].append(row)
    (args.out/'probe_results.partial.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'frame':frame,'native_rank':native['coarse_rank'],'geometry_rank':geometric['coarse_rank'],'support_rank':selected['coarse_rank']}),flush=True)
(args.out/'probe_results.json').write_text(json.dumps(report,indent=2))
if args.observations.read_bytes() != raw:
    raise ValueError('frozen observations changed during inference')
print('Frozen candidate-selection diagnostic; no pose GT read.',flush=True)
