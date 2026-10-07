"""Run the actual upstream root and smooth optimizers on complete episodes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from issue6.backend import final_dynhamr_result

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, required=True)
args = parser.parse_args()
root = Path(f'/workspace/temporal-hands/e{args.episode:03d}')
manifest = json.loads(Path(f'/workspace/issue6-full-inputs/e{args.episode:03d}/manifest.json').read_text())
rate = (len(manifest['frames'])-1)/(manifest['frames'][-1]['timestamp_s']-manifest['frames'][0]['timestamp_s'])
command = [sys.executable, 'run_opt.py', 'data=video_vipe', 'run_opt=True', 'run_vis=False',
           'run_prior=False', 'is_static=False', 'model.opt_scale=False', 'model.opt_cams=False',
           f'data.seq=e{args.episode:03d}', f'data.root={root}', f'data.sources.images={root / "frames"}',
           f'data.sources.tracks={root / "tracks"}', f'data.sources.shots={root / "shots.json"}',
           f'data.sources.cameras={root / "cameras"}', 'data.start_idx=0',
           'data.track_ids=0-1',
           f'data.end_idx={len(manifest["frames"])}', f'fps={rate}',
           f'paths.DATA_DIR=/workspace/hamer-weights/_DATA/data',
           f'hydra.run.dir={root / "fit"}']
started = time.time()
subprocess.run(command, cwd='/workspace/Dyn-HaMR/dyn-hamr', check=True)
final = final_dynhamr_result(root / 'fit')
initial, = (root / 'fit/init').glob('*world_results.npz')
fixed_shape = os.environ.get('V2D_OPT_HAND_SHAPE', '1') == '0'
with np.load(initial, allow_pickle=False) as initial_data, np.load(final, allow_pickle=False) as final_data:
    for key in ('trans', 'root_orient', 'pose_body', 'betas', 'cam_R', 'cam_t'):
        assert np.isfinite(final_data[key]).all(), f'Nonfinite final fit: {key}'
    if fixed_shape:
        assert np.array_equal(initial_data['betas'], final_data['betas']), 'Fixed shape changed during fit'
receipt = {'episode': args.episode, 'command': command, 'elapsed_s': time.time()-started,
           'dynhamr_revision': 'fa9cd7412c205fd15ee4139c8caacf79bf6167e6',
           'root_iterations': 50, 'smooth_iterations': 300,
           'BMC': 'disabled by upstream default zero loss weight, no placeholder assets',
           'HMP': 'disabled by upstream default', 'camera_scale': 'fixed to measured stereo',
           'gt_object_poses_used': False, 'final_checkpoint': str(final),
           'hand_shape': 'fixed sequence-average HaMeR shape' if fixed_shape else 'optimized',
           'fixed_shape_exactly_preserved': fixed_shape,
           'runtime_optimizer_sha256': hashlib.sha256(Path('/workspace/Dyn-HaMR/dyn-hamr/optim/optimizers.py').read_bytes()).hexdigest()}
(root / 'fit_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
