"""Run the actual upstream root and smooth optimizers on complete episodes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

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
receipt = {'episode': args.episode, 'command': command, 'elapsed_s': time.time()-started,
           'dynhamr_revision': 'fa9cd7412c205fd15ee4139c8caacf79bf6167e6',
           'root_iterations': 50, 'smooth_iterations': 300,
           'BMC': 'disabled by upstream default zero loss weight, no placeholder assets',
           'HMP': 'disabled by upstream default', 'camera_scale': 'fixed to measured stereo',
           'gt_object_poses_used': False}
(root / 'fit_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
