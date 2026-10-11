"""Run real HaMeR inference in an isolated installation without replacing fits."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime-root', type=Path, required=True)
parser.add_argument('--hand-root', type=Path, default=Path('/workspace/temporal-hands'))
parser.add_argument('--weights', type=Path, default=Path('/workspace/hamer-weights'))
parser.add_argument('--episode', type=int, required=True)
parser.add_argument('--frames', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists(), 'Preserve previous replay outputs before a new probe'
fork = args.runtime_root / 'Dyn-HaMR/third-party/hamer'
sys.path.insert(0, str(fork))
spec = importlib.util.spec_from_file_location('fresh_hamer', fork / 'run.py')
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
source = args.hand_root / f'e{args.episode:03d}/cleaned_detections.json'
saved = json.loads(source.read_text())
ids = [int(value) for value in args.frames.split(',')]
selected = [saved[index] for index in ids]
for row in selected:
    for side in ('left', 'right'):
        for suffix in ('bbox', 'keypoints'):
            field = f'{side}_{suffix}'
            if row[field] is not None:
                row[field] = np.asarray(row[field], dtype=float)
model, cfg = upstream.load_hamer(str(args.weights))
model = model.to('cuda').eval()
renderer = upstream.Renderer(cfg, faces=model.mano.faces)
args.output.mkdir(parents=True)
options = SimpleNamespace(rescale_factor=1.3, render=False,
                          res_folder=str(args.output / 'predictions.pkl'))
with torch.inference_mode():
    predictions = upstream.run_hamer_on_cleaned_bboxes(selected, model, cfg, renderer, options)
checks = []
for index, row in zip(ids, selected):
    prediction = predictions[row['img_path']]
    count = len(prediction['mano'])
    assert count > 0, f'No native model output for frame {index}'
    camera = np.asarray(prediction['cam_trans'])
    assert camera.shape == (count, 3) and np.isfinite(camera).all()
    checks.append({'original_frame_id': index, 'hands': count,
                   'source_image_sha256': digest(Path(row['img_path'])),
                   'camera_shape': list(camera.shape), 'camera_finite': True})
with (args.output / 'predictions.pkl').open('wb') as stream:
    pickle.dump(predictions, stream)
assets = [args.weights / '_DATA/hamer_ckpts/checkpoints/hamer.ckpt',
          args.weights / '_DATA/hamer_ckpts/model_config.yaml']
mano_inputs = sorted(Path(cfg.MANO.MODEL_PATH).rglob('*.pkl'))
assert mano_inputs, 'No actual MANO input found in the loaded model configuration'
assets.extend(mano_inputs)
receipt = {'status': 'passed', 'scope': 'Real HaMeR inference smoke replay; not all20 refitting or independent quality accuracy',
           'episode': args.episode, 'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'runtime_root': str(args.runtime_root), 'torch_version': torch.__version__,
           'upstream_revision': subprocess.check_output(['git', '-C', str(fork), 'rev-parse', 'HEAD'], text=True).strip(),
           'upstream_run_sha256': digest(fork / 'run.py'), 'probe_sha256': digest(Path(__file__)),
           'saved_detections_sha256': digest(source), 'checks': checks,
           'model_inputs': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)} for path in assets],
           'output_sha256': digest(args.output / 'predictions.pkl')}
(args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
