"""Run Dyn-HaMR's pinned enhanced YOLO/HaMeR tracker; save its actual detections."""
import argparse
import copy
import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import shutil
import sys

import numpy as np
import torch
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, required=True)
parser.add_argument('--exclude-reviewed-bystander-region', action='store_true')
args = parser.parse_args()
fork = Path('/workspace/Dyn-HaMR/third-party/hamer')
sys.path.insert(0, str(fork))
spec = importlib.util.spec_from_file_location('enhanced_hamer', fork / 'run.py')
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
source = Path(f'/workspace/issue6-full-inputs/e{args.episode:03d}')
out = Path(f'/workspace/temporal-hands/e{args.episode:03d}')
frames = out / 'frames'
frames.mkdir(parents=True, exist_ok=True)
for path in sorted((source / 'color').glob('*.png')):
    shutil.copy2(path, frames / path.name)
paths = sorted(frames.glob('*.png'))
detector = YOLO('/workspace/pr32-models/detector.pt').to('cuda')
if args.exclude_reviewed_bystander_region:
    assert args.episode == 22, 'Reviewed region applies only to episode 22'
    original_detector = detector
    candidates = []

    def detector(image, **kwargs):
        results = original_detector(image, **kwargs)
        result = results[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        excluded = (boxes[:, 2] < 400) & (boxes[:, 3] < 300)
        candidates.append({'frame': len(candidates), 'boxes': boxes.tolist(),
                           'confidence': result.boxes.conf.detach().cpu().tolist(),
                           'class': result.boxes.cls.detach().cpu().tolist(),
                           'excluded_bystander': excluded.tolist()})
        return [result[~excluded]]

raw = upstream.extract_raw_bboxes(paths, detector)
if args.exclude_reviewed_bystander_region:
    (out / 'reviewed_detector_candidates.json').write_text(json.dumps({
        'basis': 'Manual RGB review: upper-left background hands belong to a bystander; exclude before confidence selection for both sides. No ground-truth poses or contact labels used.',
        'excluded_region': 'candidate x2 < 400 and y2 < 300 in 1280x800 source',
        'frames': candidates}, indent=2)+'\n')


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


(out / 'raw_detections.json').write_text(json.dumps(plain(raw), indent=2) + '\n')
cleaned = upstream.clean_bbox_sequences(copy.deepcopy(raw))
(out / 'cleaned_detections.json').write_text(json.dumps(plain(cleaned), indent=2) + '\n')
model, cfg = upstream.load_hamer('/workspace/hamer-weights')
model = model.to('cuda').eval()
renderer = upstream.Renderer(cfg, faces=model.mano.faces)
options = SimpleNamespace(rescale_factor=1.3, render=False, res_folder=str(out / 'predictions.pkl'))
predictions = upstream.run_hamer_on_cleaned_bboxes(cleaned, model, cfg, renderer, options)
with (out / 'predictions.pkl').open('wb') as f:
    pickle.dump(predictions, f)
sys.path.insert(0, '/workspace/Dyn-HaMR/dyn-hamr')
from preproc.export_hamer import export_sequence_results
export_sequence_results(str(out / 'predictions.pkl'), str(out / 'tracks'), str(out / 'shots.json'))
print(json.dumps({'episode': args.episode, 'frames': len(paths),
                  'left_raw': sum(r['left_bbox'] is not None for r in raw),
                  'right_raw': sum(r['right_bbox'] is not None for r in raw),
                  'left_cleaned': sum(r['left_bbox'] is not None for r in cleaned),
                  'right_cleaned': sum(r['right_bbox'] is not None for r in cleaned)}, indent=2), flush=True)
