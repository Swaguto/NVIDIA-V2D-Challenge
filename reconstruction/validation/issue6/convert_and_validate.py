"""Validate the actual temporal result through the repaired issue-6 pipeline."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import os

# Keep numerical libraries within a bounded share of the pod's CPU allocation.
# This also propagates to the native alignment subprocess.
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'

import numpy as np
from issue6 import backend
from issue6.manifest import load, write_json
from issue6.pipeline import execute
from v2d.hand_alignment.lib.dynhamr_to_hamer_tracks import convert_dynhamr_to_hamer_tracks
from temporal_record_gate import rejection_reason
from native_identity_guard import identity_conflicts

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, required=True)
args = parser.parse_args()
root = Path(f'/workspace/temporal-hands/e{args.episode:03d}')
manifest_path = Path(os.environ.get('V2D_ISSUE6_INPUT_ROOT', '/workspace/issue6-full-inputs')) / f'e{args.episode:03d}' / 'manifest.json'
manifest = load(manifest_path)
result = backend.final_dynhamr_result(root / 'fit')
axis = backend.dynhamr_frame_axis(result, manifest['frames'])
intrinsics = root / 'intrinsics.json'
write_json(intrinsics, manifest['color_camera'])
converted = root / 'converted'
convert_dynhamr_to_hamer_tracks(str(result), str(converted), str(intrinsics), mano_assets_root='/workspace/mano')
canonical = root / 'original'
backend._copy_records(converted, canonical, axis, remap=True)
gate = json.loads((root / 'observation_gate.json').read_text())['direct_observation']
identity_audit = json.loads((root / 'wearer_candidates.json').read_text())
conflicts = identity_conflicts(identity_audit, manifest['color_camera']['width'],
                              manifest['color_camera']['height'])
conflicting_observations = {(item['frame_id'], item['hand']) for item in conflicts}
# The optimizer estimates occluded frames. Keep them in the raw optimizer output,
# while contact/overlay inputs only include directly observed hands.
kept = 0
rejected = []
for path in sorted((canonical / 'records').glob('*/*.json')):
    record = json.loads(path.read_text())
    side = 'right' if record['is_right'] else 'left'
    record['temporal_source'] = str(result)
    record['directly_observed'] = gate[side][record['frame_idx']]
    reason = rejection_reason(record, record['directly_observed'], manifest['color_camera'])
    if reason is None and (record['frame_idx'], side) in conflicting_observations:
        reason = 'opposite_hand_identity_conflict'
    if reason is not None:
        rejected.append({'frame_id': record['frame_idx'], 'track_id': record['track_id'],
                         'is_right': record['is_right'], 'reason': reason,
                         'camera': record['camera'] if reason == 'invalid_virtual_camera' else None})
        path.unlink()
    else:
        write_json(path, record)
        kept += 1
record_gate = {'version': 2, 'kept': kept, 'rejected': rejected,
               'identity_conflicts': conflicts,
               'scope': 'Invalid or unobserved native estimates remain unknown; raw fitted and converted records are preserved',
               'implementation_sha256': {
                   filename: hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
                   for filename in ('convert_and_validate.py', 'temporal_record_gate.py', 'native_identity_guard.py')}}
write_json(root / 'temporal_record_gate.json', record_gate)
manifest['hand_source'] = {'backend': 'import', 'runtime': 'native',
        'records_dir': str(canonical / 'records'), 'masks_dir': str(root / 'masks'),
        'tracks_path': str(root / 'hand_tracks.json'), 'mano_assets_root': '/workspace/mano',
        'world_results': str(result),
        'native_record_gate': record_gate,
        'model_files': ['/workspace/hamer-weights/_DATA/hamer_ckpts/checkpoints/hamer.ckpt',
                        '/workspace/pr32-models/sam2.1_hiera_large.pt', '/workspace/pr32-models/detector.pt']}
destination = manifest_path.parent / 'manifest.temporal.json'
write_json(destination, manifest)
loaded = load(destination)
out = Path(os.environ.get('V2D_ISSUE6_RUN_ROOT', '/workspace/issue6-full-runs/calibrated-temporal')) / f'e{args.episode:03d}'
started = time.time()
for stage in ('validate', 'hands', 'depth', 'align', 'contacts', 'report'):
    execute(loaded, out, stage)
    print(f'Episode {args.episode}: {stage} complete', flush=True)
write_json(root / 'validation_receipt.json', {'episode': args.episode, 'selected_temporal_result': str(result),
          'selected_frame_axis': [f['id'] for f in axis], 'direct_records': kept,
          'elapsed_s': time.time()-started, 'manifest': str(destination), 'run': str(out)})
