"""Attach observed CAD to a frozen public-episode review without reference poses."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from issue6.manifest import load, write_json
from issue6.track3_adapter import observed_object_state

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', required=True)
parser.add_argument('--source-root', type=Path, default=Path('/workspace/issue6-full-inputs'))
parser.add_argument('--input-root', type=Path, default=Path('/workspace/issue6-final-inputs'))
parser.add_argument('--cad-root', type=Path, default=Path('/workspace/cad-final'))
parser.add_argument('--protocol-receipt', type=Path, default=Path('/workspace/pr32-evidence/all20-final-protocol.json'))
args = parser.parse_args()
original_selector = Path('/workspace/pr32-evidence/detection-source-v1/wearer_selection.py')
selector = original_selector if original_selector.exists() else Path(__file__).with_name('wearer_selection.py')
protocol = {
    'selector_sha256': hashlib.sha256(selector.read_bytes()).hexdigest(),
    'cad_reinit_every': 0, 'root_iterations': 50, 'smooth_iterations': 300,
    'fixed_hand_shape': True, 'contact_threshold_m': .02, 'bridge_frames': 15,
    'wearer_manual_exclusions': False, 'gt_object_poses_used': False,
}
evaluation = {'purpose': 'frozen_all_episode_review', 'tuning_allowed': False,
              'configuration_sha256': hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()}
write_json(args.protocol_receipt, {'evaluation': evaluation, 'protocol': protocol,
                                 'selector_source': str(selector),
                                 'decision': 'Continuous CAD retained after the development episode 12 matched control worsened with periodic re-registration; same settings for every episode.'})
for episode in map(int, args.episodes.split(',')):
    label = f'e{episode:03d}'
    source = args.source_root / label / 'manifest.json'
    manifest = copy.deepcopy(load(source))
    manifest['evaluation'] = evaluation
    for frame in manifest['frames']:
        frame['objects'] = {
            obj['id']: observed_object_state(args.cad_root / label, episode, obj['id'],
                                              frame['id'], manifest['color_camera']['id'])
            for obj in manifest['objects']
        }
    # Native camera estimation consumes these observed masks. The previous
    # control pipelines must finish before their input manifests are replaced.
    write_json(source, manifest)
    destination = args.input_root / label / 'manifest.json'
    write_json(destination, manifest)
    load(destination)
    print(json.dumps({'episode': episode, 'manifest': str(destination),
                      'cad_root': str(args.cad_root), 'evaluation': evaluation}), flush=True)
