"""Compare the actual initial and final Dyn-HaMR geometry on observed frame triples."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/Dyn-HaMR/dyn-hamr')
from body_model import MANO
from body_model.utils import run_mano
from issue6.backend import dynhamr_frame_axis, final_dynhamr_result

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, required=True)
args = parser.parse_args()
root = Path(f'/workspace/temporal-hands/e{args.episode:03d}')
manifest = json.loads(Path(f'/workspace/issue6-full-inputs/e{args.episode:03d}/manifest.json').read_text())
final = final_dynhamr_result(root / 'fit')
initial, = (root / 'fit/init').glob('*_init_world_results.npz')
axis = dynhamr_frame_axis(final, manifest['frames'])
frame_ids = np.array([f['id'] for f in axis])
time = np.array([f['timestamp_s'] for f in axis])
gate = json.loads((root / 'observation_gate.json').read_text())['direct_observation']
result = {'episode': args.episode, 'initial': str(initial), 'final': str(final),
          'frame_ids': frame_ids.tolist(),
          'basis': 'Same upstream MANO implementation, using each checkpoint\'s own saved shape and pose. Estimated ViPE world frame; not independent ground-truth accuracy.',
          'samples': 'Consecutive triples with direct hand observations at all three frames', 'hands': {}}
for label, path in [('initial', initial), ('final', final)]:
    with np.load(path, allow_pickle=False) as data:
        b, t = data['trans'].shape[:2]
        model = MANO('/workspace/hamer-weights/_DATA/data/mano', batch_size=b*t, pose2rot=True,
                     num_hand_joints=15, mean_params='/workspace/hamer-weights/_DATA/data/mano_mean_params.npz',
                     create_body_pose=False).to('cpu')
        tensors = {key: torch.as_tensor(data[key], dtype=torch.float32, device='cpu')
                   for key in ('trans', 'root_orient', 'pose_body', 'is_right', 'betas')}
        with torch.no_grad():
            geom = run_mano(model, tensors['trans'], tensors['root_orient'],
                            tensors['pose_body'].reshape(b,t,45), tensors['is_right'], tensors['betas'])
        joints = geom['joints'].cpu().numpy()
        for h in range(b):
            side = 'right' if data['is_right'][h].mean() > .5 else 'left'
            direct = np.asarray(gate[side])[frame_ids]
            valid = direct[:-2] & direct[1:-1] & direct[2:]
            velocity = np.diff(joints[h], axis=0) / np.diff(time)[:,None,None]
            acceleration = np.linalg.norm(np.diff(velocity,axis=0) / ((time[2:]-time[:-2])/2)[:,None,None],axis=-1)
            values = acceleration[valid]
            assert len(values) and np.isfinite(values).all()
            result['hands'].setdefault(side,{})[label] = {
                'observed_triples': int(valid.sum()),
                'wrist_acceleration_m_s2_median': float(np.median(values[:,0])),
                'wrist_acceleration_m_s2_p95': float(np.percentile(values[:,0],95)),
                'all_joint_acceleration_m_s2_median': float(np.median(values)),
                'all_joint_acceleration_m_s2_p95': float(np.percentile(values,95))}
        del model, tensors, geom
(root / 'temporal_consistency.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
