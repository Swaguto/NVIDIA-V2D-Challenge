"""Measure final aligned-hand motion in the estimated ViPE world frame."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, required=True)
args = parser.parse_args()
root = Path(f'/workspace/temporal-hands/e{args.episode:03d}')
run = Path(os.environ.get('V2D_ISSUE6_RUN_ROOT','/workspace/issue6-full-runs/calibrated-temporal')) / f'e{args.episode:03d}'
hands = json.loads((run / 'align/hands.json').read_text())
with np.load(root / 'cameras/vipe_c2w.npz', allow_pickle=False) as cameras:
    c2w = cameras['data']
result = {
    'episode': args.episode,
    'source': str(run / 'align/hands.json'),
    'basis': 'Estimated ViPE world frame. This measures consistency, not independent ground-truth accuracy.',
    'samples': 'Consecutive, directly observed same-track frame triples; metric-valid subset also reported.',
    'hands': {},
}
for right in (False, True):
    rows = sorted((r for r in hands if bool(r['is_right']) == right), key=lambda r:r['frame_id'])
    side = 'right' if right else 'left'
    frame = np.array([r['frame_id'] for r in rows])
    times = np.array([r['timestamp_s'] for r in rows])
    tracks = np.array([r['track_id'] for r in rows])
    joints = np.array([r['joints_camera_m'] for r in rows])
    assert len(rows) >= 3 and np.isfinite(joints).all()
    world = np.einsum('nij,nkj->nki',c2w[frame,:3,:3],joints)+c2w[frame,None,:3,3]
    velocity = np.diff(world,axis=0)/np.diff(times)[:,None,None]
    acceleration = np.linalg.norm(np.diff(velocity,axis=0)/((times[2:]-times[:-2])/2)[:,None,None],axis=-1)
    consecutive = (np.diff(frame)[:-1]==1)&(np.diff(frame)[1:]==1)
    same_track = (tracks[:-2]==tracks[1:-1])&(tracks[1:-1]==tracks[2:])
    supported = np.array([r['metric_valid'] for r in rows],bool)
    metric_triples = supported[:-2]&supported[1:-1]&supported[2:]
    values = {}
    for name, valid in [('direct_observations',consecutive&same_track),
                        ('metric_valid_observations',consecutive&same_track&metric_triples)]:
        sample = acceleration[valid]
        values[name] = {'triples':int(valid.sum())}
        if len(sample):
            assert np.isfinite(sample).all()
            values[name].update({
                'wrist_acceleration_m_s2_median':float(np.median(sample[:,0])),
                'wrist_acceleration_m_s2_p95':float(np.percentile(sample[:,0],95)),
                'all_joint_acceleration_m_s2_median':float(np.median(sample)),
                'all_joint_acceleration_m_s2_p95':float(np.percentile(sample,95)),
            })
    result['hands'][side] = {'records':len(rows),'metric_valid_records':int(supported.sum()),**values}
(root / 'aligned_consistency.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
