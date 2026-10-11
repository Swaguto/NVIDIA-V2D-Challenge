"""Compare reconstructed hand targets and Dex3 states in physical units.

Fixed acceptance limits: 0.1 mm position, 0.001 rad robot rotation/joint angle.
These are engineering reproduction tolerances, not ground-truth accuracy claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def maximum_position_error(a, b):
    if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise ValueError('Position schema or finite mask differs')
    finite = np.isfinite(a).all(axis=-1)
    return float(np.linalg.norm(a[finite]-b[finite], axis=-1).max(initial=0))


def compare(reference, candidate):
    metrics = {}; paths = {}
    for label, root in [('reference',reference),('candidate',candidate)]:
        paths[label] = {name:sha(root/name) for name in ('bundle/episode.npz','retarget/retarget.npz')}
    with np.load(reference/'bundle/episode.npz',allow_pickle=False) as a, np.load(candidate/'bundle/episode.npz',allow_pickle=False) as b:
        for key in ('frame_indices','timestamps_s','hand_valid','hand_directly_observed'):
            if not np.array_equal(a[key],b[key]):raise ValueError('Bundle differs: '+key)
        metrics['hand_joint_position_m'] = maximum_position_error(a['hand_joints_world'],b['hand_joints_world'])
        metrics['hand_vertex_position_m'] = maximum_position_error(a['hand_vertices_world'],b['hand_vertices_world'])
    with np.load(reference/'retarget/retarget.npz',allow_pickle=False) as a, np.load(candidate/'retarget/retarget.npz',allow_pickle=False) as b:
        for key in ('frame_indices','timestamps_s','robot_valid','source_hand_valid'):
            if not np.array_equal(a[key],b[key]):raise ValueError('Robot differs: '+key)
        valid = a['robot_valid']; x,y = a['robot_qpos'],b['robot_qpos']
        if x.shape != y.shape or not np.array_equal(np.isnan(x),np.isnan(y)):
            raise ValueError('Robot schema or missing states differ')
        if not np.isfinite(x[valid]).all() or not np.isfinite(y[valid]).all():
            raise ValueError('Nonfinite supported robot state')
        x,y = x[valid],y[valid]
        metrics['robot_root_position_m'] = maximum_position_error(x[:,:3],y[:,:3])
        q,r=x[:,3:7],y[:,3:7]
        qn,rn=np.linalg.norm(q,axis=-1),np.linalg.norm(r,axis=-1)
        if np.any(qn<1e-8) or np.any(rn<1e-8):raise ValueError('Zero quaternion')
        cosine=np.abs(np.sum((q/qn[:,None])*(r/rn[:,None]),axis=-1))
        metrics['robot_root_rotation_rad']=float((2*np.arccos(np.clip(cosine,0,1))).max(initial=0))
        metrics['robot_finger_angle_rad']=float(np.abs(x[:,7:]-y[:,7:]).max(initial=0))
    tolerances={k:(0.0001 if k.endswith('_m') else 0.001) for k in metrics}
    passed=all(metrics[k] <= tolerances[k] for k in metrics)
    return {'status':'motion_matches' if passed else 'failed','metrics':metrics,'tolerances':tolerances,
            'files':paths,'scope':'numerical reproduction, not physical accuracy; fixed limits declared before comparison'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','out'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    try:report=compare(args.reference,args.candidate)
    except Exception as exc:report={'status':'failed','error':str(exc)}
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    raise SystemExit(0 if report['status']=='motion_matches' else 1)
