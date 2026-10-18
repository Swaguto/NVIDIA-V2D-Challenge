"""Independent source-projection check for the contact-only wrist proposal."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

S=Path(__file__).resolve().parents[1]
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
from v2d_dex3_flashchord.kinematics import URDFKinematics

P=S/'scored-cohort-14-results/policy-e037'
inputs=P/'frozen'
proposal=S/'temporal-root-contact-probe-v3/inferred-robot-command-candidate.npz'
OUT=S/'temporal-root-source-check-v1'
OUT.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((inputs/'manifest.json').read_text())
stage=np.asarray(json.loads((inputs/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
bundle=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(bundle,allow_pickle=False) as z:
    b={k:z[k].copy() for k in z.files}
with np.load(inputs/'targets.npz',allow_pickle=False) as z:
    q=z['robot_qpos'].copy()
    np.testing.assert_allclose(np.einsum('ij,tj->ti',stage,b['object_0_world'][:,:3,3],optimize=False),z['object_poses'][:,0,:3],atol=1e-8,rtol=0)
with np.load(proposal,allow_pickle=False) as z:
    candidate=z['robot_qpos'].copy()
    np.testing.assert_array_equal(z['frame_indices'],b['frame_indices'])
K=b['intrinsics']


def project(points,frame):
    c=b['camera_c2w'][frame]
    camera=np.einsum('ni,ij->nj',points-c[:3,3],c[:3,:3],optimize=False)
    assert np.isfinite(camera).all() and (camera[:,2]>.01).all()
    return np.c_[K[0,0]*camera[:,0]/camera[:,2]+K[0,2],K[1,1]*camera[:,1]/camera[:,2]+K[1,2]]


report=dict(schema='v2d.native-root-source-projection-diagnostic.v1',episode=37,
            candidate_sha256=sha(proposal),bundle_sha256=sha(bundle),producer_sha256=sha(Path(__file__)),
            rows=[],source_observations_modified=False,source_confidence_modified=False,
            protocol='Compare source fitted MANO wrist/thumb/index/middle tip projections on directly observed valid frames; unknown metric depth remains unknown. Both median and p95 pixel error must not worsen for source-screen qualification.',
            true_hand_pixel_ground_truth_known=False,trajectory_promoted=False,physical_task_accepted=False)
for si,side in enumerate(['left','right']):
    kin=URDFKinematics(inputs/f'robot/urdfs/g1/dex3_{side}.urdf')
    before=[];after=[];metric_before=[];metric_after=[];movements=[]
    for frame in range(len(q)):
        if not(b['hand_valid'][frame,si] and b['hand_directly_observed'][frame,si]):
            continue
        source=b['hand_joints_world'][frame,si,[0,4,8,12]]
        if not np.isfinite(source).all():
            continue
        projected=project(source,frame)
        fits=[]
        for commands in [q,candidate]:
            fk=kin.poses(commands[frame,si,:7],dict(zip(m['joint_names'][side],commands[frame,si,7:],strict=True)))
            native=np.stack([commands[frame,si,:3],fk[f'{side}_thumb_tip'][:3,3],fk[f'{side}_index_tip'][:3,3],fk[f'{side}_middle_tip'][:3,3]])
            world=np.einsum('ni,ij->nj',native,stage,optimize=False)
            fits.append(project(world,frame))
        before.extend(np.linalg.norm(fits[0]-projected,axis=1))
        after.extend(np.linalg.norm(fits[1]-projected,axis=1))
        movements.extend(np.linalg.norm(fits[1]-fits[0],axis=1))
        if b['hand_metric_valid'][frame,si]:
            metric_before.extend(np.linalg.norm(fits[0]-projected,axis=1))
            metric_after.extend(np.linalg.norm(fits[1]-projected,axis=1))
    def stats(x):
        x=np.asarray(x)
        return dict(points=len(x),median_px=float(np.median(x)),p95_px=float(np.quantile(x,.95)),maximum_px=float(x.max())) if len(x) else dict(points=0)
    a,c=stats(before),stats(after)
    report['rows'].append(dict(hand=side,control=a,candidate=c,screen_movement=stats(movements),
                               metric_supported_control=stats(metric_before),metric_supported_candidate=stats(metric_after),
                               projection_screen_passed=c['median_px']<=a['median_px'] and c['p95_px']<=a['p95_px']))
report['all_source_projection_screens_passed']=all(r['projection_screen_passed'] for r in report['rows'])
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
