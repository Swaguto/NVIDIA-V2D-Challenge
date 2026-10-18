"""Conservative linear polygons inside per-source-point reprojection cones.

These constraints preserve each original projected fit error. They are source
fitted MANO observations, not hidden labels or a contact/grasp annotation.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

S=Path(__file__).resolve().parents[1]
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
from v2d_dex3_flashchord.kinematics import URDFKinematics

P=S/'scored-cohort-14-results/policy-e037/frozen'
OUT=S/'root-source-projection-constraints-v1'
OUT.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((P/'manifest.json').read_text())
stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz',allow_pickle=False) as z:
    b={k:z[k].copy() for k in z.files}
with np.load(P/'targets.npz',allow_pickle=False) as z:
    q=z['robot_qpos'].copy()
K=b['intrinsics']
rows=[];columns=[];values=[];lower=[];upper=[];points=0
for si,side in enumerate(['left','right']):
    kin=URDFKinematics(P/f'robot/urdfs/g1/dex3_{side}.urdf')
    for frame in range(len(q)):
        if not(b['hand_valid'][frame,si] and b['hand_directly_observed'][frame,si]):
            continue
        source=b['hand_joints_world'][frame,si,[0,4,8,12]]
        if not np.isfinite(source).all():
            continue
        fk=kin.poses(q[frame,si,:7],dict(zip(m['joint_names'][side],q[frame,si,7:],strict=True)))
        native=np.stack([q[frame,si,:3],fk[f'{side}_thumb_tip'][:3,3],fk[f'{side}_index_tip'][:3,3],fk[f'{side}_middle_tip'][:3,3]])
        world=np.einsum('ni,ij->nj',native,stage,optimize=False)
        c=b['camera_c2w'][frame]
        camera=np.einsum('ni,ij->nj',world-c[:3,3],c[:3,:3],optimize=False)
        observed=np.einsum('ni,ij->nj',source-c[:3,3],c[:3,:3],optimize=False)
        A=np.einsum('ji,kj->ik',c[:3,:3],stage,optimize=False)
        # A = camera_R.T @ stage_R.T: verify finite-difference transform.
        np.testing.assert_allclose(A,c[:3,:3].T.dot(stage.T),atol=1e-12,rtol=0)
        for p,obs in zip(camera,observed,strict=True):
            assert min(p[2],obs[2])>.01
            uv=np.array([K[0,0]*obs[0]/obs[2]+K[0,2],K[1,1]*obs[1]/obs[2]+K[1,2]])
            M=np.array([[K[0,0],0,K[0,2]-uv[0]],[0,K[1,1],K[1,2]-uv[1]]])
            error_vector=np.einsum('ij,j->i',M,p,optimize=False)
            error=np.linalg.norm(error_vector)/p[2]
            angle=np.arctan2(error_vector[1],error_vector[0])
            for theta in angle+np.pi/16+np.arange(16)*2*np.pi/16:
                normal=np.array([np.cos(theta),np.sin(theta)])
                coeff=np.einsum('i,ij,jk->k',normal,M,A,optimize=False)-error*np.cos(np.pi/16)*A[2]
                rhs=error*np.cos(np.pi/16)*p[2]-np.dot(normal,error_vector)
                assert rhs>=-1e-9,'Control point is outside its conservative polygon'
                index=len(lower);rows.extend([index]*3);columns.extend([frame*6+si*3+j for j in range(3)]);values.extend(coeff)
                lower.append(-np.inf);upper.append(rhs)
            points+=1
A=sparse.csc_matrix((values,(rows,columns)),shape=(len(lower),len(q)*6))
np.savez_compressed(OUT/'constraints.npz',data=A.data,indices=A.indices,indptr=A.indptr,shape=A.shape,lower=lower,upper=upper)
report=dict(schema='v2d.root-source-projection-constraints.v1',episode=37,
            frozen_manifest_sha256=sha(P/'manifest.json'),constraints_sha256=sha(OUT/'constraints.npz'),
            producer_sha256=sha(Path(__file__)),source_observed_points=points,linear_constraints=len(lower),
            source_confidence_modified=False,source_inputs_modified=False,physical_task_accepted=False)
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
