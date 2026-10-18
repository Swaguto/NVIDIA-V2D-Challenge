"""CPU feasibility probe at three previously recorded worst contact knots.

Root translation, local wrist rotation and seven finger angles are optimized
together against exact native distances and original source projections. This
does not supply a full trajectory, force closure, policy or training approval.
"""
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

S=Path(__file__).resolve().parents[1]
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
from audit_hand_reference_contacts import canonical_id

P=S/'scored-cohort-14-results/policy-e037/frozen'
OUT=S/'articulated-source-contact-sample-verification-v1'
OUT.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
native_proof=S/'local-native-contact-control-v1/report.json'
assert json.loads(native_proof.read_text())['all_full_reference_controls_passed']
manifest=json.loads((P/'manifest.json').read_text())
for name,h in manifest['files'].items():
    assert sha(P/name)==h
model_path=S/'local-native-contact-control-v1/native-model.mjb'
model=mujoco.MjModel.from_binary_path(str(model_path))
data=mujoco.MjData(model)
bodies=[model.body(i).name for i in range(model.nbody)]
joints=[model.joint(i).name for i in range(model.njnt)]
with np.load(P/'targets.npz',allow_pickle=False) as z:
    original={k:z[k].copy() for k in z.files}
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz',allow_pickle=False) as z:
    source={k:z[k].copy() for k in z.files}
stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
K=source['intrinsics']
binding={}
for si,side in enumerate(['left','right']):
    wrist=[canonical_id(joints,f'{side}_wrist_{a}') for a in ['x','y','z','rotvec']]
    fingers=[canonical_id(joints,n) for n in manifest['joint_names'][side]]
    root=ET.parse(P/f'robot/urdfs/g1/dex3_{side}.urdf').getroot()
    limits={j.attrib['name']:j.find('limit').attrib for j in root.findall('joint') if j.attrib['type']!='fixed'}
    binding[side]=dict(si=si,wrist=wrist,fingers=fingers,
        dofs=np.r_[[int(model.jnt_dofadr[j]) for j in wrist[:3]],np.arange(int(model.jnt_dofadr[wrist[3]]),int(model.jnt_dofadr[wrist[3]])+3),[int(model.jnt_dofadr[j]) for j in fingers]].astype(int),
        points=[canonical_id(bodies,n) for n in [f'{side}_hand_palm_link',f'{side}_thumb_tip',f'{side}_index_tip',f'{side}_middle_tip']],
        hand_geoms=[g for g,b in enumerate(model.geom_bodyid) if bodies[b].startswith('dex3_'+side+'_')],
        finger_lower=np.array([float(limits[n]['lower']) for n in manifest['joint_names'][side]]),
        finger_upper=np.array([float(limits[n]['upper']) for n in manifest['joint_names'][side]]))
target_bodies={canonical_id(bodies,'wooden_spoon_object'):'wooden_spoon',canonical_id(bodies,'support'):'support'}
target_geoms=[g for g,b in enumerate(model.geom_bodyid) if b in target_bodies]
object_bid=next(b for b,n in target_bodies.items() if n=='wooden_spoon')
object_adr=int(model.jnt_qposadr[model.body_jntadr[object_bid]])
scales=np.r_[np.full(3,.01),np.full(10,.1)]
from v2d_dex3_flashchord.kinematics import URDFKinematics

proposal=S/'articulated-source-contact-samples-v5/sample-command-proposals.npz'
with np.load(proposal,allow_pickle=False) as z:
    frames,sides,commands=z['frames'],z['sides'],z['robot_qpos']
report=dict(schema='v2d.articulated-sample-independent-verification.v1',producer_sha256=sha(Path(__file__)),candidate_sha256=sha(proposal),frozen_manifest_sha256=sha(P/'manifest.json'),source_bundle_sha256=sha(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'),native_model_sha256=sha(model_path),full_trajectory_or_velocity_certificate=False,source_or_confidence_changed=False,physics_stepped=False,new_training_steps=0,physical_task_accepted=False,frames=[])
kin={side:URDFKinematics(P/f'robot/urdfs/g1/dex3_{side}.urdf') for side in binding}
for frame,si,command in zip(frames,sides,commands,strict=True):
    frame,si=int(frame),int(si);side=['left','right'][si]
    candidate=original['robot_qpos'][frame].copy();candidate[si]=command
    data.qpos[:]=model.qpos0;obj=original['object_poses'][frame,0]
    data.qpos[object_adr:object_adr+3]=obj[:3];data.qpos[object_adr+3:object_adr+7]=obj[[6,3,4,5]]
    fk_errors=[]
    for hand,h in binding.items():
        q=candidate[h['si']]
        for col,j in enumerate(h['wrist'][:3]):data.qpos[int(model.jnt_qposadr[j])]=q[col]
        adr=int(model.jnt_qposadr[h['wrist'][3]]);data.qpos[adr:adr+4]=q[[6,3,4,5]]
        for j,v in zip(h['fingers'],q[7:],strict=True):data.qpos[int(model.jnt_qposadr[j])]=v
    mujoco.mj_forward(model,data)
    for hand,h in binding.items():
        q=candidate[h['si']];poses=kin[hand].poses(q[:7],dict(zip(manifest['joint_names'][hand],q[7:],strict=True)))
        for bid,name in enumerate(bodies):
            if not name.startswith('dex3_'+hand+'_'):continue
            names=[link for link in poses if name==link or name.endswith('_'+link)]
            assert len(names)==1,(name,names)
            pose=poses[names[0]]
            error=max(float(abs(data.xpos[bid]-pose[:3,3]).max()),float(abs(data.xmat[bid].reshape(3,3)-pose[:3,:3]).max()))
            assert error<2e-6;fk_errors.append(error)
    c=source['camera_c2w'][frame]
    def project(world):
        camera=np.einsum('ni,ij->nj',world-c[:3,3],c[:3,:3],optimize=False)
        assert np.isfinite(camera).all() and (camera[:,2]>.01).all()
        return np.c_[K[0,0]*camera[:,0]/camera[:,2]+K[0,2],K[1,1]*camera[:,1]/camera[:,2]+K[1,2]]
    observed=project(source['hand_joints_world'][frame,si,[0,4,8,12]])
    errors=[]
    for q in [original['robot_qpos'][frame,si],command]:
        poses=kin[side].poses(q[:7],dict(zip(manifest['joint_names'][side],q[7:],strict=True)))
        points=np.stack([q[:3],poses[f'{side}_thumb_tip'][:3,3],poses[f'{side}_index_tip'][:3,3],poses[f'{side}_middle_tip'][:3,3]])
        world=np.einsum('ni,ij->nj',points,stage,optimize=False)
        errors.append(np.linalg.norm(project(world)-observed,axis=1))
    contacts=[]
    for c in data.contact[:data.ncon]:
        a,b=int(c.geom1),int(c.geom2)
        if (a in binding[side]['hand_geoms'] and b in target_geoms) or (b in binding[side]['hand_geoms'] and a in target_geoms):contacts.append(float(c.dist))
    max_pen=max([0.]+[-v for v in contacts])
    source_pass=bool((errors[1]<=errors[0]+1e-6).all() and np.median(errors[1])<=np.median(errors[0]) and np.quantile(errors[1],.95)<=np.quantile(errors[0],.95))
    report['frames'].append(dict(frame=frame,hand=side,verified_native_hand_bodies=len(fk_errors),maximum_fk_error=float(max(fk_errors)),control_pixel_errors=errors[0].tolist(),candidate_pixel_errors=errors[1].tolist(),source_projection_passed=source_pass,native_maximum_penetration_m=max_pen,native_contact_gate_passed=max_pen<=.001))
report['all_sample_checks_passed']=all(r['source_projection_passed'] and r['native_contact_gate_passed'] for r in report['frames'])
for name,h in manifest['files'].items():assert sha(P/name)==h
report['frozen_source_closure_verified_after']=True
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
