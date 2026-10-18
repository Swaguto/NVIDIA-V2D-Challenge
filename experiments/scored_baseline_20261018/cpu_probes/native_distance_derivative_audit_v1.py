"""Audit the first failed numerical stage before more articulated optimization.

No optimization, physics, source changes or training is performed.
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
OUT=S/'native-distance-derivative-audit-v1'
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
report=dict(schema='v2d.native-distance-derivative-audit.v1',producer_sha256=sha(Path(__file__)),native_model_sha256=sha(model_path),frozen_manifest_sha256=sha(P/'manifest.json'),source_or_reference_changed=False,physics_stepped=False,new_training_steps=0,cases=[])
for frame,side in [(18,'right'),(23,'right'),(64,'left')]:
    bind=binding[side]
    base=model.qpos0.copy()
    obj=original['object_poses'][frame,0]
    base[object_adr:object_adr+3]=obj[:3];base[object_adr+3:object_adr+7]=obj[[6,3,4,5]]
    for hand,h in binding.items():
        q=original['robot_qpos'][frame,h['si']]
        for col,j in enumerate(h['wrist'][:3]):base[int(model.jnt_qposadr[j])]=q[col]
        adr=int(model.jnt_qposadr[h['wrist'][3]]);base[adr:adr+4]=q[[6,3,4,5]]
        for j,v in zip(h['fingers'],q[7:],strict=True):base[int(model.jnt_qposadr[j])]=v
    pairs=[(a,b) for a in bind['hand_geoms'] for b in target_geoms]
    def evaluate(x):
        qpos=base.copy();v=np.zeros(model.nv);v[bind['dofs']]=x*scales
        mujoco.mj_integratePos(model,qpos,v,1.)
        data.qpos[:]=qpos;mujoco.mj_fwdPosition(model,data)
        return np.array([mujoco.mj_geomDistance(model,data,a,b,.1,None) for a,b in pairs])
    zero=np.zeros(13);before=evaluate(zero)
    contacts=[]
    for c in data.contact[:data.ncon]:
        if (int(c.geom1),int(c.geom2)) in pairs:
            pair=(int(c.geom1),int(c.geom2));normal=-np.array(c.frame[:3])
        elif (int(c.geom2),int(c.geom1)) in pairs:
            pair=(int(c.geom2),int(c.geom1));normal=np.array(c.frame[:3])
        else:continue
        point=np.array(c.pos);body=int(model.geom_bodyid[pair[0]])
        jp=np.zeros((3,model.nv));jr=jp.copy()
        mujoco.mj_jac(model,data,jp,jr,point,body)
        grad=np.einsum('i,ij,j->j',normal,jp[:,bind['dofs']],scales,optimize=False)
        contacts.append(dict(pair=pair,gap=float(c.dist),gradient=grad))
    runs={};jumps=[]
    for step in [1e-4,1e-5]:
        gradients=[]
        for index in range(13):
            dx=zero.copy();dx[index]=step
            minus=evaluate(-dx);plus=evaluate(dx)
            gradients.append((plus-minus)/(2*step))
            for pair_index in np.flatnonzero(abs(plus-minus)>.001):
                a,b=pairs[pair_index]
                jumps.append(dict(step=float(step),axis=index,pair=[a,b],geom_names=[model.geom(a).name,model.geom(b).name],baseline_gap_m=float(before[pair_index]),minus_gap_m=float(minus[pair_index]),plus_gap_m=float(plus[pair_index]),native_contact_present=any(v['pair']==(a,b) for v in contacts)))
        runs[step]=np.stack(gradients,axis=1)
    checks=[]
    for c in contacts:
        i=pairs.index(c['pair'])
        checks.append(dict(pair=list(c['pair']),native_gap_m=c['gap'],distance_query_m=float(before[i]),maximum_native_jacobian_error=float(abs(runs[1e-4][i]-c['gradient']).max()),maximum_step_sensitivity=float(abs(runs[1e-4][i]-runs[1e-5][i]).max())))
    report['cases'].append(dict(frame=frame,hand=side,queried_pairs=len(pairs),native_contact_pairs=len(contacts),query_discontinuity_records=jumps,native_contact_derivative_checks=checks,discontinuous_pair_count=len({tuple(j['pair']) for j in jumps}),discontinuous_native_contact_pairs=len({tuple(j['pair']) for j in jumps if j['native_contact_present']})))
for name,h in manifest['files'].items():assert sha(P/name)==h
report['frozen_source_closure_verified_after']=True
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{k:v for k,v in c.items() if k not in ['query_discontinuity_records','native_contact_derivative_checks']} for c in report['cases']]),flush=True)
