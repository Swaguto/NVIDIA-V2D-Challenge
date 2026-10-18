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
OUT=S/'native-plane-support-audit-v1'
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
records=[]
for frame,side in [(18,'right'),(23,'right'),(64,'left')]:
    bind=binding[side]
    data.qpos[:]=model.qpos0
    obj=original['object_poses'][frame,0]
    data.qpos[object_adr:object_adr+3]=obj[:3];data.qpos[object_adr+3:object_adr+7]=obj[[6,3,4,5]]
    for hand,h in binding.items():
        q=original['robot_qpos'][frame,h['si']]
        for col,j in enumerate(h['wrist'][:3]):data.qpos[int(model.jnt_qposadr[j])]=q[col]
        adr=int(model.jnt_qposadr[h['wrist'][3]]);data.qpos[adr:adr+4]=q[[6,3,4,5]]
        for j,v in zip(h['fingers'],q[7:],strict=True):data.qpos[int(model.jnt_qposadr[j])]=v
    mujoco.mj_fwdPosition(model,data)
    def vertices(geom):
        mesh=int(model.geom_dataid[geom]);adr=int(model.mesh_vertadr[mesh]);num=int(model.mesh_vertnum[mesh])
        return np.einsum('ni,ji->nj',model.mesh_vert[adr:adr+num],data.geom_xmat[geom].reshape(3,3),optimize=False)+data.geom_xpos[geom]
    pairs={(a,b) for a in bind['hand_geoms'] for b in target_geoms}
    for c in data.contact[:data.ncon]:
        a,b=int(c.geom1),int(c.geom2)
        if (a,b) in pairs:hand,target,normal=a,b,-np.array(c.frame[:3])
        elif (b,a) in pairs:hand,target,normal=b,a,np.array(c.frame[:3])
        else:continue
        hs=np.einsum('ni,i->n',vertices(hand),normal,optimize=False)
        ts=np.einsum('ni,i->n',vertices(target),normal,optimize=False)
        records.append(dict(frame=frame,side=side,geoms=[hand,target],geom_names=[model.geom(hand).name,model.geom(target).name],native_gap_m=float(c.dist),convex_projection_gap_m=float(hs.min()-ts.max()),hand_interval=[float(hs.min()),float(hs.max())],target_interval=[float(ts.min()),float(ts.max())],normal=normal.tolist(),normal_norm=float(np.linalg.norm(normal)),geom_margin=[float(model.geom_margin[hand]),float(model.geom_margin[target])],includemargin=float(c.includemargin)))
report=dict(schema='v2d.native-plane-support-audit.v1',producer_sha256=sha(Path(__file__)),native_model_sha256=sha(model_path),contacts=records,source_or_reference_changed=False,physics_stepped=False,new_training_steps=0)
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(records,indent=2))
