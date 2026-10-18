"""Private CPU diagnostic of between-knot geometry, without physics steps.

Linear positions/fingers and shortest-arc quaternion SLERP at <=2ms are a
proposed reference interpolation, not a claim about the policy's actual motion.
A discrete sweep is not a continuous-collision or grasp-force certificate.
"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation,Slerp
S=Path(__file__).resolve().parents[1]
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
from audit_hand_reference_contacts import canonical_id
P=S/'plane-aligned-articulated-native-validation-v1'
OUT=S/'plane-aligned-reference-sweep-v1';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'schema':'v2d.inferred-reference-discrete-sweep.v1','producer_sha256':sha(Path(__file__)),'candidate_manifest_sha256':sha(P/'frozen/manifest.json'),'physics_stepped':False,'new_training_steps':0,'physical_task_accepted':False,'continuous_collision_certified':False,'actual_policy_motion_assessed':False,'contact_forces_assessed':False,'source_observations_created_between_knots':False,'tolerance_m':.001}
save=lambda:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
started=time.monotonic();save()
try:
 v=json.loads((P/'report.json').read_text());assert v['complete_static_source_contact_motion_stage_passed']
 manifest=json.loads((P/'frozen/manifest.json').read_text())
 for n,h in manifest['files'].items():assert sha(P/'frozen'/n)==h
 with np.load(P/'frozen/targets.npz',allow_pickle=False) as z:q=z['robot_qpos'].copy();o=z['object_poses'].copy();times=z['timestamps_s'].copy()
 assert o.shape==(151,1,7) and q.shape==(151,2,14)
 m=mujoco.MjModel.from_binary_path(str(P/'native-model.mjb'));state=mujoco.MjData(m)
 joints=[m.joint(i).name for i in range(m.njnt)];bodies=[m.body(i).name for i in range(m.nbody)]
 obj=canonical_id(bodies,'wooden_spoon_object');sup=canonical_id(bodies,'support');oj=int(m.body_jntadr[obj]);oa=int(m.jnt_qposadr[oj])
 assert m.jnt_type[oj]==mujoco.mjtJoint.mjJNT_FREE
 sides=['left','right'];wa={};fa={};hand={}
 for si,side in enumerate(sides):
  w=[canonical_id(joints,f'{side}_wrist_{a}') for a in ['x','y','z','rotvec']]
  assert all(m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE for j in w[:3]) and m.jnt_type[w[3]]==mujoco.mjtJoint.mjJNT_BALL
  wa[side]=[int(m.jnt_qposadr[j]) for j in w]
  f=[canonical_id(joints,n) for n in manifest['joint_names'][side]]
  assert all(m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE for j in f)
  fa[side]=[int(m.jnt_qposadr[j]) for j in f]
  for bi,n in enumerate(bodies):
   if n.startswith(f'dex3_{side}_'):hand[bi]=side
 # Use each exact source endpoint once, and ceil(dt/2ms)-1 interiors per interval.
 sample_times=np.r_[np.concatenate([np.linspace(a,b,int(np.ceil((b-a)/.002))+1)[:-1] for a,b in zip(times[:-1],times[1:])]),times[-1]]
 def linear(values):
  flat=values.reshape(len(times),-1)
  return np.stack([np.interp(sample_times,times,c) for c in flat.T],axis=1).reshape((len(sample_times),)+values.shape[1:])
 qi=linear(q);oi=linear(o)
 for si in range(2):qi[:,si,3:7]=Slerp(times,Rotation.from_quat(q[:,si,3:7]))(sample_times).as_quat()
 oi[:,0,3:7]=Slerp(times,Rotation.from_quat(o[:,0,3:7]))(sample_times).as_quat()
 assert np.isfinite(qi).all() and np.isfinite(oi).all() and np.diff(sample_times).max()<=.002+1e-10
 rows=[];max_depth={};failures=[];knot_worst=0.;close_object={s:[] for s in sides};near_tips={s:{} for s in sides}
 for ix,t in enumerate(sample_times):
  state.qpos[:]=m.qpos0
  state.qpos[oa:oa+3]=oi[ix,0,:3];state.qpos[oa+3:oa+7]=oi[ix,0,[6,3,4,5]]
  for si,side in enumerate(sides):
   for col,a in enumerate(wa[side][:3]):state.qpos[a]=qi[ix,si,col]
   a=wa[side][3];state.qpos[a:a+4]=qi[ix,si,[6,3,4,5]]
   state.qpos[fa[side]]=qi[ix,si,7:]
  mujoco.mj_forward(m,state)
  gaps={};contacts=[]
  for c in state.contact[:state.ncon]:
   a,b=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);pair=None
   if a in hand and b in [obj,sup]:pair=hand[a]+'/'+('spoon' if b==obj else 'support');hb=a;tb=b
   elif b in hand and a in [obj,sup]:pair=hand[b]+'/'+('spoon' if a==obj else 'support');hb=b;tb=a
   elif {a,b}=={obj,sup}:pair='spoon/support';hb=None;tb=None
   if pair is None:continue
   gap=float(c.dist);assert np.isfinite(gap);gaps[pair]=min(gaps.get(pair,0.),gap)
   if hb is not None and tb==obj and gap<=.002:
    key=bodies[hb];near_tips[hand[hb]][key]=near_tips[hand[hb]].get(key,0)+1
   contacts.append({'pair':pair,'bodies':[bodies[a],bodies[b]],'gap_m':gap})
  depth=max([-v for v in gaps.values()]+[0.])
  knot=bool(np.min(abs(times-t))<1e-9)
  if knot:knot_worst=max(knot_worst,depth)
  for pair,gap in gaps.items():max_depth[pair]=max(max_depth.get(pair,0.),-gap)
  if depth>.001:failures.append({'sample':ix,'time_s':float(t),'interval':min(int(np.searchsorted(times,t,side='right')-1),149),'maximum_penetration_m':depth,'contacts':contacts})
  for side in sides:
   if any(c['pair']==side+'/spoon' and c['gap_m']<=.002 for c in contacts):close_object[side].append(float(t))
  rows.append({'sample':ix,'time_s':float(t),'source_knot':knot,'contacts':contacts})
 (OUT/'contacts.json').write_text(json.dumps({'frames':rows},indent=2)+'\n')
 assert knot_worst<1e-9,'Sweep setup must reproduce the fresh static knot control'
 report.update(samples=len(sample_times),interior_samples=sum(not r['source_knot'] for r in rows),source_knots=sum(r['source_knot'] for r in rows),maximum_sample_spacing_s=float(np.diff(sample_times).max()),maximum_penetration_by_pair_m=max_depth,failed_samples=len(failures),failures=failures,static_knot_control_penetration_m=knot_worst,discrete_interpolated_contact_gate_passed=not failures,near_spoon_contact_samples={s:len(close_object[s]) for s in sides},near_spoon_bodies=near_tips,contacts_sha256=sha(OUT/'contacts.json'))
 for n,h in manifest['files'].items():assert sha(P/'frozen'/n)==h
 report['frozen_closure_verified_after']=True
except BaseException as error:report['error']=repr(error);raise
finally:
 report['wall_seconds']=time.monotonic()-started;save();print(json.dumps({k:v for k,v in report.items() if k!='failures'},indent=2),flush=True)
