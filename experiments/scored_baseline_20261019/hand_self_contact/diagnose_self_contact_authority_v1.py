"""NEW source-authority inventory and neutral/self-gap derivative controls."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.monotonic();O=S/'e037-self-contact-authority-controls-v1';plan_path=S/'compute/self-contact-authority-CPU-plan-v1.json';plan=json.loads(plan_path.read_text())
import mujoco,numpy as np
from audit_hand_reference_contacts import canonical_id
from v2d_dex3_flashchord.kinematics import URDFKinematics
P=S/'e037-motion-constrained-wrist-native-v1';f=P/'frozen';model_path=P/'native-model.mjb';assert sha(model_path)==plan['native_model_sha256'];model=mujoco.MjModel.from_binary_path(str(model_path));data=mujoco.MjData(model);jn=[model.joint(i).name for i in range(model.njnt)];bn=[model.body(i).name for i in range(model.nbody)];manifest=json.loads((f/'manifest.json').read_text());audit_path=S/'e037-cached-hand-self-contacts-v1/report.json';audit=json.loads(audit_path.read_text());assert audit['native_model_sha256']==sha(model_path)
with np.load(f/'targets.npz') as z:q=z['robot_qpos'].copy();ts=z['timestamps_s'].copy()
with np.load(f/'source-confidence.npz') as z:direct=z['hand_directly_observed'].copy();metric=z['hand_metric_valid'].copy();valid=z['hand_valid'].copy();source_ts=z['timestamps_s'].copy()
np.testing.assert_array_equal(ts,source_ts);assert (np.diff(ts)>0).all()
fingers={side:[canonical_id(jn,n) for n in manifest['joint_names'][side]] for side in ['left','right']};wrists={side:[canonical_id(jn,f'{side}_wrist_{a}') for a in ['x','y','z','rotvec']] for side in ['left','right']};kins={side:URDFKinematics(f/f'robot/urdfs/g1/dex3_{side}.urdf') for side in ['left','right']};rows=[]
def state(frame,replacement=None):
 data.qpos[:]=model.qpos0;data.qvel[:]=0
 for si,side in enumerate(['left','right']):
  target=q[frame,si].copy()
  if replacement and replacement[0]==side:target[7:]=replacement[1]
  w=wrists[side];data.qpos[model.jnt_qposadr[w[:3]]]=target[:3];a=int(model.jnt_qposadr[w[3]]);data.qpos[a:a+4]=target[[6,3,4,5]];data.qpos[model.jnt_qposadr[fingers[side]]]=target[7:]
 mujoco.mj_fwdPosition(model,data)
for si,side in enumerate(['left','right']):
 bad=np.array([any(c['hand']==side and c['signed_gap_m']<-.001 for c in row['contacts']) for row in audit['frames']]);kin=kins[side];names=manifest['joint_names'][side];limits=np.array([kin.limits[n] for n in names]);vel=np.array([kin.velocity_limits[n] for n in names]) if hasattr(kin,'velocity_limits') else None
 positions_ok=bool(((q[:,si,7:]>=limits[:,0]-1e-8)&(q[:,si,7:]<=limits[:,1]+1e-8)).all());rows.append(dict(hand=side,source_direct_frames=int(direct[:,si].sum()),source_metric_frames=int(metric[:,si].sum()),self_penetrating_frames=int(bad.sum()),self_penetrating_direct_frames=int((bad&direct[:,si]).sum()),self_penetrating_metric_frames=int((bad&metric[:,si]).sum()),self_penetrating_neither_direct_nor_metric=int((bad&~direct[:,si]&~metric[:,si]).sum()),actual_URDF_joint_names=names,actual_URDF_joint_position_limits_rad=limits.tolist(),original_joint_positions_within_URDF=positions_ok,original_joint_speed_max_abs_rad_s=np.max(abs(np.diff(q[:,si,7:],axis=0)/np.diff(ts)[:,None]),axis=0).tolist()))
controls=[];derivatives=[]
for si,side in enumerate(['left','right']):
 palm=canonical_id(bn,f'{side}_hand_palm_link');thumb=canonical_id(bn,f'{side}_hand_thumb_1_link');ga=int(np.flatnonzero(model.geom_bodyid==palm)[0]);gb=int(np.flatnonzero(model.geom_bodyid==thumb)[0])
 for label,values in [('reference50',q[50,si,7:]),('zero_neutral',np.zeros(7))]:
  state(50,(side,values));segment=np.zeros(6);gap=float(mujoco.mj_geomDistance(model,data,ga,gb,1.,segment));contacts=[dict(bodies=[bn[int(model.geom_bodyid[c.geom1])],bn[int(model.geom_bodyid[c.geom2])]],signed_gap_m=float(c.dist)) for c in data.contact[:data.ncon] if int(model.geom_bodyid[c.geom1]) in [palm,thumb] and int(model.geom_bodyid[c.geom2]) in [palm,thumb]];controls.append(dict(hand=side,configuration=label,declared_joint_values_rad=np.asarray(values).tolist(),palm_thumb1_signed_gap_m=gap,native_actual_pair_contacts=contacts))
 state(50);segment=np.zeros(6);gap=float(mujoco.mj_geomDistance(model,data,ga,gb,1.,segment));normal=(segment[3:]-segment[:3])/gap;JA=np.zeros((3,model.nv));JB=JA.copy();mujoco.mj_jac(model,data,JA,None,segment[:3],palm);mujoco.mj_jac(model,data,JB,None,segment[3:],thumb);analytic=normal@(JB-JA)
 for col in [4,5,6]:
  jid=fingers[side][col];dof=int(model.jnt_dofadr[jid]);values=[]
  for h in [1e-4,5e-5]:
   gaps=[]
   for sign in [-1,1]:x=q[50,si,7:].copy();x[col]+=sign*h;state(50,(side,x));gaps.append(float(mujoco.mj_geomDistance(model,data,ga,gb,1.,None)))
   values.append((gaps[1]-gaps[0])/(2*h))
  derivatives.append(dict(hand=side,joint=manifest['joint_names'][side][col],analytic_gap_derivative_m_per_rad=float(analytic[dof]),central_differences_m_per_rad=values,maximum_error_m_per_rad=max(abs(v-analytic[dof]) for v in values)))
report=dict(schema='v2d.self-contact-source-authority-and-neutral-controls.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),inputs_sha256={str(p):sha(p) for p in [audit_path,model_path,f/'manifest.json',f/'targets.npz',f/'source-confidence.npz']},authority_rows=rows,neutral_controls=controls,gap_derivative_controls=derivatives,source_human_observations_confidence_and_commands_not_modified=True,robot_finger_targets_are_derived_not_measured_joint_angles=True,source_nonregression_and_authority_required_before_retargeting=True,no_prior_audit_optimizer_or_source_fit_repeated=True,no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
