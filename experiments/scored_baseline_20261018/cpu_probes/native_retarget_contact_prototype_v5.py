"""CPU feasibility experiment only; native contacts, bounded robot retarget changes."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np,mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/code/scripts/dex3_training')
from check_native_scene import compile_scene
from v2d_dex3_flashchord.kinematics import URDFKinematics
I=Path('/inputs');M=json.loads((I/'manifest.json').read_text());scene,reference,solver,input_sha,runtime=compile_scene(I);model=solver.mj_model
with np.load(I/'targets.npz',allow_pickle=False) as f:q=f['robot_qpos'].copy();frames=f['frame_indices']
position,quaternion=reference.object_body_pos_w(),reference.object_body_quat_w();names={i:model.body(i).name for i in range(1,model.nbody)}
def jid(n):
 a=[j for j in range(model.njnt) if model.joint(j).name==n or model.joint(j).name.endswith('_'+n) or model.joint(j).name.endswith('/'+n)];assert len(a)==1,(n,a);return a[0]
kin={s:URDFKinematics(I/f'robot/urdfs/g1/dex3_{s}.urdf') for s in ['left','right']};hand_ids={};finger_ids={};wrist_ids={}
for si,s in enumerate(['left','right']):
 fk=kin[s].poses(q[0,si,:7],dict(zip(M['joint_names'][s],q[0,si,7:],strict=True)))
 hand_ids[s]=[bid for bid,name in names.items() if any(name==n or name.endswith('_'+n) or name.endswith('/'+n) for n in fk)];assert len(hand_ids[s])>=8
 finger_ids[s]=[jid(n) for n in M['joint_names'][s]];wrist_ids[s]=[jid(f'{s}_wrist_{axis}') for axis in 'xyz']+[jid(f'{s}_wrist_rotvec')]
object_ids={o.name:[b.body_id+1 for b in o.bodies] for o in scene.objects};environment=set(sum(object_ids.values(),[]));environment.add(scene.support.body_id+1)
def apply_robot(state,side,target):
 for j,v in zip(wrist_ids[side][:3],target[:3],strict=True):state.qpos[int(model.jnt_qposadr[j])]=v
 adr=int(model.jnt_qposadr[wrist_ids[side][3]]);state.qpos[adr:adr+4]=target[[6,3,4,5]]
 for j,v in zip(finger_ids[side],target[7:],strict=True):state.qpos[int(model.jnt_qposadr[j])]=v
def reset(frame):
 state=mujoco.MjData(model)
 for obj in scene.objects:
  b,rid=obj.root.body_id+1,obj.root.reference_body_id;j=int(model.body_jntadr[b]);a=int(model.jnt_qposadr[j]);state.qpos[a:a+3]=position[frame,rid];state.qpos[a+3:a+7]=quaternion[frame,rid]
 for si,s in enumerate(['left','right']):apply_robot(state,s,q[frame,si])
 mujoco.mj_forward(model,state);return state
def contacts(state,side):
 out=[];parts=set(hand_ids[side])
 for c in state.contact[:state.ncon]:
  a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
  if (a in parts and b in environment) or (b in parts and a in environment):out.append(dict(bodies=[names[a],names[b]],signed_gap_m=float(c.dist)))
 return out
def verify_fk(state,side,target):
 expected=kin[side].poses(target[:7],dict(zip(M['joint_names'][side],target[7:],strict=True)));maximum=0
 for bid in hand_ids[side]:
  hits=[n for n in expected if names[bid]==n or names[bid].endswith('_'+n) or names[bid].endswith('/'+n)];assert len(hits)==1;T=expected[hits[0]];err=max(np.max(np.abs(state.xpos[bid]-T[:3,3])),np.max(np.abs(state.xmat[bid].reshape(3,3)-T[:3,:3])));assert err<2e-6;maximum=max(maximum,float(err))
 return maximum
def pad_gaps(state,side):
 out={}
 for obj,parts in object_ids.items():
  og=np.flatnonzero(np.isin(model.geom_bodyid,parts));digits={}
  for digit,last in [('thumb',2),('index',1),('middle',1)]:
   hs=[b for b in hand_ids[side] if names[b].endswith(f'{side}_hand_{digit}_{last}_link')];hg=np.flatnonzero(np.isin(model.geom_bodyid,hs));assert len(hg)
   digits[digit]=min(float(mujoco.mj_geomDistance(model,state,int(a),int(b),1.0,np.zeros(6))) for a in hg for b in og)
  out[obj]=digits
 return out
result=dict(schema='v2d.native-retarget-contact-feasibility-prototype.v5',input_manifest_sha256=input_sha,runtime_commit=runtime,physics_stepped=False,source_inputs_changed=False,robot_targets_changed_in_memory_only=True,physical_task_accepted=False,contact_force_or_grasp_assessed=False,rows=[],producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());start=time.monotonic()
candidate_q=q.copy();original_q=q.copy()
for frame,side in [(f,s) for f in range(len(frames)) for s in ['left','right']]:
 if time.monotonic()-start>240:raise TimeoutError('Full-trajectory CPU probe deadline')
 si=['left','right'].index(side);state=reset(frame);target=q[frame,si].copy();before=contacts(state,side);verify_fk(state,side,target)
 if min([c['signed_gap_m'] for c in before],default=0)>=-.0006:
  result['rows'].append(dict(frame=frame,hand=side,projection_needed=False,optimizer_success=True,minimum_signed_gap_before_m=min([c['signed_gap_m'] for c in before],default=0),minimum_signed_gap_after_m=min([c['signed_gap_m'] for c in before],default=0)));continue
 base_finger=target[7:];bounds=[(-3,3)]*3+[(-2.5,2.5)]*3+[(max(-2,float((model.jnt_range[j,0]-v)/.2)),min(2,float((model.jnt_range[j,1]-v)/.2))) for j,v in zip(finger_ids[side],base_finger,strict=True)]
 def pose(x):
  t=target.copy();t[:3]+=.01*x[:3];t[3:7]=(Rotation.from_rotvec(.1*x[3:6])*Rotation.from_quat(target[3:7])).as_quat();t[7:]+=.2*x[6:];return t
 def objective(x):
  t=pose(x);apply_robot(state,side,t);mujoco.mj_forward(model,state);c=contacts(state,side);penetration=np.minimum([v['signed_gap_m']+.0005 for v in c],0)/.005
  return float(.05*np.dot(x,x)+100*np.dot(penetration,penetration))
 # Iterative native contact projection using analytic point Jacobians.
 initial=objective(np.zeros(13));t0=time.monotonic();original_state=state.qpos.copy()
 dofs=[]
 for j in wrist_ids[side]+finger_ids[side]:
  count=3 if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_BALL else 1
  dofs.extend(range(int(model.jnt_dofadr[j]),int(model.jnt_dofadr[j])+count))
 assert len(dofs)==13
 scales=np.array([.005]*3+[.05]*3+[.07]*7);parts=set(hand_ids[side]);qp_trace=[];jac_error=0.0
 for iteration in range(100):
  mujoco.mj_forward(model,state);A=[];b=[]
  for c in state.contact[:state.ncon]:
   ba,bb=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
   if ba in parts and bb in environment:bid=ba;sign=-1
   elif bb in parts and ba in environment:bid=bb;sign=1
   else:continue
   if c.dist>.005:continue
   jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(model,state,jp,jr,c.pos,bid)
   A.append(sign*np.asarray(c.frame[:3])@jp[:,dofs]);b.append(.0005-float(c.dist))
   if iteration==0 and jac_error==0:
    # Verify the native hand-point Jacobian under native quaternion integration.
    local=state.xmat[bid].reshape(3,3).T@(c.pos-state.xpos[bid]);base_point=c.pos.copy();base_q=state.qpos.copy();direction=np.zeros(model.nv);direction[dofs]=np.linspace(-.2,.2,13);eps=1e-6
    mujoco.mj_integratePos(model,state.qpos,direction,eps);mujoco.mj_forward(model,state);actual=state.xpos[bid]+state.xmat[bid].reshape(3,3)@local;predicted=base_point+eps*(jp@direction);jac_error=float(np.max(np.abs(actual-predicted)));assert jac_error<1e-9,jac_error
    state.qpos[:]=base_q;mujoco.mj_forward(model,state)
  if not A:break
  A=np.asarray(A)*scales[None];b=np.asarray(b);drift=np.zeros(model.nv);mujoco.mj_differentiatePos(model,drift,1,original_state,state.qpos);d=drift[dofs]/np.array([.02]*3+[.2]*3+[.3]*7)
  prior_scale=scales/np.array([.02]*3+[.2]*3+[.3]*7)
  def quadratic(z):
   residual=np.minimum((A@z-b)/.005,0);prior=d+prior_scale*z;value=.1*np.dot(prior,prior)+100*np.dot(residual,residual);gradient=.2*prior*prior_scale+200*(A/.005).T@residual;return float(value),gradient
  z_bounds=[]
  for col,j in enumerate(wrist_ids[side][:3]):
   delta=state.qpos[int(model.jnt_qposadr[j])]-original_state[int(model.jnt_qposadr[j])]
   z_bounds.append((max(-1,float((-.04-delta)/scales[col])),min(1,float((.04-delta)/scales[col]))))
  z_bounds += [(-1,1)]*3
  for j,scale in zip(finger_ids[side],scales[6:],strict=True):
   val=state.qpos[int(model.jnt_qposadr[j])];z_bounds.append((max(-1,float((max(model.jnt_range[j,0],original_state[int(model.jnt_qposadr[j])]-.4)-val)/scale)),min(1,float((min(model.jnt_range[j,1],original_state[int(model.jnt_qposadr[j])]+.4)-val)/scale))))
  solution=minimize(quadratic,np.zeros(13),jac=True,method='L-BFGS-B',bounds=z_bounds,options=dict(maxiter=300,ftol=1e-12,gtol=1e-8))
  step=np.zeros(model.nv);step[dofs]=scales*solution.x;base=state.qpos.copy()
  def merit():
   depths=np.maximum([.0005-c['signed_gap_m'] for c in contacts(state,side)],0)
   return float(np.max(depths,initial=0)**2+.1*np.dot(depths,depths))
  before_merit=merit();best=base.copy();best_merit=before_merit;chosen_alpha=0
  for alpha in [1,.5,.25,.125,.0625,.03125]:
   state.qpos[:]=base;mujoco.mj_integratePos(model,state.qpos,step,alpha);mujoco.mj_forward(model,state)
   delta=np.zeros(model.nv);mujoco.mj_differentiatePos(model,delta,1,original_state,state.qpos)
   if np.linalg.norm(delta[dofs[:3]])>.05 or np.linalg.norm(delta[dofs[3:6]])>.5:continue
   value=merit()
   if value<best_merit-1e-15:best=state.qpos.copy();best_merit=value;chosen_alpha=alpha
  state.qpos[:]=best;mujoco.mj_forward(model,state)
  worst=min([c['signed_gap_m'] for c in contacts(state,side)],default=0);qp_trace.append(dict(iteration=iteration,minimum_native_gap_m=worst,qp_success=bool(solution.success),accepted_step_fraction=chosen_alpha))
  if chosen_alpha==0:break
  if worst>=-.0006:break
 proposed=target.copy()
 for col,j in enumerate(wrist_ids[side][:3]):proposed[col]=state.qpos[int(model.jnt_qposadr[j])]
 ad=int(model.jnt_qposadr[wrist_ids[side][3]]);quat=state.qpos[ad:ad+4];proposed[3:7]=quat[[1,2,3,0]]
 for col,j in enumerate(finger_ids[side]):proposed[7+col]=state.qpos[int(model.jnt_qposadr[j])]
 error=verify_fk(state,side,proposed);after=contacts(state,side)
 opt=type('Result',(),dict(success=min([c['signed_gap_m'] for c in after],default=0)>=-.0006,message='native iterative contact projection; contact proximity not optimized',nfev=len(qp_trace),fun=float(quadratic(np.zeros(13))[0])))()
 row=dict(frame=frame,hand=side,optimizer_success=bool(opt.success),message=str(opt.message),evaluations=int(opt.nfev),wall_seconds=time.monotonic()-t0,objective_before=initial,objective_after=float(opt.fun),minimum_signed_gap_before_m=min([v['signed_gap_m'] for v in before],default=0),minimum_signed_gap_after_m=min([v['signed_gap_m'] for v in after],default=0),maximum_native_fk_error=error,root_translation_change_m=float(np.linalg.norm(proposed[:3]-target[:3])),root_rotation_change_rad=float((Rotation.from_quat(target[3:7]).inv()*Rotation.from_quat(proposed[3:7])).magnitude()),maximum_finger_change_rad=float(np.max(np.abs(proposed[7:]-target[7:]))),original_robot_qpos=target.tolist(),proposed_robot_qpos=proposed.tolist(),native_contacts_before=before,native_contacts_after=after,pad_gaps_after_m=pad_gaps(state,side),native_jacobian_fd_max_error_m=jac_error,qp_trace=qp_trace,contact_proximity_not_optimized=True,physical_task_accepted=False)
 candidate_q[frame,si]=proposed;result['rows'].append(row)
 if side=='right' and frame%25==0:print(json.dumps(dict(frame=frame,elapsed_seconds=time.monotonic()-start)),flush=True)
assert hashlib.sha256((I/'manifest.json').read_bytes()).hexdigest()==input_sha
np.savez_compressed('/proof/native-projected-robot-commands-v5.npz',frame_indices=frames,robot_qpos_stage=candidate_q,original_robot_qpos_stage=original_q,source_manifest_sha256=np.array(input_sha))
result['projection_complete']=True;result['full_trajectory_samples']=len(result['rows']);result['failed_hand_slots']=sum(not r['optimizer_success'] for r in result['rows']);result['worst_remaining_signed_gap_m']=min(r['minimum_signed_gap_after_m'] for r in result['rows']);result['projected_command_sha256']=hashlib.sha256(Path('/proof/native-projected-robot-commands-v5.npz').read_bytes()).hexdigest();result['wall_seconds']=time.monotonic()-start
result['root_translation_change_max_m']=float(np.linalg.norm(candidate_q[:,:,:3]-original_q[:,:,:3],axis=-1).max())
result['root_rotation_change_max_rad']=float((Rotation.from_quat(original_q[:,:,3:7].reshape(-1,4)).inv()*Rotation.from_quat(candidate_q[:,:,3:7].reshape(-1,4))).magnitude().max())
result['maximum_finger_change_rad']=float(np.abs(candidate_q[:,:,7:]-original_q[:,:,7:]).max());result['maximum_adjacent_finger_speed_rad_s']=float(np.abs(np.diff(candidate_q[:,:,7:],axis=0)/.05).max())
Path('/proof/contact-feasibility-prototype-v5.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)

