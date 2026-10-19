"""NEW one-knot thumb-only self-contact correction with unchanged source gates."""
from pathlib import Path
import datetime,hashlib,itertools,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.monotonic();O=S/'e037-source-thumb-self-contact-sample-v1';plan_path=S/'compute/source-thumb-self-contact-sample-CPU-plan-v1.json';plan=json.loads(plan_path.read_text())
import mujoco,numpy as np
from scipy.optimize import minimize
from audit_hand_reference_contacts import canonical_id
P=S/'e037-motion-constrained-wrist-native-v1';f=P/'frozen';mp=P/'native-model.mjb';assert sha(mp)==plan['native_model_sha256'];model=mujoco.MjModel.from_binary_path(str(mp));data=mujoco.MjData(model);bn=[model.body(i).name for i in range(model.nbody)];jn=[model.joint(i).name for i in range(model.njnt)];manifest=json.loads((f/'manifest.json').read_text());frame=plan['frame'];side=plan['hand'];si=0 if side=='left' else 1
with np.load(f/'targets.npz') as z:q=z['robot_qpos'].copy();objects=z['object_poses'].copy()
with np.load(f/'aligned-source-reference.npz') as z:aligned=z['robot_qpos'].copy();camera=z['camera_c2w'][frame].copy();human=z['hand_joints_world'][frame,si,[0,4,8,12]].copy()
with np.load(f/'source-confidence.npz') as z:direct=bool(z['hand_directly_observed'][frame,si]);metric=bool(z['hand_metric_valid'][frame,si]);valid=bool(z['hand_valid'][frame,si])
assert valid and direct and not metric,'This predeclared probe is observed but nonmetric, so it supplies image constraints without invented depth truth'
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz') as z:K=z['intrinsics'].copy()
stage=np.asarray(json.loads((f/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);A=np.einsum('ji,kj->ik',camera[:3,:3],stage,optimize=False);offset=np.einsum('ij,j->i',camera[:3,:3].T,camera[:3,3],optimize=False);observed=np.einsum('ni,ij->nj',human-camera[:3,3],camera[:3,:3],optimize=False)
def pixels(C):return np.c_[K[0,0]*C[:,0]/C[:,2]+K[0,2],K[1,1]*C[:,1]/C[:,2]+K[1,2]]
observed_uv=pixels(observed);base=model.qpos0.copy();bindings={}
for sidx,s in enumerate(['left','right']):
 w=[canonical_id(jn,f'{s}_wrist_{a}') for a in ['x','y','z','rotvec']];ids=[canonical_id(jn,n) for n in manifest['joint_names'][s]];base[model.jnt_qposadr[w[:3]]]=q[frame,sidx,:3];a=int(model.jnt_qposadr[w[3]]);base[a:a+4]=q[frame,sidx,[6,3,4,5]];base[model.jnt_qposadr[ids]]=q[frame,sidx,7:];bindings[s]=ids
for oi,obj in enumerate(manifest['objects']):
 bid=canonical_id(bn,obj['name']+'_object');a=int(model.jnt_qposadr[model.body_jntadr[bid]]);v=objects[frame,oi];base[a:a+3]=v[:3];base[a+3:a+7]=v[[6,3,4,5]]
ids=bindings[side];thumb=np.array([i for i,n in enumerate(manifest['joint_names'][side]) if '_thumb_' in n]);tj=np.array(ids)[thumb];dofs=model.jnt_dofadr[tj];tipids=[canonical_id(bn,n) for n in [f'{side}_hand_palm_link',f'{side}_thumb_tip',f'{side}_index_tip',f'{side}_middle_tip']];hand_geoms=[g for g,b in enumerate(model.geom_bodyid) if bn[b].startswith('dex3_'+side+'_')];pairs=list(itertools.combinations(hand_geoms,2));pair_index={tuple(sorted(p)):i for i,p in enumerate(pairs)};qbase=q[frame,si,7+thumb].copy();origin=aligned[frame,si,7+thumb];limits=model.jnt_range[tj];lower=np.maximum(limits[:,0],origin-.25)-qbase;upper=np.minimum(limits[:,1],origin+.25)-qbase;assert ((lower<=0)&(upper>=0)).all();cache={}
def evaluate(x):
 if time.monotonic()-start>220:raise TimeoutError('Thumb sample guard deadline')
 if 'x' in cache and np.array_equal(x,cache['x']):return cache['v']
 data.qpos[:]=base;data.qpos[model.jnt_qposadr[tj]]=qbase+x;mujoco.mj_fwdPosition(model,data);C=np.einsum('ni,ji->nj',data.xpos[tipids],A,optimize=False)-offset;assert (C[:,2]>.01).all();uv=pixels(C);errors=np.linalg.norm(uv-observed_uv,axis=1);gap=np.zeros(len(pairs));jac=np.zeros((len(pairs),3));all_self=[]
 for c in data.contact[:data.ncon]:
  ga,gb=int(c.geom1),int(c.geom2)
  if ga not in hand_geoms or gb not in hand_geoms:continue
  all_self.append(dict(native_bodies=[bn[int(model.geom_bodyid[ga])],bn[int(model.geom_bodyid[gb])]],signed_gap_m=float(c.dist)))
  k=pair_index[tuple(sorted([ga,gb]))]
  if c.dist>=gap[k]:continue
  gap[k]=float(c.dist);normal=np.asarray(c.frame[:3]);pa=np.asarray(c.pos)-.5*c.dist*normal;pb=np.asarray(c.pos)+.5*c.dist*normal;JA=np.zeros((3,model.nv));JB=JA.copy();mujoco.mj_jac(model,data,JA,None,pa,int(model.geom_bodyid[ga]));mujoco.mj_jac(model,data,JB,None,pb,int(model.geom_bodyid[gb]));jac[k]=np.einsum('i,ij->j',normal,JB[:,dofs]-JA[:,dofs],optimize=False)
 JP=np.zeros((3,model.nv));mujoco.mj_jac(model,data,JP,None,data.xpos[tipids[1]],tipids[1]);jp=np.einsum('ij,jk->ik',A,JP[:,dofs],optimize=False);c=C[1];pixelJ=np.array([[K[0,0]/c[2],0,-K[0,0]*c[0]/c[2]**2],[0,K[1,1]/c[2],-K[1,1]*c[1]/c[2]**2]]);du=np.einsum('ij,jk->ik',pixelJ,jp,optimize=False);errJ=np.einsum('i,ij->j',uv[1]-observed_uv[1],du,optimize=False)/max(errors[1],1e-12);v=dict(gaps=gap,gap_jac=jac,errors=errors,error_jac=errJ,self_contacts=all_self);cache.update(x=x.copy(),v=v);return v
zero=np.zeros(3);before=evaluate(zero);cap=before['errors'][1]-min(.001,.01*before['errors'][1]);normalization=max(before['errors'][1],.05)
def constraints(x):
 v=evaluate(x);return np.r_[(v['gaps']+.0005)/.01,(cap-v['errors'][1])/normalization]
def jac(x):
 v=evaluate(x);return np.r_[v['gap_jac']/.01,(-v['error_jac']/normalization)[None]]
solution=minimize(lambda x:float(np.dot(x,x)),zero,jac=lambda x:2*x,method='SLSQP',bounds=list(zip(lower,upper,strict=True)),constraints=[dict(type='ineq',fun=constraints,jac=jac)],options=dict(maxiter=100,ftol=1e-12));after=evaluate(solution.x);returned=q[frame,si].copy();returned[7+thumb]+=solution.x;bounds=bool((solution.x>=lower-1e-8).all() and (solution.x<=upper+1e-8).all());source=bool((after['errors']<=before['errors']+1e-6).all());maxpen=max(0.,float(-after['gaps'].min()));passed=bool(solution.success and bounds and source and maxpen<=.001 and constraints(solution.x).min()>=-1e-7)
np.savez_compressed(O/'single-knot-thumb-command.npz',frame=frame,side=si,original_command=q[frame,si],returned_command=returned)
report=dict(schema='v2d.source-constrained-thumb-self-contact-sample.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),inputs_sha256={str(p):sha(p) for p in [mp,f/'manifest.json',f/'targets.npz',f/'aligned-source-reference.npz',f/'source-confidence.npz']},frame=frame,hand=side,source_directly_observed=direct,source_metric_valid=metric,source_human_landmarks_flags_and_roots_not_modified=True,all_other_fingers_and_other_hand_unchanged=True,source_constraints='all four original aligned human/image anchor errors nonregressing; optimize thumb only, no inferred depth truth',source_errors_before_after_px=[before['errors'].tolist(),after['errors'].tolist()],solver_success=bool(solution.success),solver_message=str(solution.message),solver_iterations=int(solution.nit),returned_thumb_delta_rad=solution.x.tolist(),old_aligned_reference_component_cap_rad=.25,returned_in_original_component_and_URDF_bounds=bounds,before_self_contacts=before['self_contacts'],after_self_contacts=after['self_contacts'],maximum_returned_self_penetration_m=maxpen,single_knot_source_and_self_prerequisites_passed=passed,optimizer_return_only_no_best_intermediate=True,full_trajectory_motion_swept_hand_object_motor_grasp_not_qualified=True,no_native_compile_or_physics_steps=True,no_old_optimization_or_fit_repeated=True,new_GPU_jobs=0,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
