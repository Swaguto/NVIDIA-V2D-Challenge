"""NEW coupled observed-source orientation/finger trajectory under ORIGINAL limits."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-source-root-finger-trajectory-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sys.path[:0]=[str(S/'compute'),str(R/'scripts/track3')]
import source_orientation_context_v1 as c
np=c.np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize,LinearConstraint
import scipy.sparse as sp
from temporal_pose_path import motion_summary
original=c.q.copy();N=len(original);free=np.arange(104,118);nf=len(free);nv=nf*10;B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037';bm=json.loads((B/'manifest.json').read_text());source_hashes={r['path']:r['sha256'] for r in bm['sources']};authority_path=S/'e037-wrist-source-authority-v1/report.json';authority=json.loads(authority_path.read_text());assert authority['source_exporter_and_retarget_code_match_frozen_manifests'];caps=authority['original_root_motion_caps'];oldR=Rotation.from_quat(original[:,1,3:7]).as_matrix();alignedR=Rotation.from_quat(c.aligned[:,1,3:7]).as_matrix();initial=np.empty((nf,10));lower=np.empty_like(initial);upper=np.empty_like(initial);sourceR=[];before=[];metadata=[]
for i,frame in enumerate(free):
 p=B/f'source/hands/right/{frame:06d}.json';assert sha(p)==source_hashes[str(p.relative_to(B))];record=json.loads(p.read_text());assert record['directly_observed'] and record['visual_valid'];assert c.flags['hand_valid'][frame,1] and c.flags['hand_directly_observed'][frame,1];sourceR.append(c.stage@c.cameras[frame,:3,:3]@Rotation.from_rotvec(record['mano']['global_orient']).as_matrix()@np.diag([-1.,-1.,1.]));initial[i]=np.r_[Rotation.from_matrix(oldR[frame]@alignedR[frame].T).as_rotvec(),original[frame,1,7:]];lo,hi=c.bounds(frame);lower[i]=np.r_[[-.08]*3,lo];upper[i]=np.r_[[.08]*3,hi];before.append(c.evaluate(frame,original[frame,1,7:]));metadata.append(dict(frame=int(frame),record_sha256=sha(p),source_direct=True,source_metric_frame_flag=bool(c.flags['hand_metric_valid'][frame,1]),orientation_model_derived_not_measured=True))
sourceR=np.stack(sourceR);initial=initial.ravel();lower=lower.ravel();upper=upper.ravel();assert (initial>=lower-1e-9).all() and (initial<=upper+1e-9).all();scales=np.tile(np.r_[[.08]*3,[.25]*7],nf);image_caps=np.array([b['errors'][1:] for b in before]);image_margin=np.minimum(1e-4,image_caps*.001);image_caps-=image_margin;image_norm=np.maximum(image_caps,1.);ori_before=np.array([Rotation.from_matrix(sourceR[i]@oldR[f].T).magnitude() for i,f in enumerate(free)]);ori_caps=ori_before-np.minimum(1e-6,ori_before*.001);ori_norm=np.maximum(ori_before,.01);cache={};frame_cache=[{} for _ in free];evaluations=0

def single(i,x):
 global evaluations
 if time.monotonic()-start>210:raise TimeoutError('Observed root/finger trajectory wall cap')
 frame=int(free[i]);entry=frame_cache[i]
 if 'x' in entry and np.array_equal(x,entry['x']):return entry['value']
 if not ((x>=lower.reshape(nf,10)[i]-1e-9).all() and (x<=upper.reshape(nf,10)[i]+1e-9).all()):raise ValueError('Out-of-original-bound evaluation')
 RR=Rotation.from_rotvec(x[:3]).as_matrix()@alignedR[frame];c.q[frame,1,3:7]=Rotation.from_matrix(RR).as_quat();v=c.evaluate(frame,x[3:]);ori=float(Rotation.from_matrix(sourceR[i]@RR.T).magnitude());cons=np.r_[(v['gaps']+.0005)/.01,(image_caps[i]-v['errors'][1:])/image_norm[i],(ori_caps[i]-ori)/ori_norm[i]];v.update(orientation_error_rad=ori,constraints=cons,rotation=RR);entry.update(x=x.copy(),value=v);evaluations+=1;return v

def nonlinear(x):
 X=x.reshape(nf,10);vals=[single(i,a) for i,a in enumerate(X)];out=np.concatenate([v['constraints'] for v in vals]);cache.update(x=x.copy(),vals=vals);return out

def nonlinear_jac(x):
 X=x.reshape(nf,10);size=len(c.pairs)+4;J=np.zeros((nf*size,nv));L=lower.reshape(nf,10);U=upper.reshape(nf,10)
 for i,a in enumerate(X):
  v=single(i,a);block=np.zeros((size,10));block[:len(c.pairs),3:]=v['gap_jac']/.01;block[len(c.pairs):len(c.pairs)+3,3:]=-v['error_jac'][1:]/image_norm[i,:,None]
  for j in range(3):
   h=2e-6;left=max(L[i,j],a[j]-h);right=min(U[i,j],a[j]+h);xl=a.copy();xr=a.copy();xl[j]=left;xr[j]=right;vl=single(i,xl)['constraints'];vr=single(i,xr)['constraints'];block[:,j]=(vr-vl)/(right-left)
  J[i*size:(i+1)*size,i*10:(i+1)*10]=block
 return J
# Exact original-reference joint bounds/added motion and actual hardware velocity.
E=np.zeros((N*7,nv))
for i,f in enumerate(free):E[f*7:(f+1)*7,i*10+3:i*10+10]=np.eye(7)
E=sp.csc_matrix(E);D1=sp.kron(sp.diags([-np.ones(N-1),np.ones(N-1)],[0,1],shape=(N-1,N)),sp.eye(7),format='csc');D2=sp.kron(sp.diags([np.ones(N-2),-2*np.ones(N-2),np.ones(N-2)],[0,1,2],shape=(N-2,N)),sp.eye(7),format='csc');Dspeed=sp.diags(np.repeat(1/np.diff(c.ts),7))@D1;fixed=original[:,1,7:].ravel()-E@initial;delta0=fixed-c.aligned[:,1,7:].ravel();speedcaps=np.tile([12,12,12,12,6.857,12,12],N-1);M=sp.vstack([D1@E,D2@E,Dspeed@E],format='csc');low=np.r_[-.05-D1@delta0,-.05-D2@delta0,-speedcaps-Dspeed@fixed];high=np.r_[.05-D1@delta0,.05-D2@delta0,speedcaps-Dspeed@fixed];changing=np.asarray(M.getnnz(axis=1)>0).ravel();M=M[changing].toarray();low=low[changing];high=high[changing];linear=LinearConstraint(M,low,high)
# Root translation is unchanged. Assess the ORIGINAL angular max-speed/acceleration
# bounds on modified edges plus the original global p95, without tightening/relabeling.
Droot=np.diff(np.eye(N),axis=0);Aroot=np.diff(Droot,axis=0);speed_edges=np.flatnonzero(np.any(Droot[:,free]!=0,axis=1));acc_edges=np.flatnonzero(np.any(Aroot[:,free]!=0,axis=1));dt=np.diff(c.ts);mid=(dt[:-1]+dt[1:])/2

def angular(x):
 X=x.reshape(nf,10);rot=oldR.copy();rot[free]=Rotation.from_rotvec(X[:,:3]).as_matrix()@alignedR[free];Q=Rotation.from_matrix(rot);omega=(Q[1:]*Q[:-1].inv()).as_rotvec()/dt[:,None];alpha=np.diff(omega,axis=0)/mid[:,None];speed=np.linalg.norm(omega,axis=1);acc=np.linalg.norm(alpha,axis=1)
 return np.r_[(caps['maximum_angular_speed_rad_s']-speed[speed_edges])/caps['maximum_angular_speed_rad_s'],(caps['maximum_angular_acceleration_rad_s2']-acc[acc_edges])/caps['maximum_angular_acceleration_rad_s2'],(caps['p95_angular_speed_rad_s']-np.quantile(speed,.95))/caps['p95_angular_speed_rad_s']]

def angular_jac(x):
 base=angular(x);J=np.zeros((len(base),nv))
 for i in range(nf):
  for j in range(3):
   k=i*10+j;h=2e-6;left=max(lower[k],x[k]-h);right=min(upper[k],x[k]+h);xl=x.copy();xr=x.copy();xl[k]=left;xr[k]=right;J[:,k]=(angular(xr)-angular(xl))/(right-left)
 return J

# Squared change with a fixed small adjacent-correction penalty, no reward/gain sweep.
G=(D1@E).toarray();g0=D1@delta0;weight=.01/.05**2

def objective(x):return float(np.sum(((x-initial)/scales)**2)+weight*np.sum((G@x+g0)**2))
def objective_jac(x):return 2*(x-initial)/scales**2+2*weight*G.T@(G@x+g0)
history=[]
def callback(x):
 vals=nonlinear(x);history.append(dict(iteration=len(history)+1,minimum_source_self_constraint=float(vals.min()),minimum_angular_constraint=float(angular(x).min()),objective=objective(x)))
 if len(history)%10==0:print(json.dumps(history[-1]),flush=True)
solution=minimize(objective,initial,jac=objective_jac,method='SLSQP',bounds=list(zip(lower,upper,strict=True)),constraints=[linear,dict(type='ineq',fun=nonlinear,jac=nonlinear_jac),dict(type='ineq',fun=angular,jac=angular_jac)],callback=callback,options=dict(maxiter=120,ftol=1e-9,disp=False));X=solution.x.reshape(nf,10);commands=original.copy();rows=[]
for i,f in enumerate(free):
 v=single(i,X[i]);commands[f,1,3:7]=Rotation.from_matrix(v['rotation']).as_quat();commands[f,1,7:]=X[i,3:];rows.append(dict(frame=int(f),source_image_before_after_px=[before[i]['errors'].tolist(),v['errors'].tolist()],source_orientation_before_after_deg=[float(np.degrees(ori_before[i])),float(np.degrees(v['orientation_error_rad']))],maximum_self_penetration_m=max(0.,float(-v['gaps'].min())),source_image_nonregression=bool((v['errors']<=before[i]['errors']+1e-6).all()),source_model_orientation_nonregression=bool(v['orientation_error_rad']<=ori_before[i]+1e-9),root_original_residual_rad=X[i,:3].tolist()))
def transforms(a):
 out=np.tile(np.eye(4),(len(a),1,1));out[:,:3,:3]=Rotation.from_quat(a[:,3:7]).as_matrix();out[:,:3,3]=a[:,:3];return out
motion=motion_summary(transforms(commands[:,1]),c.ts);root_motion=all(motion[k]<=caps[k]+1e-7 for k in caps);linear_value=M@solution.x;finger_motion=bool((linear_value>=low-1e-7).all() and (linear_value<=high+1e-7).all());bound=bool((solution.x>=lower-1e-7).all() and (solution.x<=upper+1e-7).all());source_ok=all(r['source_image_nonregression'] and r['source_model_orientation_nonregression'] for r in rows);self_ok=all(r['maximum_self_penetration_m']<=.001 for r in rows);passed=bool(solution.success and root_motion and finger_motion and bound and source_ok and self_ok and nonlinear(solution.x).min()>=-1e-7 and angular(solution.x).min()>=-1e-7)
np.testing.assert_array_equal(commands[:,0],original[:,0]);np.testing.assert_array_equal(commands[:,:,:3],original[:,:,:3]);fixedmask=np.ones(N,bool);fixedmask[free]=False;np.testing.assert_array_equal(commands[fixedmask,1],original[fixedmask,1]);np.savez_compressed(O/'reference-commands.npz',robot_qpos=commands,original_robot_qpos=original,timestamps_s=c.ts,frame_indices=np.arange(N));c.q=original
report=dict(schema='v2d.source-root-finger-trajectory.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),context_sha256=sha(S/'compute/source_orientation_context_v1.py'),input_manifest_sha256=sha(c.F/'manifest.json'),authority_report_sha256=sha(authority_path),source_neighborhood_authority_records=metadata,free_frames_half_open=[104,118],left_early_and_qualified_late_components_preserved=True,all_root_translations_and_human_source_data_flags_objects_model_unchanged=True,solver_success=bool(solution.success),solver_status=int(solution.status),solver_message=str(solution.message),solver_iterations=int(solution.nit),optimizer_return_only_not_best_intermediate=True,history=history,rows=rows,original_root_motion_caps=caps,returned_root_motion=motion,root_original_motion_passed=root_motion,finger_original_motion_passed=finger_motion,original_root_component_and_URDF_finger_bounds=bound,source_image_and_model_orientation_nonregression=source_ok,all_modified_self_knots_passed=self_ok,returned_minimum_source_self_constraint=float(nonlinear(solution.x).min()),returned_minimum_angular_constraint=float(angular(solution.x).min()),trajectory_source_self_motion_prerequisites_passed=passed,commands_sha256=sha(O/'reference-commands.npz'),frame_evaluations=evaluations,original_8mm_08rad_and_25rad_05rad_bounds_not_enlarged=True,fresh_independent_whole_clock_native_source_motion_verification_required=True,no_new_source_inference_or_old_source_pose_family_repeated=True,no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_physical_task_qualified=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['solver_success','solver_message','solver_iterations','root_original_motion_passed','finger_original_motion_passed','original_root_component_and_URDF_finger_bounds','source_image_and_model_orientation_nonregression','all_modified_self_knots_passed','trajectory_source_self_motion_prerequisites_passed','frame_evaluations','wall_seconds']},indent=2))
