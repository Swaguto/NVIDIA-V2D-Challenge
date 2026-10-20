"""Independent SAVED-return audit only; no method/solver/source/NN/raster replay."""
from pathlib import Path
import datetime,json,hashlib,os,math
import numpy as np
from scipy import sparse
S=Path(__file__).resolve().parents[1];C=S/'compute';I=S/'e037-full-pose-evaluation-v2';O=S/'e037-full-pose-return-audit-v1';F=S/'e037-object-full-pose-factors-v1';P=S/'e037-full-pose-preflight-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 with (O/n).open('w')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def arrays(n,**d):
 with (O/n).open('wb')as f:np.savez_compressed(f,**d);f.flush();os.fsync(f.fileno())
def exp_independent(a):
 t=math.sqrt(math.fsum(float(x)*float(x)for x in a));x,y,z=map(float,a);H=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
 if t==0:return np.eye(3)
 return np.eye(3)+(math.sin(t)/t)*H+((1-math.cos(t))/(t*t))*(H@H)
r=json.loads((I/'report.json').read_text());save('NEW-return-audit-BEFORE.json',dict(report_sha256=sha(I/'report.json'),no_solver_method_source_NN_raster_FK_native_replay=True))
with np.load(I/'ONE-actual-raw-x-z-s-RETURN.npz')as z:x=z['x'];rawz=z['z'];raws=z['s']
solver=json.loads((I/'ONE-actual-solver-RETURN.json').read_text());journal=[json.loads(s)for s in (I/'normalized-BEFORE-AFTER-journal.jsonl').read_text().splitlines()];callbacks=[z for z in journal if z['kind']=='AFTER_NEW_actual_info_callback_RETURN'];assert [v['callback']for v in callbacks]==list(range(r['callbacks']));assert [v['actual_RETURN']for v in callbacks]==solver['callbacks']
if r['actual_nonzero_trials']==0:
 save('report.json',dict(schema='v2d.NEW-full-pose-saved-return-audit.v1',NEW_raw_solver_return=True,callbacks=len(callbacks),no_trial_or_candidate=True,no_solver_or_old_audit_replay=True));raise SystemExit(0)
with np.load(r['sealed_proposal']['path'])as z:params=z['parameters'];sealed_raw=z['raw_x'];indices=z['frame_indices'];times=z['timestamps_s']
assert np.array_equal(x,sealed_raw);cfg=r['configuration'];radius=cfg['rho_upper_m'];actual=x[:906].reshape(151,6).copy();actual[:,:3]*=radius;assert np.array_equal(params,actual);pid=r['proposal_id'];assert pid=='ONE-raw-full-pose-'+hashlib.sha256(params.tobytes()).hexdigest()[:16]
with np.load(F/'NEW-source-rotation-factors-RETURN.npz')as z:base=z['original_world_from_object'];cam0=z['original_camera_from_object'];rcw=z['camera_from_world_translation_linear'];V=z['visual_vertices']
with np.load(r['pose_RETURN']['path'])as z:world=z['world_from_object'];assert np.array_equal(z['parameters'],params)
Rerr=[];terr=[]
for f in range(151):
 Rerr.append(world[f,:3,:3]-base[f,:3,:3]@exp_independent(params[f,3:]));terr.append(world[f,:3,3]-(base[f,:3,3]+params[f,:3]))
re=float(np.max(abs(np.array(Rerr))));te=float(np.max(abs(np.array(terr))));arrays('NEW-raw-proposal-composition-residuals-RETURN.npz',rotation=np.array(Rerr),translation=np.array(terr));save('NEW-raw-proposal-identity-composition-RETURN.json',dict(all151_clock_identity=True,raw_x_deconditioning_exact=True,sealed_proposal_id=pid,rotation_error=re,translation_error=te,diagnostic_rotation_bound=2e-14,original_gates_NOT_evaluated_by_this_identity=True));assert re<2e-14 and te==0
# Bind source decisions using actual returned masks/distances only.
stats=json.loads((S/'e037-training-source-constraints-v1/all151-NEW-training-source-statistics.json').read_text());source=json.loads(Path(r['training_source_RETURN']['path']).read_text());out=[];fail=[];counts=dict(IoU=0,median=0,quality=0,unknown=0);camera_errors=[];query_count=0
for f,ret in enumerate(source):
 assert ret['frame']==int(indices[f])and ret['proposal_id']==pid
 zero=bool(np.array_equal(params[f],np.zeros(6)))
 if zero:assert ret['status']=='CACHED_F_BY_HASH';continue
 query_count+=1;assert ret['status']=='NEW_actual_full_pose_TRAINING_ONLY'
 part=Path(stats[f]['partition']['path']);assert sha(part)==stats[f]['partition']['sha256']
 with np.load(part)as z:M=z['training_mask'];D=z['domain']
 with np.load(I/f'{pid}-f{f:03d}-image-RETURN.npz')as z:mask=z['predicted_training_mask']
 with np.load(I/f'{pid}-f{f:03d}-median-RETURN.npz')as z:first=z['nearest_vertex_distances'];central=z['central_point_indices'];central_dist=z['central_distances'];local=z['point_local'];nearest=z['nearest_vertex_indices']
 with np.load(I/f'{pid}-f{f:03d}-NEW-camera-pose-RETURN.npz')as z:T=z['camera_from_object'];camera=z['camera_vertices']
 expectedR=cam0[f,:3,:3]@exp_independent(params[f,3:]);expectedt=cam0[f,:3,3]+rcw[f]@params[f,:3];camera_errors.append(max(float(np.max(abs(T[:3,:3]-expectedR))),float(np.max(abs(T[:3,3]-expectedt)))))
 # actual camera positions are saved, not recomputed/rerendered; finiteness and
 # positive depth bind the returned projection scope, never source accuracy.
 assert np.isfinite(camera).all()and np.all(camera[:,2]>.01)and np.isfinite(first).all()
 Icount=int((mask&M&D).sum());Ucount=int(((mask|M)&D).sum());ordered=np.sort(first);n=len(first);median=float((ordered[(n-1)//2]+ordered[n//2])/2);q=float((Icount/max(1,Ucount))*np.exp(-median/.02));det=Icount*stats[f]['training_union']-stats[f]['training_intersection']*Ucount;dm=median-stats[f]['training_mesh_vertex_median_m'];dq=q-stats[f]['training_quality'];passed=bool(det>=0 and dm<=0 and dq>=0)
 assert Icount==ret['training_intersection']and Ucount==ret['training_union']and median==ret['training_median_m']and q==ret['training_quality']and passed==ret['passes_TRAINING_construction']
 if not passed:fail.append(f)
 counts['IoU']+=int(det<0);counts['median']+=int(dm>0);counts['quality']+=int(dq<0);out.append(dict(frame=f,intersection=Icount,union=Ucount,rounded_median_m=median,quality=q,IoU_determinant=det,passes=passed,actual_NN_return_entries=len(first)))
 # Every frame's audit RETURN is durable before subsequent frame checks.
 save(f'NEW-frame-{f:03d}-saved-source-decision-RETURN.json',out[-1])
assert fail==r['training_failed_frames'];ce=max(camera_errors,default=0);save('NEW-all151-source-return-binding-RETURN.json',dict(actual_nonzero_frame_queries=query_count,failed_frames=fail,failure_counts=counts,maximum_camera_pose_map_error=ce,source_point_distance_entries=sum(x['actual_NN_return_entries']for x in out),no_NN_search_or_fullpixel_gate_replayed=True));assert ce<2e-14
# Motion decisions from SAVED vectors only; no world-to-angular function replay.
motion=json.loads((I/'ONE-actual-original-six-motion-RETURN.json').read_text())
with np.load(I/'ONE-actual-original-motion-arrays-RETURN.npz')as z:ma={k:z[k]for k in z.files}
values=[float(np.sqrt(np.sum(ma[k]*ma[k],axis=1)).max())for k in ['linear_velocity','angular_velocity','linear_acceleration','angular_acceleration']]+[float(np.quantile(np.sqrt(np.sum(ma[k]*ma[k],axis=1)),.95))for k in ['linear_velocity','angular_velocity']];passes=[bool(v<=c+1e-9)for v,c in zip(values,motion['caps'])];save('NEW-six-SAVED-motion-decision-RETURN.json',dict(values=values,returned_values=motion['values'],passes=passes,original_comparison=1e-9));assert passes==motion['passes']==r['all6motion_passes']and np.max(abs(np.array(values)-motion['values']))<1e-14
# Actual returned full face-union maxima; do not recompute geometry/normals.
with np.load(r['geometry_RETURN']['path'])as z:gaps=z['gaps'];axis=z['all_axis_gaps'];ptr=z['axis_ptr'];branches=z['branches'];objposes=z['NEW_object_body_poses'];actual_params=z['parameters']
assert np.array_equal(actual_params,params);maxima=np.maximum.reduceat(axis,ptr[:-1]);assert np.array_equal(maxima,gaps);phase=float(np.maximum(0,-.001-maxima).max(initial=0));assert phase==r['raw_trial_proxy_phase_m'];save('NEW-allpair-SAVED-union-maxima-RETURN.json',dict(pair_maxima=len(gaps),actual_axis_values=len(axis),all_arrays_finite=bool(np.isfinite(axis).all()),proxy_phase_m=phase,NOT_native_penetration_or_force=True,current_J_not_absolute_theta_NOT_reused=True))
A=sparse.load_npz(P/'NEW-canonical-A-RETURN.npz')
with np.load(P/'NEW-canonical-vectors-RETURN.npz')as z:b=z['b']
res=A@x+raws-b;arrays('NEW-raw-linear-slack-residual-RETURN.npz',residual=res);slack=float(np.max(abs(res)));bounds=bool(np.all(np.linalg.norm(params[:,:3],axis=1)<=cfg['translation_norm_halfwidth_m'])and np.all(np.linalg.norm(params[:,3:],axis=1)<=cfg['rotation_norm_halfwidth_rad']));assert bounds==r['construction_norm_bounds_pass'];accepted=bool(not fail and all(passes)and bounds and phase<r['initial_proxy_phase_m']);assert accepted==r['accepted']
warning_lines=[s for s in (I/'worker.log').read_text().splitlines()if 'RuntimeWarning:'in s]
save('report.json',dict(schema='v2d.NEW-full-pose-saved-return-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_raw_solver_returns=1,actual_callbacks=len(callbacks),NEW_sealed_raw_pose_return=1,NEW_actual_training_decisions=151,actual_nonzero_source_queries=query_count,actual_source_point_distance_entries=sum(x['actual_NN_return_entries']for x in out),training_failed_frames=len(fail),training_failure_counts=counts,all6_original_motion_pass=all(passes),NEW_actual_axis_values=len(axis),NEW_allpair_maxima=len(gaps),raw_linear_slack_residual_DIAGNOSTIC=slack,actual_evaluation_RuntimeWarnings_retained=warning_lines,warning_cause_UNPROVEN=True,accepted=False if not accepted else True,no_method_solver_source_NN_raster_FK_native_physics_replay=True,all_old_failed_returns_and_gates_UNCHANGED=True,no_reference_task_policy_AUC_qualification=True));print(json.dumps(dict(success=True,callbacks=len(callbacks),training_failures=len(fail),failure_counts=counts,axis_values=len(axis),raw_slack_residual=slack)))
