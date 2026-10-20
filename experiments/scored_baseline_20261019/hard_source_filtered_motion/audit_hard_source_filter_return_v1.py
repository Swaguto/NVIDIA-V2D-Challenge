"""Independent saved-return audit of the SOLE NEW hard-source-filter episode.

Never invokes a source metric/raster/KD/FK/native operator, model builder or
solver. Actual returned masks/distances/steps are evidence, not source truth.
"""
from pathlib import Path
import datetime,hashlib,json,math,os
import numpy as np
from scipy import sparse

S=Path(__file__).resolve().parents[1];O=S/'e037-hard-source-filter-return-audit-v1';E=S/'e037-hard-source-filter-evaluation-v1';B=S/'e037-hard-source-filter-preflight-v1';D=S/'e037-object-dynamic-runtime-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
 if isinstance(v,np.ndarray):return plain(v.tolist())
 if isinstance(v,np.generic):return plain(v.item())
 if isinstance(v,dict):return {str(k):plain(x)for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [plain(x)for x in v]
 return v
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def arrays(n,**v):
 p=O/n;np.savez_compressed(p,**v);p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):log.write(json.dumps(plain(v),allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
 scope=dict(NEW_saved_QP_trial_geometry_source_motion_decisions_ONLY=True,
  no_function_source_NN_raster_FK_native_method_model_solver_replay=True,
  numerical_closure_guards_NOT_candidate_tolerances=True,
  raw_results_rejected_unless_original_gates_independently_qualify=True)
 save('ONE-saved-return-audit-scope-BEFORE.json',scope);emit(dict(kind='BEFORE_NEW_saved_return_audit',scope=scope))
 report=read(E/'report.json');decl=read(E/'ONE-episode-declaration-BEFORE.json');cfg=decl['config']
 provenance=read(E/'NEW-input-provenance-BY-HASH.json')
 for p,h in provenance.items():assert sha(p)==h
 stats=read(S/'e037-training-source-constraints-v1/all151-NEW-training-source-statistics.json')
 caps=read(D/'runtime-config-and-original-authority.json')['original_object_caps']
 with np.load(D/'dynamic-axis-group-bindings.npz')as z:ptr=z['ptr'];pair_frames=z['pair_frames']
 axes=np.load(D/'ALL-retained-axis-world-coefficients.npy',mmap_mode='r');g0=np.load(D/'ALL-retained-axis-saved-gaps.npy',mmap_mode='r')
 with np.load(D/'dynamic-source-data.npz')as z:world0=z['original_world_from_object'];times=z['timestamps_s']
 source_frames=0;source_unknowns=0;all_source=[];all_trial=[];axis_count=0;max_axis_error=0.;callback_total=0
 # Raw QP numerical slacks are diagnostics, never original feasibility gates.
 for major in report['major_RETURNS']:
  k=major['major'];prefix=f'major-{k:02d}';raw=major['raw_RETURN'];assert sha(raw['path'])==raw['sha256']
  with np.load(raw['path'])as z:x=z['x'];dual=z['z'];slack=z['s']
  model_dir=B if k==0 else E
  A=sparse.load_npz(model_dir/('NEW-canonical-A-RETURN.npz'if k==0 else prefix+'-A-RETURN.npz'))
  with np.load(model_dir/('NEW-canonical-vectors-RETURN.npz'if k==0 else prefix+'-vectors-RETURN.npz'))as z:b=z['b']
  meta=read(model_dir/('NEW-canonical-metadata-RETURN.json'if k==0 else prefix+'-metadata-RETURN.json'))
  raw_residual=b-A@x-slack
  residual_ref=arrays(prefix+'-NEW-independent-raw-slack-residual-RETURN.npz',residual=raw_residual)
  callbacks=sorted(E.glob(prefix+'-callback-*-RETURN.json'));callback_total+=len(callbacks)
  info=[read(p)['fields']for p in callbacks]
  actual_iters=[a.get('iterations')for a in info]
  r=dict(major=k,rows=A.shape[0],columns=A.shape[1],raw_return_hash_verified=True,
   numerical_slack_closure_max=float(np.max(abs(raw_residual))),raw_slack_residual=residual_ref,
   callback_returns=len(callbacks),callback_iteration_sequence=actual_iters,info_status=major['status'],
   no_exact_primal_or_dual_certificate_or_original_feasibility_claim=True)
  save(prefix+'-NEW-saved-QP-return-binding.json',r);emit(dict(kind='AFTER_NEW_saved_QP_RETURN',actual_RETURN=r))
  assert x.shape==(454,)and dual.shape==slack.shape==b.shape and A.shape[1]==454
  assert len(callbacks)==major['callbacks']and all(a is not None for a in actual_iters)and actual_iters==sorted(actual_iters)
  assert len(b)==meta['rows']and np.isfinite(np.r_[x,dual,slack]).all()
 for trial in report['trial_RETURNS']:
  pid=trial['proposal_id'];i=trial['trial'];major=trial['major'];prefix=f'major-{major:02d}'
  emit(dict(kind='BEFORE_NEW_trial_saved_return_binding',trial=i,proposal_id=pid))
  sealed=trial['sealed_proposal'];assert sha(sealed['path'])==sealed['sha256']
  with np.load(sealed['path'])as z:delta=z['world_delta'];sealed_step=z['raw_QP_step'];alpha=float(z['alpha']);saved_world0=z['original_world_poses']
  with np.load(E/(prefix+'-state-RETURN.npz'))as z:current=z['world_delta'];current_gap=z['gap_proxy']
  with np.load(report['major_RETURNS'][major]['raw_RETURN']['path'])as z:actual_step=z['x']
  assert np.array_equal(sealed_step,actual_step)and np.array_equal(saved_world0,world0)
  assert np.array_equal(delta,current+alpha*actual_step[:-1].reshape(151,3))and alpha==trial['alpha']and alpha in cfg['trial_fractions']
  assert hashlib.sha256(np.ascontiguousarray(delta).tobytes()).hexdigest()==trial['delta_sha256']
  with np.load(trial['geometry_RETURN']['path'])as z:raw_axis=z['all_axis_values'];active=z['active_axis_indices'];gap=z['gap_proxy'];dgeo=z['world_delta']
  assert sha(trial['geometry_RETURN']['path'])==trial['geometry_RETURN']['sha256']and np.array_equal(delta,dgeo)
  # Different scalar product association on the NEW returned point only. This
  # does not rerun the old all-axis constructor/audit or any native geometry.
  errors=np.empty(len(g0));independent_values=np.empty(len(g0));expected_active=np.empty(len(active),np.int64);expected_gap=np.empty(len(gap));maxerr=0.;max_ratio=0.;bad_bounds=0;bad_alias=[]
  for pair,f in enumerate(pair_frames):
   a,b=map(int,ptr[pair:pair+2]);v=axes[a:b];d=delta[int(f)]
   independent=((g0[a:b]+v[:,0]*d[0])+v[:,1]*d[1])+v[:,2]*d[2]
   independent_values[a:b]=independent
   diff=raw_axis[a:b]-independent;errors[a:b]=diff;maxerr=max(maxerr,float(np.max(abs(diff),initial=0)))
   magnitude=abs(g0[a:b])+np.sum(abs(v*d),axis=1)
   # Conservative bound for TWO finite RN sums of <=8 operations each.
   # 4e-15 > 2*gamma8; outward conversion is a numerical audit guard only.
   guard=np.nextafter(4e-15*magnitude+64*np.nextafter(0.,1.),np.inf)
   bad_bounds+=int(np.count_nonzero(abs(diff)>guard))
   if len(diff):max_ratio=max(max_ratio,float(np.max(np.divide(abs(diff),guard,out=np.zeros_like(diff),where=guard>0))))
   local=int(np.argmax(raw_axis[a:b]));expected_active[pair]=a+local;expected_gap[pair]=raw_axis[a+local]
   if int(active[pair])!=a+local or gap[pair]!=raw_axis[a+local]:bad_alias.append(int(pair))
  axis_residual=arrays(pid+'-NEW-independent-all-axis-errors-RETURN.npz',errors=errors,independent_axis_values=independent_values,expected_active_indices=expected_active,expected_gap=expected_gap)
  emit(dict(kind='AFTER_NEW_all_axis_audit_RETURN',proposal_id=pid,actual_RETURN=axis_residual,bad_error_bounds=bad_bounds,bad_alias_pairs=bad_alias))
  assert bad_bounds==0 and not bad_alias
  phase=float(np.maximum(-.001-gap,0).max(initial=0));current_phase=float(np.maximum(-.001-current_gap,0).max(initial=0))
  assert phase==trial['proxy_phase_m']and current_phase==trial['current_phase_m']
  axis_count+=len(raw_axis);max_axis_error=max(max_axis_error,maxerr)
  with np.load(trial['pose_RETURN']['path'])as z:poses=z['world_from_object']
  assert sha(trial['pose_RETURN']['path'])==trial['pose_RETURN']['sha256']
  assert np.array_equal(poses[:,:3,:3],world0[:,:3,:3])and np.array_equal(poses[:,3],world0[:,3])
  assert np.array_equal(poses[:,:3,3],world0[:,:3,3]+delta)
  motion=read(E/(pid+'-NEW-original-motion-RETURN.json'));motion_checks=read(E/(pid+'-NEW-original-motion-decisions-RETURN.json'))
  dt=np.diff(times);linear=np.array([(poses[k+1,:3,3]-poses[k,:3,3])/dt[k]for k in range(150)])
  acc=np.array([(linear[k+1]-linear[k])/((dt[k]+dt[k+1])/2)for k in range(149)])
  independent_motion=dict(maximum_linear_speed_m_s=float(np.linalg.norm(linear,axis=1).max()),maximum_linear_acceleration_m_s2=float(np.linalg.norm(acc,axis=1).max()),p95_linear_speed_m_s=float(np.quantile(np.linalg.norm(linear,axis=1),.95)))
  m_errors={key:independent_motion[key]-motion[key]for key in independent_motion}
  save(pid+'-NEW-independent-motion-RETURN.json',dict(values=independent_motion,errors=m_errors))
  assert all(abs(v)<=1e-12 for v in m_errors.values())
  assert motion_checks=={k:bool(motion[k]<=v+1e-9)for k,v in caps.items()}
  source=read(trial['source_RETURN']['path']);assert sha(trial['source_RETURN']['path'])==trial['source_RETURN']['sha256']
  assert [r['frame']for r in source]==list(range(151))and all(r['proposal_id']==pid for r in source)
  source_counts=dict(IoU=0,median=0,quality=0,unknown=0,cache=0)
  for f,r in enumerate(source):
   nonzero=not np.array_equal(delta[f],np.zeros(3));base=stats[f]
   if not nonzero:
    assert r['status']=='CACHED_F_training_values_BY_HASH'and r['passes_TRAINING_construction']is True;source_counts['cache']+=1;continue
   if r['status'].startswith('UNRESOLVED'):
    assert not r['passes_TRAINING_construction'];source_unknowns+=1;source_counts['unknown']+=1
    all_source.append(dict(trial=i,frame=f,status=r['status'],numeric_values_NOT_invented=True));continue
   assert r['status']=='NEW_training_source_predicate_ONLY'
   ip=E/f'{pid}-f{f:03d}-image-RETURN.npz';mp=E/f'{pid}-f{f:03d}-median-RETURN.npz'
   with np.load(ip)as z:pred=z['predicted_training_mask'];uv=z['uv_float'];integer=z['uv_integer']
   expected_integer=np.clip(uv,-20000,20000).astype(np.int32)
   with np.load(base['partition']['path'])as z:M=z['training_mask'];domain=z['domain']
   I=int((pred&M&domain).sum());U=int(((pred|M)&domain).sum())
   determinant=I*base['training_union']-base['training_intersection']*U
   with np.load(mp)as z:
    distances=z['nearest_vertex_distances'];central=z['central_point_indices'];central_values=z['central_distances'];weights=z['central_weights']
   order=np.argsort(distances,kind='stable');N=len(distances);expected_ids=order[[(N-1)//2,N//2]]
   raw_metric_ref=arrays(pid+f'-f{f:03d}-NEW-saved-metric-map-RETURN.npz',expected_uv_integer=expected_integer,expected_central_indices=expected_ids,expected_central_values=distances[expected_ids],returned_central_indices=central,returned_central_values=central_values)
   emit(dict(kind='AFTER_NEW_saved_metric_map_RETURN',proposal_id=pid,frame=f,actual_RETURN=raw_metric_ref))
   assert np.array_equal(expected_integer,integer)
   assert np.array_equal(central,expected_ids)and np.array_equal(central_values,distances[expected_ids])and np.array_equal(weights,[.5,.5])
   median=float((distances[expected_ids[0]]+distances[expected_ids[1]])/2)
   iou=float(I/max(1,U));quality=float(iou*np.exp(-median/.02));dm=float(median-base['training_mesh_vertex_median_m']);dq=float(quality-base['training_quality'])
   passed=bool(U>0 and determinant>=0 and dm<=0 and dq>=0)
   actual=dict(trial=i,frame=f,I=I,U=U,exact_IoU_determinant=determinant,
    median=median,quality=quality,median_change=dm,quality_change=dq,decision=passed,
    returned_masks_and_distances_ONLY_no_raster_or_NN_replay=True)
   actual_ref=save(pid+f'-f{f:03d}-NEW-source-decision-binding-RETURN.json',actual)
   emit(dict(kind='AFTER_NEW_source_decision_binding_RETURN',proposal_id=pid,frame=f,actual_RETURN=actual_ref))
   assert I==r['training_intersection']and U==r['training_union']and determinant==r['training_IoU_determinant']
   assert median==r['training_median_m']and quality==r['training_quality']and dm==r['training_median_change_m']and dq==r['training_quality_change']
   assert passed==r['passes_TRAINING_construction']
   source_counts['IoU']+=int(determinant<0);source_counts['median']+=int(dm>0);source_counts['quality']+=int(dq<0)
   all_source.append(actual);source_frames+=1
  source_pass=all(x['passes_TRAINING_construction']is True for x in source)
  box=bool(np.isfinite(delta).all()and np.max(abs(delta))<=cfg['world_box_halfwidth_m'])
  accepted=bool(source_pass and all(motion_checks.values())and box and phase<current_phase)
  assert accepted==trial['accepted']==trial['decision']['accepted']
  r=dict(trial=i,proposal_id=pid,all_axis_values_checked=len(raw_axis),maximum_axis_roundoff_closure=maxerr,maximum_error_guard_ratio=max_ratio,
   axis_residual_RETURN=axis_residual,phase_m=phase,current_phase_m=current_phase,
   independent_motion_errors=m_errors,all6_original_motion_decisions=motion_checks,
   actual_source_failure_categories=source_counts,actual_source_failed_frames=[x['frame']for x in source if not x['passes_TRAINING_construction']],
   independently_bound_acceptance=accepted,raw_step_and_poses_unaltered=True)
  save(pid+'-NEW-independent-trial-RETURN.json',r);emit(dict(kind='AFTER_NEW_independent_trial_RETURN',actual_RETURN=r));all_trial.append(r)
 # Method terminal is the current accepted vector, never a rejected best trial.
 with np.load(report['terminal_RETURN']['path'])as z:terminal={k:z[k]for k in z.files}
 assert sha(report['terminal_RETURN']['path'])==report['terminal_RETURN']['sha256']
 if report['accepted_method_steps']==0:
  with np.load(B/'NEW-initial-state-RETURN.npz')as z:
   for k in z.files:assert np.array_equal(terminal[k],z[k])
 assert report['accepted_method_steps']==sum(r['independently_bound_acceptance']for r in all_trial)
 assert report['QP_calls']==len(report['major_RETURNS'])and report['actual_trial_vectors']==len(all_trial)
 save('ALL-NEW-actual-source-decision-bindings.json',all_source);save('ALL-NEW-independent-trial-bindings.json',all_trial)
 result=dict(schema='v2d.NEW-hard-source-filter-saved-return-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  NEW_QP_raw_return_bindings=report['QP_calls'],NEW_actual_info_callbacks=callback_total,
  NEW_trial_identity_raw_pose_motion_and_acceptance_bindings=len(all_trial),
  NEW_returned_axis_values_independently_checked=axis_count,maximum_axis_closure_m=max_axis_error,
  NEW_saved_mask_distance_and_source_decisions=source_frames,typed_source_unknowns=source_unknowns,
  terminal_exactly_initial_F_if_no_accepted_steps=True,accepted_method_steps=report['accepted_method_steps'],
  all_failed_trials_retained_REJECTED_no_best_or_repair=True,
  actual_source_motion_native_swept_motor_grasp_source_accuracy_NOT_certified=True,
  no_function_solver_model_source_raster_NN_FK_native_physics_GPU_replay=True,
  original_F_source_parameters_gates_and_prior_failed_returns_UNCHANGED=True)
 save('report.json',result);emit(dict(kind='AFTER_NEW_saved_return_audit_RETURN',actual_RETURN=result));print(json.dumps(result))
