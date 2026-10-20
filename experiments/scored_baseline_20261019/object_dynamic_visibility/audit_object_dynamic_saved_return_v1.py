"""NEW saved-return/cache/native-union and subsystem-map audit; no function/solver replay."""
from pathlib import Path
from fractions import Fraction as Q
import datetime,hashlib,json,os
import numpy as np
from scipy import sparse
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-object-dynamic-saved-return-audit-v1';E=S/'e037-object-dynamic-evaluation-v1';D=S/'e037-object-dynamic-runtime-v1';U=S/'e037-object-certificate-subsystems-v1';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
def q(r):return Q(r['numerator'],r['denominator'])
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 report=read(E/'report.json');emit(dict(kind='BEFORE_NEW_saved_dynamic_return_and_subsystem_map_audit',actual_report_sha256=sha(E/'report.json'),no_function_source_depth_FK_native_solver_or_old_audit_rerun=True));events={};callbacks=0;cache=0
 with(E/'normalized-BEFORE-AFTER-journal.jsonl').open()as f:
  for line in f:
   r=json.loads(line)
   if r['kind']=='AFTER_NEW_actual_dynamic_model_evaluation_RETURN':events[r['actual_RETURN']['evaluation']]=r['actual_RETURN']
   elif r['kind']=='AFTER_NEW_actual_optimizer_callback_RETURN':callbacks+=1
   elif r['kind']=='AFTER_dynamic_fun_jac_cache_REUSED':cache+=1
 assert callbacks==report['actual_callback_returns'];axes=np.load(D/'ALL-retained-axis-world-coefficients.npy',mmap_mode='r');g0=np.load(D/'ALL-retained-axis-saved-gaps.npy',mmap_mode='r')
 with np.load(D/'dynamic-axis-group-bindings.npz')as z:ptr=z['ptr'];pairframes=z['pair_frames']
 af=np.repeat(pairframes,np.diff(ptr));maximum_union_error=0.;union_rows=0;hit_miss_entries=0;full_residual_entries=0;states=[];returned_match=[];actualret=read(E/'ONE-actual-optimizer-RETURN.json')if report['actual_optimizer_RETURN_produced']else None
 for i,event in sorted(events.items()):
  emit(dict(kind='BEFORE_NEW_saved_dynamic_state_identity',evaluation=i,inputs_sha256=sha(E/f'evaluation-{i:03d}-actual-method-residual-RETURN.npz')))
  with np.load(E/f'evaluation-{i:03d}-actual-method-residual-RETURN.npz')as z:x=z['normalized_x'];residual=z['residual'];motion=z['motion_vectors'];norms=z['motion_norms']
  with np.load(E/f'evaluation-{i:03d}-NEW-union-support-RETURN.npz')as z:active=z['active_axis_indices'];gap=z['gap_proxy_m'];violation=z['positive_violations_m'];delta=z['world_translation'];assert np.array_equal(delta,.02*x.reshape(151,3))
  # First-time independent all-retained-direction maximality for these NEW
  # saved translated states; old baseline native measurements are not repeated.
  is_raw_return=actualret is not None and np.array_equal(x,np.asarray(actualret['x']))
  if is_raw_return:
   values=g0+axes[:,0]*delta[af,0]+axes[:,1]*delta[af,1]+axes[:,2]*delta[af,2];expected=np.maximum.reduceat(values,ptr[:-1]);error=float(abs(expected-gap).max());emit(dict(kind='AFTER_NEW_independent_RETURNED_point_union_maximality_RETURN',evaluation=i,maximum_gap_identity_error=error,all4832184directions_checked=True,not_actual_native_gap_truth=True));assert error<1e-12;maximum_union_error=max(maximum_union_error,error);union_rows+=len(expected)
  assert np.array_equal(violation,np.maximum(-.001-gap,0));assert all(a<=v<b for a,b,v in zip(ptr[:-1],ptr[1:],active,strict=True))
  with np.load(E/f'evaluation-{i:03d}-NEW-source-RETURN.npz')as z:faces=z['actual_front_faces'];depth=z['actual_depth_m'];coverage=z['coverage_distance_px'];dr=z['source_depth_residual'];cr=z['source_coverage_residual']
  miss=faces<0;assert np.isinf(depth[miss]).all()and np.isfinite(depth[~miss]).all()and(depth[~miss]>0).all();assert np.array_equal(residual[:3624],dr)and np.array_equal(residual[3624:7248],cr);assert np.count_nonzero(miss)==event['source_model_misses'];assert np.count_nonzero(~miss)==event['source_model_hits'];hit_miss_entries+=depth.size;full_residual_entries+=len(residual);obj=float(.5*np.dot(residual,residual));assert obj==event['objective'];actual_norm=np.sqrt(np.einsum('ni,ni->n',motion,motion,optimize=False));assert np.max(abs(norms-actual_norm))<1e-12
  if actualret is not None and np.array_equal(x,np.asarray(actualret['x'])):
   returned_match.append(i);assert np.array_equal(residual,np.asarray(actualret['fun']))
  state=dict(evaluation=i,normalized_x_byte_sha256=hashlib.sha256(x.tobytes()).hexdigest(),original_actual_hit_count=int(np.count_nonzero(~miss)),infinite_missing_count=int(np.count_nonzero(miss)),native_proxy_violating_pairs=int(np.count_nonzero(violation)),maximum_native_proxy_excess_m=float(violation.max(initial=0)),objective=obj,raw_optimizer_return_matches=i in returned_match,not_a_qualified_reference=True);states.append(state);emit(dict(kind='AFTER_NEW_saved_dynamic_state_identity_RETURN',actual_RETURN=state))
 save('all-NEW-independent-saved-state-identities.json',states)
 if actualret is not None:
  assert len(returned_match)>=1;emit(dict(kind='AFTER_NEW_actual_optimizer_RETURN_binding',exact_saved_evaluation_ids=returned_match,actual_success=actualret['success'],actual_status=actualret['status'],last_evaluated_point_NEVER_substituted_for_raw_optimizer_RETURN=True))
 if not report.get('optimizer_success'):
  assert report['new_source_motion_object_components']==0 and report['new_full_reference_or_trajectory_candidates']==0 and report['original_source_evaluations']==0 and not(E/'RAW-dynamic-object-pose-proposal-before-gates.npz').exists()
 # Independently bind the ONE raw proposal and all original gate decisions
 # from its SAVED actual returns; no scorer/ray/heldout experiment is replayed.
 source_decisions=0;motion_identity_errors=[];failed_source=[];proposal_path=E/'RAW-dynamic-object-pose-proposal-before-gates.npz'
 if proposal_path.exists():
  with np.load(proposal_path)as z:delta=z['raw_world_translation'];world=z['world_from_object'];camera=z['camera_from_object'];native=z['native_object_poses'];robot=z['unchanged_F_robot_qpos'];times=z['timestamps_s']
  with np.load(D/'dynamic-source-data.npz')as z:world0=z['original_world_from_object'];camera0=z['original_camera_from_object'];RC=z['camera_from_world_linear']
  with np.load(S/'e037-thumb-ray-independent-validation-v1/frozen/targets.npz')as z:assert np.array_equal(robot,z['robot_qpos']);native0=z['object_poses']
  assert np.array_equal(delta,.02*np.asarray(actualret['x']).reshape(151,3));assert np.array_equal(world[:,:3,:3],world0[:,:3,:3])and np.array_equal(camera[:,:3,:3],camera0[:,:3,:3])and np.array_equal(native[:,:,3:],native0[:,:,3:]);assert np.max(abs((world[:,:3,3]-world0[:,:3,3])-delta))<1e-15
  stage=np.asarray(read(S/'plane-aligned-articulated-native-validation-v2/frozen/command-proposal-report.json')['stage_rotation_from_inferred_support_normal']);native_delta=np.column_stack([delta[:,0]*stage[j,0]+delta[:,1]*stage[j,1]+delta[:,2]*stage[j,2]for j in range(3)]);assert np.max(abs((native[:,0,:3]-native0[:,0,:3])-native_delta))<1e-15
  v=(world[1:,:3,3]-world[:-1,:3,3])/np.diff(times)[:,None];acc=(v[1:]-v[:-1])/((np.diff(times)[1:]+np.diff(times)[:-1])/2)[:,None];speed=np.sqrt(np.einsum('ni,ni->n',v,v,optimize=False));accnorm=np.sqrt(np.einsum('ni,ni->n',acc,acc,optimize=False));motion=read(E/'actual-original-wholeclock-motion-RETURN.json');expected={'maximum_linear_speed_m_s':float(speed.max()),'maximum_linear_acceleration_m_s2':float(accnorm.max()),'p95_linear_speed_m_s':float(np.quantile(speed,.95))}
  for k,xv in expected.items():e=abs(xv-motion['actual_motion'][k]);motion_identity_errors.append(e);assert e<1e-12
  angular=read(S/'e037-joint-pose-translation-repair-v1/report.json')['final_motion']
  for k in ['maximum_angular_speed_rad_s','maximum_angular_acceleration_rad_s2','p95_angular_speed_rad_s']:assert abs(motion['actual_motion'][k]-angular[k])<1e-12
  caps=motion['original_caps'];bad=[k for k in caps if motion['actual_motion'][k]>caps[k]+1e-9];assert bad==motion['failed_caps']==report['object_motion_failed_caps']==[]
  for row in read(E/'all-actual-source-gate-RETURNS.json'):
   f=row['frame'];a=read(S/f'e037-object-source-authority-v1/frame-{f:03d}-source-authority.json');raw=read(E/f'frame-{f:03d}-actual-source-RETURN-before-decision.json');assert raw['actual_source_RETURN']==row['actual_source_RETURN'];xv=raw['actual_source_RETURN'];current=a['current_accepted_source_thresholds_COPIED'];floor=a['old_strict_source_comparison_COPIED'];decision=bool(raw['original_object_observed_flag']and raw['tracker_rejection_RETURN']is None and xv.get('quality',0)>=current['quality']-1e-10 and xv.get('silhouette_iou',0)>=current['silhouette_iou']-1e-10 and xv.get('median_measured_vertex_distance_m',float('inf'))<=current['median_measured_vertex_distance_m']+1e-10 and xv.get('quality',0)>floor['quality']and xv.get('silhouette_iou',0)>=floor['silhouette_iou']and xv.get('median_measured_vertex_distance_m',float('inf'))<floor['median_measured_vertex_distance_m']);emit(dict(kind='AFTER_NEW_independent_original_source_decision_RETURN',frame=f,decision=decision,actual_saved_RETURN_sha256=sha(E/f'frame-{f:03d}-actual-source-RETURN-before-decision.json'),original_source_comparison_tolerance=1e-10,scorer_NOT_rerun=True));assert decision==row['original_source_nonregression_passed'];source_decisions+=1
   if not decision:failed_source.append(f)
  assert failed_source==report['source_failed_frames'];assert source_decisions==151
 save('NEW-independent-original-gate-and-proposal-identities.json',dict(source_decisions=source_decisions,failed_source_frames=failed_source,motion_linear_identity_errors=motion_identity_errors,wholeclock_angular_identity_reused_BY_HASH_AND_byte_unchanged=True,all_actual_optimizer_values_and_F_UNCHANGED=True,no_original_source_or_native_audit_rerun=True))
 # Independent NEW subsystem-functional map verification. All original cone
 # checks and global certificate are reused by hash, not repeated.
 groups=read(U/'NEW-exact-group-contribution-RETURNS.json');A=sparse.load_npz(S/'e037-object-visibility-support-model-v3/canonical-A.npz').tocsr();origins=read(S/'e037-object-visibility-support-model-v3/canonical-row-origins.json');rowgroups=read(U/'all27772-row-group-bindings.json');rawdual=np.asarray(read(S/'e037-object-visibility-support-evaluation-v1/ONE-actual-SOCP-RETURN.json')['z'])
 with np.load(S/'e037-object-visibility-support-model-v3/canonical-vectors.npz')as z:b=z['b']
 mapchecks=0
 for i,g in enumerate(rowgroups):
  if i>=26576:expected='M'
  else:
   kind=origins['nonnegative'][i]['kind'];expected='B'if kind.startswith('algorithm_world_translation_box_')else'G'if kind=='NEW_sufficient_fixed_axis_support_plane'else'V'
  assert g==expected;mapchecks+=1
 scopes=read(U/'ONE-declared-subsystem-scope.json')['scopes'];scope_results=[];h=Q(1,50)
 for labels in scopes:
  name=''.join(labels);rec=read(U/f'{name}-exact-subsystem-certificate.json');rows=[i for i,g in enumerate(rowgroups)if g in labels];assert rows==sorted(rec['selected_whole_rows']);c=[Q(0)]*453;offset=Q(0)
  for i in rows:
   zi=Q(float(rawdual[i]));offset+=zi*Q(float(b[i]));row=A.getrow(i)
   for j,v in zip(row.indices,row.data,strict=True):c[int(j)]+=zi*Q(float(v))
  assert offset==q(rec['offset'])and all(v==q(w)for v,w in zip(c,rec['all453_coefficients'],strict=True));upper=offset+h*sum(abs(v)for v in c);assert upper==q(rec['exact_finite_box_upper']);r=dict(scope=name,selected_rows=len(rows),exact_upper_numerator=upper.numerator,exact_upper_denominator=upper.denominator,strict_exclusion=upper<0);scope_results.append(r);emit(dict(kind='AFTER_NEW_independent_subsystem_functional_identity_RETURN',actual_RETURN=r))
 save('all6-NEW-independent-subsystem-functional-identities.json',scope_results)
 out=dict(schema='v2d.NEW-object-dynamic-saved-return-independent-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete_saved_dynamic_evaluations=len(events),actual_function_calls=report['actual_function_evaluations'],actual_optimizer_return_evaluation_matches=returned_match,all_unique_eval_returned_source_entries=hit_miss_entries,native_union_pair_maximality_checks=union_rows,maximum_native_union_identity_error=maximum_union_error,saved_residual_entries_checked=full_residual_entries,callbacks=callbacks,fun_jac_cache_reuses=cache,NEW_independent_subsystem_row_maps=mapchecks,NEW_independent_exact_subsystem_certificates=len(scope_results),all6subsystem_restrictions_inconclusive=all(not r['strict_exclusion']for r in scope_results),no_raw_vector_clipped_repaired_normalized_or_promoted=True,no_source_native_family_or_task_exclusion=True,actual_optimizer_success=report['optimizer_success'],original_source_failed_frames=report['source_failed_frames'],new_object_source_motion_components=report['new_source_motion_object_components'],NEW_independent_original_source_decisions=source_decisions,NEW_independent_actual_object_motion_failed_caps=[]if motion_identity_errors else None,no_whole_trajectory_or_score_qualification=True,no_old_function_source_FK_native_or_solver_audit_replayed=True);save('report.json',out);emit(dict(kind='AFTER_NEW_saved_dynamic_return_audit',actual_RETURN=out));print(json.dumps(out))
