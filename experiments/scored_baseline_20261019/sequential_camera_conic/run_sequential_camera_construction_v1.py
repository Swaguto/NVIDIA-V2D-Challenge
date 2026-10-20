"""ONE declared bounded episode method; all actual steps checked before solve."""
from pathlib import Path
import datetime,hashlib,json,os,pickle,time
import numpy as np
import sequential_camera_conic_v1 as method
from sequential_camera_precomputed_v1 import construct_precomputed
from camera_step_binding_v1 import exact_expected_step,validate_actual_step
from thumb_conic_geometry_v1 import plain,journal
S=Path('/Users/rohithiruku/v2d_scored_baseline_20261017');C=S/'compute';P=S/'e037-sequential-camera-binding-v1';O=S/'e037-sequential-camera-evaluation-v1';start=time.monotonic();read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
binding=read(P/'trusted-runtime-binding.json');preflight=read(P/'report.json');assert preflight['all_NEW_runtime_canonical_and_cache_bindings_passed']and binding['configuration']==method.CONFIG
for name,digest in binding['source_sha256'].items():assert sha(name)==digest
assert sha(binding['cache_path'])==binding['cache_sha256']and sha(P/'ONE-first-canonical-independent-certificate.json')==binding['first_canonical_certificate_sha256']
# Only a locally produced, SHA-bound private cache is loaded. No external pickle.
bundle=pickle.loads(Path(binding['cache_path']).read_bytes());system=bundle['system'];initial_view=bundle['initial_view'];first_problem=bundle['initial_problem'];assert np.array_equal(initial_view['x'],system.x0)and initial_view is system.cached_initial_view
state=dict(current_view=initial_view,pending_view=None,pending_sources={},qp_returns=[],new_canonical_certificates=[],trials=0)
declaration=dict(schema='v2d.ONE-new-sequential-camera-episode-declaration.v1',declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configuration=method.CONFIG,actual_angle_variables=465,original_linear_rows=8114,camera_source_factors=194,retained_halfangle_domain_factors=38,canonical_rows=9897,canonical_columns=466,initial_angles='original_F_UNCHANGED,no_clipping,no_failed_intermediate_or_old_pose_seed',saved_runtime_binding_sha256=sha(P/'trusted-runtime-binding.json'),preflight_first_canonical_certificate_sha256=binding['first_canonical_certificate_sha256'],old12_and_NEW2cache_contracts_reused_BY_HASH=True,local_step_cones_NOT_global_exclusion=True,all_original_model_gates_tolerances_authority_and_fixed_components_UNCHANGED=True,no_restart_warmstart_continuation_or_tolerance_profile_limit_sweep=True,independent_actual_model_acceptance_required_before_any_proposal=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0);save('ONE-declared-sequential-camera-construction.json',declaration)
with(O/'ONE-all-camera-step-proposals-returns-and-decisions.jsonl').open('w')as log:
 def emit(v):
  journal(log,v) # Every raw return is durable BEFORE checks or further numerical work.
  kind=v.get('kind')
  if kind=='BEFORE_NEW_actual_unlinearized_model_return':state['pending_x']=np.asarray(v['proposed_angles']).copy();state['pending_label']=v['label'];state['pending_sources']={}
  elif kind=='AFTER_NEW_trial_URDF_source_RETURN':state['pending_sources'][v['factor_id']]=v['actual_source_RETURN']
  elif kind=='AFTER_NEW_actual_unlinearized_model_return':
   returned=v['returned'];source=[]
   for i,(spec,cached,original)in enumerate(zip(system.specs,system.initial,initial_view['source'],strict=True)):
    raw=state['pending_sources'][spec['id']];source.append(dict(id=spec['id'],columns=cached['columns'],C=np.asarray(raw['camera_point']),J=np.asarray(raw['camera_jacobian'])[:,cached['joints']],B=original['B'],radius=raw['constructed_radius_px'],direct_image_error_px=raw['direct_image_error_px']))
   state['pending_view']=dict(x=np.asarray(returned['proposed_angles']),source=source,linear_violation=np.asarray(returned['all_original_linear_residuals']),half_values=np.asarray(returned['retained_half_values']),half_derivatives=np.asarray(returned['retained_half_derivatives']),merit=returned['merit']);assert np.array_equal(state['pending_x'],state['pending_view']['x'])
  elif kind=='AFTER_NEW_unlinearized_Armijo_trial':
   if v.get('accepted'):state['current_view']=state['pending_view']
   state['trials']+=1
  elif kind=='BEFORE_NEW_camera_SOCP_canonical_step':
   assert np.array_equal(v['actual_angles'],state['current_view']['x'])
  elif kind=='AFTER_NEW_camera_SOCP_canonical_step':
   expected=exact_expected_step(system,state['current_view']);certificate=validate_actual_step(v['returned'],expected,system);p=save(f'{v["label"]}-independent-actual-canonical-certificate.json',certificate);r=dict(label=v['label'],certificate_sha256=sha(p),coefficient_identities=certificate['coefficient_identities'],RHS_identities=certificate['RHS_identities'],before_solver_constructor_and_solve=True);state['new_canonical_certificates'].append(r);journal(log,dict(kind='AFTER_NEW_independent_actual_camera_step_binding',returned=r))
  elif kind=='AFTER_NEW_SOCP_solve_RETURN':
   r=v['actual_RETURN'];p=save(f'{v["label"]}-actual-SOCP-RETURN.json',r);state['qp_returns'].append(dict(label=v['label'],actual_status=r['status'],iterations=r.get('iterations'),solve_time=r.get('solve_time'),actual_RETURN_sha256=sha(p)))
  elif kind=='AFTER_ONE_NEW_sequential_camera_construction_RETURN':save('ONE-actual-sequential-camera-method-RETURN.json',v['actual_RETURN'])
 emit(dict(kind='BEFORE_NEW_episode_call',declaration=declaration,preflight_initial_view_and_first_canonical_REUSED_BY_HASH=True,first_canonical_already_independently_verified=True))
 result=construct_precomputed(system,initial_view,first_problem,emit,S/'e037-direct-SDP-runtime-v2/site-packages')
 if result['proposal_returned']:
  q=system.base.copy()
  for variable in system.variables:q[variable['frame'],1,variable['command_column']]=result['terminal_angles'][variable['column']]
  emit(dict(kind='BEFORE_NEW_UNQUALIFIED_proposal_write',terminal_angles=result['terminal_angles'],all_independent_wholeclock_source_URDF_root_motion_native_self_swept_motor_grasp_checks_PENDING=True));path=O/'UNQUALIFIED-actual-model-source-motion-proposal.npz';np.savez_compressed(path,robot_qpos=q);path.chmod(0o600);emit(dict(kind='AFTER_NEW_UNQUALIFIED_proposal_write',path=str(path),sha256=sha(path),not_reference_or_physical_qualification=True))
 summary=dict(schema='v2d.ONE-sequential-camera-episode-result.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_episode_method_calls=1,actual_method_status=result['status'],actual_major_iterations=result['major_iterations'],actual_subproblem_calls=result['subproblem_calls'],actual_trial_evaluations=result['trial_evaluations'],actual_exact_acceptance_checks=result['exact_acceptance_checks'],actual_method_seconds=result['actual_seconds'],initial_actual_merit=initial_view['merit'],actual_terminal_merit=result['terminal_merit'],actual_model_proposal_returned=result['proposal_returned'],all_unsuccessful_returns_and_intermediates_REJECTED=not result['proposal_returned'],all_solver_statuses_retained=state['qp_returns'],initial_canonical_binding_REUSED_BY_HASH=True,new_actual_dynamic_canonical_bindings=state['new_canonical_certificates'],all_new_canonical_returns_saved_and_checked_BEFORE_solver_calls=True,all_raw_primal_dual_slack_returns_UNCHANGED=True,no_original_family_or_task_exclusion=True,source_URDF_wholeclock_root_finger_native_motor_grasp_qualification_PENDING=True,current_F_separate_components_source_missing_heldout_model_parameters_caps_centers_tolerances_gates_UNCHANGED=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
 emit(dict(kind='AFTER_NEW_episode_scope',returned=summary));summary['all_outputs_sha256']={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log','report.json']};save('report.json',summary);print(json.dumps({k:v for k,v in summary.items()if k not in['all_outputs_sha256','new_actual_dynamic_canonical_bindings','all_solver_statuses_retained']}))
