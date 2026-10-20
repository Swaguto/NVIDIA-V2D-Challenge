"""NEW saved-return audit only; no SOCP, FK, Jacobian or method replay."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,math,os,pickle,time
import numpy as np
from thumb_conic_geometry_v1 import plain,journal
S=Path('/Users/rohithiruku/v2d_scored_baseline_20261017');P=S/'e037-sequential-camera-evaluation-v1';O=S/'e037-sequential-camera-saved-independent-v1';start=time.monotonic();read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
binding=read(S/'e037-sequential-camera-binding-v1/trusted-runtime-binding.json');assert sha(binding['cache_path'])==binding['cache_sha256'];bundle=pickle.loads(Path(binding['cache_path']).read_bytes());system=bundle['system'];initial=bundle['initial_view'];raw=read(P/'ONE-actual-sequential-camera-method-RETURN.json');report=read(P/'report.json');canonical_certificates=[S/'e037-sequential-camera-binding-v1/ONE-first-canonical-independent-certificate.json']+[P/f'{r["label"]}-independent-actual-canonical-certificate.json'for r in report['new_actual_dynamic_canonical_bindings']];coefs=rhs=0
for path in canonical_certificates:
 d=read(path);assert d['all_NEW_actual_canonical_coefficients_RHS_objective_cones_bound']
 for r in d['all_coefficient_certificates']:
  v=r['actual_minus_expected'];g=r['numerical_packing_guard'];assert abs(F(int(v['numerator']),int(v['denominator'])))<=F(int(g['numerator']),int(g['denominator']));coefs+=1
 for r in d['all_RHS_certificates']:
  v=r['actual_minus_expected'];g=r['numerical_packing_guard'];assert abs(F(int(v['numerator']),int(v['denominator'])))<=F(int(g['numerator']),int(g['denominator']));rhs+=1
states={};pending={};decisions=[];accepted=[];qps=[];expected_pending_label=None;new_source_returns=0;merit_errors=[];linear_errors=[];source_pixel_errors=[];direction=[];latest=initial;terminal_sources=None
def derived_view(x,source,halfvalues):
 exactx=list(map(F,x));linear=[]
 for terms,b in zip(system.exact_terms,system.b,strict=True):linear.append(float(sum((a*exactx[j]for j,a in terms),F(0))-F(float(b))))
 linear=np.asarray(linear)/system.linear_scale;entries=[dict(kind='linear',row=i,id=system.rows[i]['kind'],joint=system.rows[i]['joint'],value=float(v))for i,v in enumerate(linear)];pixelerrors=[]
 for i,(s,spec)in enumerate(zip(source,system.specs,strict=True)):
  C=s['C'];B=s['B'];radius=s['radius'];X=[math.fsum(float(B[a,k])*float(C[k])for k in range(3))for a in range(2)];cone=(math.hypot(*X)-radius*C[2])/system.source_scale[i];depth=(.01-C[2])/.01;entries.append(dict(kind='source',factor=i,frame=spec['frame'],link=spec['link'],value=cone,direct_pixel_error=s['direct_image_error_px'],ORIGINAL_cap_px=spec['ORIGINAL_cap_px'],ORIGINAL_image_tolerance_px=1e-6));entries.append(dict(kind='depth',factor=i,value=depth))
  if C[2]>.01:pixelerrors.append(abs(math.hypot(*X)/C[2]-s['direct_image_error_px']))
 for i,u in enumerate(halfvalues):entries.append(dict(kind='half_lower',domain=i,value=(system.half_lo[i]-u)/.05));entries.append(dict(kind='half_upper',domain=i,value=(u-system.half_hi[i])/.05))
 return max([0.]+[r['value']for r in entries]),linear,pixelerrors,entries
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 emit(dict(kind='BEFORE_NEW_saved_return_audit',raw_method_RETURN_sha256=sha(P/'ONE-actual-sequential-camera-method-RETURN.json'),raw_journal_sha256=sha(P/'ONE-all-camera-step-proposals-returns-and-decisions.jsonl'),no_solver_method_actual_FK_Jacobian_or_previous_audit_replayed=True))
 source=initial['source'];m,lin,pe,entries=derived_view(initial['x'],source,initial['half_values']);assert abs(m-initial['merit'])<1e-10;initial_merit=m;latest=dict(x=initial['x'],source=source,halfvalues=initial['half_values'],merit=initial['merit'],entries=entries)
 with(P/'ONE-all-camera-step-proposals-returns-and-decisions.jsonl').open()as f:
  for line in f:
   event=json.loads(line);kind=event.get('kind')
   if kind=='BEFORE_NEW_actual_unlinearized_model_return':pending={};expected_pending_label=event['label']
   elif kind=='AFTER_NEW_trial_URDF_source_RETURN':pending[event['factor_id']]=event['actual_source_RETURN'];new_source_returns+=1
   elif kind=='AFTER_NEW_actual_unlinearized_model_return':
    r=event['returned'];sources=[]
    for i,(cached,spec,orig)in enumerate(zip(system.initial,system.specs,initial['source'],strict=True)):
     ret=pending[spec['id']];sources.append(dict(C=np.asarray(ret['camera_point']),B=orig['B'],radius=ret['constructed_radius_px'],direct_image_error_px=ret['direct_image_error_px']))
    actual,linear,pe,entries=derived_view(r['proposed_angles'],sources,r['retained_half_values']);error=abs(actual-r['merit']);le=float(abs(linear-np.asarray(r['all_original_linear_residuals'])).max());merit_errors.append(error);linear_errors.append(le);source_pixel_errors.extend(pe);certificate=dict(label=event['label'],independent_actual_merit=actual,reported_actual_merit=r['merit'],merit_error=error,maximum_exact_binary_linear_residual_difference=le,maximum_direct_pixel_homogeneous_error=max(pe,default=0.),diagnostics_NOT_candidate_tolerances=True);p=save(event['label']+'-independent-saved-merit.json',certificate);emit(dict(kind='AFTER_NEW_independent_saved_merit',returned=certificate,certificate_sha256=sha(p)));assert error<1e-10 and le<1e-10 and max(pe,default=0.)<1e-9;states[event['label']]=dict(x=np.asarray(r['proposed_angles']),source=sources,halfvalues=r['retained_half_values'],merit=r['merit'],entries=entries)
   elif kind=='AFTER_NEW_unlinearized_Armijo_trial':
    if 'actual_merit'in event:
     inequality=event['actual_merit']<=event['Armijo_bound'];assert inequality==event['accepted'];decisions.append(dict(iteration=event['iteration'],backtrack=event['backtrack'],accepted=event['accepted'],exact_saved_float_comparison_verified=True))
     if event['accepted']:label=f'trial-{event["iteration"]}-{event["backtrack"]}';latest=states[label];accepted.append(label)
   elif kind=='AFTER_NEW_raw_predicted_reduction':
    actual=event['predicted_reduction'];recomputed=latest['merit']-event['actual_rho'];assert actual==recomputed;direction.append(dict(iteration=event['iteration'],actual_rho=event['actual_rho'],actual_current_merit=latest['merit'],actual_predicted_reduction=actual,strict_positive_descent=bool(actual>0)))
   elif kind=='AFTER_NEW_SOCP_solve_RETURN':qps.append(dict(label=event['label'],status=event['actual_RETURN']['status'],iterations=event['actual_RETURN']['iterations'],raw_return_file_sha256=sha(P/f'{event["label"]}-actual-SOCP-RETURN.json')))
 assert np.array_equal(raw['terminal_angles'],latest['x'])and raw['terminal_merit']==latest['merit']and raw['subproblem_calls']==len(qps)and raw['trial_evaluations']==len(states)and raw['status']=='UNRESOLVED_no_predicted_descent'and direction[-1]['actual_predicted_reduction']<=0
 terminal_positive=sorted([v for v in latest['entries']if v['value']>0],key=lambda v:-v['value']);save('actual-terminal-positive-model-residuals.json',dict(all_positive_residuals=terminal_positive,last_accepted_saved_trial=accepted[-1],not_family_impossibility_or_physical_force=True))
 save('all-independent-decisions-and-raw-direction-identities.json',dict(Armijo=decisions,raw_predicted_reductions=direction,actual_SOCPreturns=qps,accepted_trial_labels=accepted,no_status_dual_slack_or_angle_return_repaired=True))
 summary=dict(schema='v2d.new-sequential-camera-saved-return-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_saved_coefficients_merits_decisions_and_terminal_provenance_checks_passed=True,new_saved_canonical_certificates=len(canonical_certificates),saved_canonical_coefficient_residuals_checked=coefs,saved_canonical_RHS_residuals_checked=rhs,new_saved_unlinearized_views_checked=len(states),new_saved_actual_URDF_source_returns_retained=new_source_returns,new_Armijo_decisions_checked=len(decisions),accepted_trials=len(accepted),raw_predicted_reduction_identities=len(direction),actual_subproblem_statuses=[v['status']for v in qps],maximum_saved_merit_difference=max(merit_errors,default=0.),maximum_exact_binary_linear_residual_difference=max(linear_errors,default=0.),maximum_direct_pixel_homogeneous_difference_px=max(source_pixel_errors,default=0.),actual_initial_merit=initial_merit,actual_terminal_merit=latest['merit'],actual_terminal_positive_constraint_count=len(terminal_positive),actual_last_raw_rho=direction[-1]['actual_rho'],actual_last_predicted_reduction=direction[-1]['actual_predicted_reduction'],negative_prediction_is_NOT_certified_nonlinear_stationarity_or_family_exclusion=True,original_method_status_RETAINED=raw['status'],actual_model_candidate_returned=False,all_terminal_and_intermediate_angles_REJECTED=True,new_method_calls=0,new_solver_calls=0,new_actual_URDF_or_Jacobian_calls=0,new_native_position_checks=0,new_GPU_jobs=0,new_PPO_steps=0,wall_seconds=time.monotonic()-start);emit(dict(kind='AFTER_NEW_saved_return_audit',returned=summary));summary['all_outputs_sha256']={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log','report.json']};save('report.json',summary);print(json.dumps({k:v for k,v in summary.items()if k!='all_outputs_sha256'}))
