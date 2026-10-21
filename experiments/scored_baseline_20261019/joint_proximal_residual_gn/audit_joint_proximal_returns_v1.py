"""Independent NEW saved-return audit; no method/solve/geometry/source replay."""
from joint_saved_mean_bound_v1 import S,C,sha,save,typed
from joint_lossless_returns_v1 import seal_return
from pathlib import Path
import numpy as np,json,re
O=S/'e037-joint-proximal-evaluation-v1';D=S/'e037-joint-proximal-return-audit-v1';PRE=S/'e037-joint-proximal-preflight-v1'
files=sorted(p for p in O.iterdir() if p.is_file());save(D/'NEW-ALL-actual-return-bindings-BEFORE.json',{p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in files})
report=json.loads((O/'report.json').read_text());assert report['raw_trial_sealed'] and report['complete_NEW_trial_query']
get=lambda pattern:typed(next(O.glob(pattern)))
inputs=get('*-method-input-BEFORE.json');raw=get('*-ONE-raw-joint-trial-BEFORE-query.json');step=get('*-ALL500729-backsub-step-RETURN.json');solve=get('*-actual-two-RHS-RN-guard-RETURN.json');phase=get('*-original-phase-domain-quadratic-RETURN.json')
trial=raw['w'];w0=inputs['w0'];d=step['step']
seal_return(D,'NEW-actual-step-trial-solve-return-identity-RETURN',dict(trial_hash_matches=raw['w_bytes_sha256']==__import__('hashlib').sha256(trial.tobytes()).hexdigest(),
 trial_equals_raw_F_start_plus_step=bool(np.array_equal(trial,w0+d)),full_coordinate_count=len(trial),all_saved_guard_arrays_finite=bool(np.isfinite(solve['residual']).all() and np.isfinite(solve['conditional_RN_outer']).all()),
 all_saved_solve_guard_decisions=bool(np.all(np.abs(solve['residual'])<=solve['conditional_RN_outer'])),
 actual_phase=float(trial[1522]),declared_original_target=float(phase['target_phase']),raw_phase_defect_RETAINED=True))
assert raw['w_bytes_sha256']==__import__('hashlib').sha256(trial.tobytes()).hexdigest() and np.array_equal(trial,w0+d) and np.isfinite(solve['residual']).all() and np.isfinite(solve['conditional_RN_outer']).all() and np.all(np.abs(solve['residual'])<=solve['conditional_RN_outer'])
F=get('*-ONE-trial-ALL-F-RETURN.json')['F'];decision=get('*-ALL-construction-decision-RETURN.json');motion=get('*-ALL6-actual-object-motion-RETURN.json')
with np.load(S/'e037-joint-method-preflight-v1/NEW-original-objective-and-cone-layout-RETURN.npz',allow_pickle=False) as z:cones=[tuple(map(int,v)) for v in z['cones']]
mask=np.ones(len(F),dtype=np.bool_);margins=[]
for start,width in cones:mask[start:start+width]=False;v=F[start:start+width];margins.append(float(-v[0]-np.linalg.norm(v[1:])))
assert np.array_equal(np.array(margins),decision['ALL1245_cone_margins'])
assert float(F[mask].max())==decision['maximum_affine_F'] and min(margins)==decision['minimum_negative_F_cone_margin']
assert int(np.count_nonzero(F[mask]>0))==decision['violated_affine_rows'] and np.count_nonzero(np.array(margins)<0)==decision['violated_cones']
# Full NEW row groups, all read from the saved F vector, not operator calls.
nobj=4551571;nrob=8114;src=nobj+nrob;gstart=src+852
counts=dict(object_nonnegative=int(np.count_nonzero(F[:4547216]>0)),robot_linear=int(np.count_nonzero(F[nobj:src]>0)),source_SOC=int(np.count_nonzero(np.array(margins)[1051:]<0)),
 object_SOC=int(np.count_nonzero(np.array(margins)[:1051]<0)),source_strict_depth=int(np.count_nonzero(F[src+582:src+776]>=0)),halfdomain=int(np.count_nonzero(F[src+776:gstart]>0)),geometric_pair_ideal=int(np.count_nonzero(F[gstart:]>0)))
seal_return(D,'NEW-ALL4810140-row-and1245-cone-decision-binding-RETURN',dict(ALL_F_values=len(F),group_failure_counts=counts,all_returned_F_finite=bool(np.isfinite(F).all()),
 original_decision=decision,no_physical_native_distance_scope=True))
with np.load(S/'e037-joint-executable-interface-v1/NEW-interface-constants-RETURN.npz',allow_pickle=False) as z:caps=z['object_caps']
vals=[np.linalg.norm(motion['arrays'][name],axis=1).max(initial=0) for name in ['linear_velocity','angular_velocity','linear_acceleration','angular_acceleration']]+[np.quantile(np.linalg.norm(motion['arrays'][name],axis=1),.95) for name in ['linear_velocity','angular_velocity']]
assert np.array_equal(vals,motion['values']) and [bool(x<=cap+1e-9) for x,cap in zip(vals,caps)]==motion['passes']
seal_return(D,'NEW-ALL6-SAVED-motion-decision-binding-RETURN',dict(values=vals,passes=motion['passes'],original_comparison=1e-9,actual_p95_from_saved_velocity_arrays=True))
frames=[];pairs=0;commands=0;slopes=0;unknown=0
for p in sorted(O.glob('*-ONE-trial-frame-*-ALL-pair-RETURN.json')):
 a=typed(p);f=int(re.search(r'frame-(\d+)',p.name).group(1));frames.append(f)
 assert np.array_equal(a['pairs'],np.arange(1653)) and a['chart'].shape==(1653,2) and np.array_equal(a['chart'],trial[1523:].reshape(151,1653,2)[f])
 assert len(a['valid'])+len(a['unknown_raw'])==1653;unknown+=len(a['unknown_raw']);pairs+=1653
 cp=a['command_ptr'];assert len(cp)==1654 and cp[-1]==len(a['columns'])==len(a['mean_command']);commands+=int(cp[-1])
 for side in ['A','B']:
  ptr=a[side+'_ptr'];assert ptr[-1]==len(a[side+'_ties'])==len(a[side+'_direction_slopes']) and len(ptr)==len(a['valid'])+1
  sp=a[side+'_slope_ptr'];assert len(sp)==len(a['columns'])+1 and sp[-1]==len(a[side+'_slopes']);slopes+=int(sp[-1])
 assert a['mean_is_SEARCH_only'].all()
assert frames==list(range(151)) and pairs==249603 and unknown==0
warnings=(O/'worker.log').read_text().count('RuntimeWarning:')
result=dict(passed=True,status=report['status'],ALL_saved_constraint_values=len(F),ALL_saved_cones=len(cones),ALL_saved_pair_instances=pairs,ALL_saved_command_incidences=commands,ALL_saved_tied_slopes=slopes,ALL151_frames=True,
 row_failure_counts=counts,actual_normalized_phase=float(trial[1522]),original_phase_algorithm_target=float(phase['target_phase']),phase_return_NOT_repaired=True,
 ALL6_actual_object_motion_pass=all(motion['passes']),actual_matmul_warnings_retained=warnings,checked_actual_solve_guard_and_F_returns_finite=True,
 warning_cause_and_future_reliability_UNPROVEN=True,no_method_factor_solve_Ax_adjoint_source_geometry_native_or_motion_operator_replay=True,
 terminal_J_E_NOT_used_for_update_and_old_bound_NOT_inherited=True,no_native_or_physical_improvement_or_trajectory_qualification=True)
seal_return(D,'NEW-independent-SAVED-return-audit-RETURN',result);save(D/'report.json',result)
