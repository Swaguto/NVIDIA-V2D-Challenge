"""SOLE declared two-update NEW whole-joint construction; no retries or best state."""
from joint_primal_dual_method_v1 import *
from joint_method_pair_columns_v1 import MethodRecorder
import traceback
D=S/'e037-joint-primal-dual-evaluation-v1';PREF=S/'e037-joint-method-preflight-v1';start=time.monotonic();deadline=start+150
record=MethodRecorder(D);steps_done=0;oracles_done=0;status='UNRESOLVED_before_constructor'
paths=[PREF/'NEW-complete-method-declaration-RETURN.json',PREF/'NEW-constant-ALL-row-column-metric-bounds-RETURN.npz',PREF/'NEW-original-objective-and-cone-layout-RETURN.npz',
 S/'e037-joint-method-binding-v1/report.json',S/'e037-joint-method-pair-columns-binding-v1/report.json',C/'joint_primal_dual_method_v1.py',C/'joint_method_pair_columns_v1.py',
 S/'e037-joint-executable-interface-v1/NEW-interface-input-bindings-RETURN.json']
save(D/'NEW-fixed-evaluation-declaration-BEFORE.json',dict(input_sha256={str(p):sha(p) for p in paths},updates=2,terminal_step_ONLY=2,unique_oracles_max=3,
 method_wall_seconds=150,outer_guard_seconds=240,RSS_bytes=2*1024**3,one_thread=True,no_restart_best_selection_or_old_initializer=True,
 all4810140_rows_and249603_pair_scopes_REQUIRED=True,actual_training_POST_deferred_until_terminal_gate_pass=True))
assert json.loads((S/'e037-joint-method-binding-v1/report.json').read_text())['passed']
assert json.loads((S/'e037-joint-method-pair-columns-binding-v1/report.json').read_text())['passed']
variables=json.loads((M/'continuous-angle-variables.json').read_text());w=np.zeros(NVAR);w[:465]=[v['initial_F_angle_rad'] for v in variables];y=np.zeros(NROW)
record('NEW-initial-input-BEFORE-constructor',dict(w=w,y=y,initialization='F free angle bytes; object eta-u-sigma-phase zero; continuous geometric chart zero',fixed_root_middle='SAME saved UNQUALIFIED model hypotheses'))
try:
 api=JointExecutableInterface(S/'e037-joint-executable-interface-v1')
 record('NEW-method-runtime-constructor-RETURN',api.__dict__)
 with np.load(PREF/'NEW-constant-ALL-row-column-metric-bounds-RETURN.npz',allow_pickle=False) as z:
  const_rows=np.r_[z['object_row'],z['robot_row']];const_cols=z['column']
 with np.load(PREF/'NEW-original-objective-and-cone-layout-RETURN.npz',allow_pickle=False) as z:
  Pdiag=z['Pdiag'];q=z['q'];cones=[tuple(map(int,v)) for v in z['cones']]
 for step in range(1,3):
  if time.monotonic()>deadline:raise TimeoutError('declared method150sec cap')
  tag=f'NEW-step-{step:02d}';Fvalue,J,E=method_oracle(api,w,record,tag,deadline);oracles_done+=1
  record(tag+'-raw-operator-validation-RETURN',dict(all_finite=finite_or_unknown(w,y,Fvalue,J,E),same_sparse_structure=bool(np.array_equal(J.indptr,E.indptr) and np.array_equal(J.indices,E.indices)),all_SEARCH_inside_tie_envelope=bool(np.all(np.abs(J.data)<=E.data))))
  if not finite_or_unknown(w,y,Fvalue,J,E) or not(np.array_equal(J.indptr,E.indptr) and np.array_equal(J.indices,E.indices)) or not np.all(np.abs(J.data)<=E.data):
   raise ValueError('UNKNOWN_nonfinite_or_SEARCH_tie_envelope')
  dr,dc=positive_sum_outer(E);row_bounds=np.r_[const_rows,dr];col_bounds=const_cols+dc
  sigma,tau=steps(row_bounds,col_bounds,cones)
  record(tag+'-raw-metrics-RETURN',dict(row_bounds=row_bounds,col_bounds=col_bounds,sigma=sigma,tau=tau))
  if not finite_or_unknown(row_bounds,col_bounds,sigma,tau):raise ValueError('UNKNOWN_nonfinite_metric')
  dual_input=y+sigma*Fvalue;record(tag+'-dual-input-RETURN',dict(value=dual_input))
  ynext,branches=product_projection(dual_input,cones);record(tag+'-whole-product-projection-RETURN',dict(value=ynext,ALL_cone_branches=branches))
  gradient=adjoint(api,J,ynext);record(tag+'-ALL-adjoint-RETURN',dict(gradient=gradient))
  raw=w-tau*(gradient+q);record(tag+'-primal-prox-input-RETURN',dict(value=raw))
  wnext,phase=primal_prox(raw,tau,Pdiag);record(tag+'-primal-prox-RETURN',dict(w=wnext,phase=phase))
  if not finite_or_unknown(ynext,gradient,raw,wnext):raise ValueError('UNKNOWN_nonfinite_update')
  w,y=wnext,ynext;steps_done=step;del Fvalue,J,E,dr,dc,row_bounds,col_bounds,sigma,tau,dual_input,gradient,raw
 if time.monotonic()>deadline:raise TimeoutError('declared method150sec cap')
 Fvalue,J,E=method_oracle(api,w,record,'NEW-terminal-step-02',deadline);oracles_done+=1
 record('NEW-terminal-step-02-ONLY-raw-selection-RETURN',dict(w=w,y=y,steps=steps_done,selection='exact terminal2 only; no best/intermediate'))
 motion=api.actual_object_motion(w[:1523]);record('NEW-terminal-ALL6-actual-object-motion-RETURN',motion)
 orthant=np.ones(NROW,dtype=np.bool_);cone_values=[]
 for st,wi in cones:
  orthant[st:st+wi]=False;v=Fvalue[st:st+wi];cone_values.append(float(-v[0]-np.linalg.norm(v[1:])))
 max_affine=float(Fvalue[orthant].max(initial=-np.inf));min_cone=float(min(cone_values));strict=bool(np.all(Fvalue[NOBJ+NROB+NSOCSRC:NOBJ+NROB+NSOCSRC+NDEPTH]<0))
 decision=dict(all_rows_finite=finite_or_unknown(Fvalue,J,E,w,y),maximum_affine_F=max_affine,minimum_negative_F_cone_margin=min_cone,ALL1245_cone_margins=np.array(cone_values),
 exact_zero_phase=bool(w[1522]==0),normalized_phase=float(w[1522]),ALL194_strictdepth_pass=strict,ALL6_actual_motion_pass=all(motion['passes']),
 original_source_NN_rank_roundedmedian_quality_integercells_training_fullpixel_heldout_native_self_swept_motor_grasp_STILL_REQUIRED=True)
 record('NEW-terminal-construction-decision-RETURN',decision)
 passed=decision['all_rows_finite'] and max_affine<=0 and min_cone>=0 and decision['exact_zero_phase'] and strict and all(motion['passes'])
 status='UNQUALIFIED_terminal_model_pass_PENDING_actual_source_and_native_gates' if passed else 'REJECTED_terminal_construction_gate_before_actual_training'
except BaseException as exc:
 record('NEW-actual-boundary-exception-RETURN',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),completed_updates=steps_done,completed_full_oracles=oracles_done,
  unavailable_partial_internal_returns_NOT_reconstructed=True))
 status='UNRESOLVED_declared_method_limit' if isinstance(exc,TimeoutError) else 'UNRESOLVED_actual_method_exception'
 elapsed=time.monotonic()-start
 save(D/'report.json',dict(schema='v2d.NEW-joint-two-update-evaluation.v1',status=status,completed_updates=steps_done,completed_full_oracles=oracles_done,wall_seconds=elapsed,
  terminal_selected=False,actual_training_NN_raster_native_physics_GPU_PPO_queries=0,all_raw_durable_returns_retained=True,no_restart_no_missing_state_reconstruction=True,no_qualification=True))
else:
 save(D/'report.json',dict(schema='v2d.NEW-joint-two-update-evaluation.v1',status=status,completed_updates=steps_done,completed_full_oracles=oracles_done,wall_seconds=time.monotonic()-start,
  terminal_selected=True,terminal_step=2,actual_training_NN_raster_native_physics_GPU_PPO_queries=0,all_raw_durable_returns_retained=True,no_restart_no_missing_state_reconstruction=True,no_qualification=True))
