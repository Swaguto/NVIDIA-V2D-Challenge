"""SOLE bounded NEW hard joint zero-target KKT construction, terminal16 only."""
from joint_zero_conic_v1 import *
from joint_zero_return_guard_v1 import linear_return_guard
from joint_saved_mean_bound_v1 import save
import shutil,traceback
D=S/'e037-joint-zero-evaluation-v1';PRE=S/'e037-joint-zero-preflight-v1';I=S/'e037-joint-executable-interface-v1'
record=MethodRecorder(D);start=time.monotonic();deadline=start+150;completed=0;point_sealed=False;full_query=False
paths=[PRE/'NEW-fixed-method-declaration-RETURN.json',PRE/'NEW-zero-slice-Schur-map-RETURN.json',PRE/'NEW-zero-slice-Schur-map-RETURN.npz',
 PRE/'NEW-ALL-hard-rows-RHS-objective-cones-RETURN.json',PRE/'NEW-ALL-hard-rows-RHS-objective-cones-RETURN.npz',PRE/'NEW-F-authority-and-exact-zero-slice-RETURN.json',PRE/'NEW-F-authority-and-exact-zero-slice-RETURN.npz',
 S/'e037-joint-zero-binding-v1/report.json',S/'e037-joint-zero-return-contract-v1/report.json',C/'joint_zero_conic_v1.py',C/'joint_zero_return_guard_v1.py',C/'joint-mean-proof-checkpoint-v1.json',
 I/'NEW-object-retained-A-RETURN.npz',I/'NEW-robot-linear-A-RETURN.npz',dynamic_file(),dynamic_file().with_suffix('.npz')]
save(D/'NEW-fixed-evaluation-declaration-BEFORE.json',dict(input_sha256={str(p):sha(p) for p in paths},method='NEW joint hard zero-target affine KKT Douglas-Rachford',
 updates=16,terminal_update_ONLY=16,method_seconds=150,outer_seconds=240,RSS_bytes=2*1024**3,disk_reserve_bytes=int(1.5*1024**3),
 original_phase_coordinate_domain_Pq_and_all_rows_unchanged=True,exact_zero_target_affine_slice=True,no_retry_restart_warmstart_linefraction_tuning_or_bestintermediate=True,
 every_raw_intermediate_typed_lossless_BEFORE_checks_and_further_work=True,old_saved_query_RN_bound_identity_BY_HASH_only=True,
 hard_affine_gate_BEFORE_any_NEW_nonlinear_or_actual_source_or_native_query=True,no_qualification_by_solver_or_model=True))
assert json.loads((S/'e037-joint-zero-binding-v1/report.json').read_text())['passed'] and json.loads((S/'e037-joint-zero-return-contract-v1/report.json').read_text())['passed']
try:
 maps=typed(PRE/'NEW-zero-slice-Schur-map-RETURN.json');con=typed(PRE/'NEW-ALL-hard-rows-RHS-objective-cones-RETURN.json');authority=typed(PRE/'NEW-F-authority-and-exact-zero-slice-RETURN.json')
 A=sparse.load_npz(I/'NEW-object-retained-A-RETURN.npz');R=sparse.load_npz(I/'NEW-robot-linear-A-RETURN.npz');raw=typed(dynamic_file());J=raw['J'];del raw
 B=maps['base_J'];Ds=maps['direction_J'];weight=maps['weight'];normal=maps['normal'];b=con['b'];q=con['q'];Pdiag=con['Pdiag'];cones=[tuple(map(int,a)) for a in con['cones']]
 zx=authority['free_start'].copy();zy=np.zeros(NROW)
 record('NEW-method-input-BEFORE',dict(zx=zx,zy=zy,normal=normal,original_q=q,original_Pdiag=Pdiag,original_cones=cones,
  full_start_from_F=authority['full_start'],all_row_RHS_immutable_return_sha256=sha(PRE/'NEW-ALL-hard-rows-RHS-objective-cones-RETURN.npz'),
  all_operator_inputs_immutable_bound_BY_HASH=True,formal_original_phase_index=PHASE,required_zero_target_literal=0.,saved_F_tie_RN_certificate_scope_ONLY=True))
 needed=16*(8*NROW*8+12*NFREE*8+12*250455*8)+2*1024**3;free=shutil.disk_usage(S).free
 record('NEW-before-factor-resource-RETURN',dict(free_disk_bytes=free,declared_return_estimate=needed,reserve_bytes=int(1.5*1024**3),all_one_heavy_process=True))
 if free-needed<int(1.5*1024**3):raise OSError('declared durable-return disk reserve prerequisite')
 factor=linalg.cho_factor(normal,lower=True,check_finite=True);record('NEW-zero-slice-normal-factor-RETURN',dict(factor=factor[0],lower=factor[1]))
 if not finite_or_unknown(factor[0]):raise ValueError('UNKNOWN_nonfinite_factor')
 for step in range(1,17):
  if time.monotonic()>deadline:raise TimeoutError('declared method150sec cap')
  tag=f'NEW-update-{step:02d}';record(tag+'-input-BEFORE',dict(zx=zx,zy=zy))
  t=zy-b;record(tag+'-dual-minus-RHS-RETURN',dict(value=t))
  gradient=full_adjoint(A,R,J,t,record,tag)
  rhs=zx-q-gradient;record(tag+'-full-resolvent-RHS-RETURN',dict(value=rhs))
  ux=resolvent_solve(B,Ds,weight,factor,rhs,record,tag)
  # The actual returned linear solve is screened before any split update. The
  # initial tied J is unchanged, under its completed SAVED-binary proof scope.
  schur_rhs=typed(next(D.glob(f'*-{tag}-Schur-RHS-RETURN.json')))['value']
  guard=linear_return_guard(normal,ux[:1522],schur_rhs,record,tag)
  if not finite_or_unknown(ux,guard['residual'],guard['conditional_RN_outer']) or not guard['all_within']:raise ValueError('UNRESOLVED_actual_linear_SEARCH_return_guard')
  ax=full_action(A,R,J,ux,record,tag)
  uy=t+ax;record(tag+'-dual-resolvent-RETURN',dict(value=uy))
  reflected=2*uy-zy;record(tag+'-whole-K-reflection-BEFORE',dict(value=reflected))
  projected,branches=product_projection(reflected,cones);record(tag+'-whole-K-projection-RETURN',dict(value=projected,branches=branches))
  nx=ux.copy();ny=zy+projected-uy;record(tag+'-split-update-RETURN',dict(zx=nx,zy=ny,shadow_primal=ux,completed_update=step))
  if not finite_or_unknown(nx,ny,uy,projected,ax):raise ValueError('UNKNOWN_nonfinite_split_return')
  zx,zy=nx,ny;completed=step
  if time.monotonic()>deadline:raise TimeoutError('declared method150sec cap')
 # This terminal shadow is unqualified; all affine constraints and the formal
 # phase equality are screened before a new nonlinear episode point is used.
 terminal=lift(ux);Faff=ax-b;record('NEW-terminal16-hard-affine-residual-RETURN',dict(F=Faff,unqualified_full_shadow=terminal,original_phase_value=terminal[PHASE],terminal_only=True))
 orthant=np.ones(NROW,dtype=np.bool_);margins=[]
 for st,wi in cones:
  orthant[st:st+wi]=False;v=Faff[st:st+wi];margins.append(float(-v[0]-np.linalg.norm(v[1:])))
 margins=np.asarray(margins);maximum=float(Faff[orthant].max(initial=-np.inf));minimum=float(margins.min());strict=bool(np.all(Faff[NOBJ+NROB+582:NOBJ+NROB+776]<0))
 objective=float(.5*np.dot(Pdiag*ux,ux)+np.dot(q,ux))
 decision=dict(all_finite=finite_or_unknown(terminal,Faff,margins),ALL_existing_rows=NROW,formal_equality_rows=NROW+1,original_phase_value=float(terminal[PHASE]),exact_zero_phase=bool(terminal[PHASE]==0.),
  maximum_affine_F=maximum,minimum_negative_F_cone_margin=minimum,all1245_cone_margins=margins,ALL194_strict_depth=strict,
  violated_affine_rows=int(np.count_nonzero(Faff[orthant]>0)),violated_cones=int(np.count_nonzero(margins<0)),original_objective=objective,
  mixed_model_units_NOT_native_metres_contact_force_or_score=True,all_original_source_motion_native_self_swept_motor_grasp_gates_STILL_REQUIRED=True)
 record('NEW-terminal16-hard-affine-decision-RETURN',decision)
 passed=decision['all_finite'] and maximum<=0 and minimum>=0 and strict and decision['exact_zero_phase']
 if not passed:
  status='UNRESOLVED_fixed16_hard_affine_zero_target_rejected_before_NEW_nonlinear_query'
 else:
  record('NEW-ONE-terminal16-raw-joint-point-BEFORE-query',dict(w=terminal,UNQUALIFIED=True,terminal_only=True));point_sealed=True
  if time.monotonic()>deadline:raise TimeoutError('declared method150sec cap')
  api=JointExecutableInterface(I);record('NEW-actual-runtime-constructor-RETURN',api.__dict__)
  Fnew,Jnew,Enew=method_oracle(api,terminal,record,'NEW-ONE-terminal16',deadline);full_query=True
  motion=api.actual_object_motion(terminal[:1523]);record('NEW-ONE-terminal16-ALL6-actual-object-motion-RETURN',motion)
  record('NEW-terminal-nonlinear-raw-return-boundary-RETURN',dict(F=Fnew,J=Jnew,E=Enew,terminal_no_further_SEARCH_update=True,
   old_SAVED_numeric_bound_NOT_inherited=True,actual_training_fullsource_native_self_swept_motor_grasp_STILL_REQUIRED=True))
  status='UNQUALIFIED_terminal_nonlinear_return_PENDING_independent_hard_gates'
except BaseException as e:
 record('NEW-actual-method-boundary-RETURN',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),completed_updates=completed,
  NEW_nonlinear_point_sealed=point_sealed,complete_NEW_nonlinear_query=full_query,missing_returns_NOT_reconstructed=True))
 save(D/'report.json',dict(status='UNRESOLVED_declared_method_limit' if isinstance(e,TimeoutError) else 'UNRESOLVED_actual_method_exception',wall_seconds=time.monotonic()-start,
  completed_updates=completed,NEW_nonlinear_point_sealed=point_sealed,complete_NEW_nonlinear_query=full_query,no_retry_no_raw_repair=True,no_actual_training_native_physics_GPU_query=True,no_qualification=True))
else:
 save(D/'report.json',dict(status=status,wall_seconds=time.monotonic()-start,completed_updates=completed,terminal_update_ONLY=16,NEW_nonlinear_point_sealed=point_sealed,
  complete_NEW_nonlinear_query=full_query,all_initial_query_interfaces_matrices_maps_numeric_certificate_BY_HASH_no_replay=True,
  no_original_source_actual_training_NN_raster_integercell_fullpixel_heldout_native_self_swept_motor_grasp_physics_GPU_PPO_query=True,
  original_F_components_parameters_gates_8self5object_UNCHANGED=True,no_reference_trajectory_task_policy_or_AUC_qualification=True))
