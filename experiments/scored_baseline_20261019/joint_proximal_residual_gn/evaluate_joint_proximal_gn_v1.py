"""SOLE one-step full-row proximal/GN construction and ONE raw NEW joint trial."""
from joint_proximal_gn_v1 import *
from joint_saved_mean_bound_v1 import save
import traceback
D=S/'e037-joint-proximal-evaluation-v1';PRE=S/'e037-joint-proximal-preflight-v1';record=MethodRecorder(D);start=time.monotonic();deadline=start+150;point_sealed=False;full_query=False
paths=[PRE/'NEW-fixed-method-declaration-RETURN.json',PRE/'NEW-ALL-row-Schur-normal-RETURN.json',PRE/'NEW-ALL-row-Schur-normal-RETURN.npz',PRE/'NEW-Schur-RHS-and-original-objective-RETURN.json',PRE/'NEW-Schur-RHS-and-original-objective-RETURN.npz',S/'e037-joint-proximal-binding-v1/report.json',C/'joint_proximal_gn_v1.py',C/'joint-mean-proof-checkpoint-v1.json']
save(D/'NEW-fixed-evaluation-declaration-BEFORE.json',dict(input_sha256={str(p):sha(p) for p in paths},method_seconds=150,outer_seconds=240,RSS_bytes=2*1024**3,steps=1,raw_trials=1,no_linefraction_restart_warmstart_bestintermediate=True,
 original_phase_interval_and_Pq_unchanged=True,new_trial_must_pass_ALLoriginal_source_motion_native_self_swept_motor_grasp_before_training=True))
assert json.loads((S/'e037-joint-proximal-binding-v1/report.json').read_text())['passed']
try:
 normal=typed(PRE/'NEW-ALL-row-Schur-normal-RETURN.json');rhsdata=typed(PRE/'NEW-Schur-RHS-and-original-objective-RETURN.json');residual=typed(PRE/'NEW-ALL-row-Moreau-residual-RETURN.json')['value'];w0=rhsdata['w0']
 record('NEW-method-input-BEFORE',dict(w0=w0,normal_inputs_bound_to_NEW_proximal_method=True,saved_base_query_and_RN_certificate_reused_BY_HASH_only=True))
 dz,both,phase=phase_constrained_quadratic(normal['normal'],rhsdata['rhs'],float(w0[1522]),record)
 guard=solve_RN_guard(normal['normal'],both,rhsdata['rhs']);record('NEW-actual-two-RHS-RN-guard-RETURN',guard)
 if not guard['all_within'] or not finite_or_unknown(dz,both):raise ValueError('UNRESOLVED_normal_search_return_guard')
 step=backsubstitute(normal,dz,residual);record('NEW-ALL500729-backsub-step-RETURN',dict(step=step,base_step=dz,continuous_direction_step=step[1523:]))
 trial=w0+step;record('NEW-ONE-raw-joint-trial-BEFORE-query',dict(w=trial,w_bytes_sha256=hashlib.sha256(trial.tobytes()).hexdigest(),phase=phase,only_trial=True,UNQUALIFIED=True));point_sealed=True
 if not finite_or_unknown(trial):raise ValueError('UNKNOWN_nonfinite_raw_trial')
 if time.monotonic()>deadline:raise TimeoutError('declared method150sec cap')
 api=JointExecutableInterface(S/'e037-joint-executable-interface-v1');record('NEW-method-runtime-constructor-RETURN',api.__dict__)
 F,J,E=method_oracle(api,trial,record,'NEW-ONE-trial',deadline);full_query=True
 # Terminal only: no update uses the NEW terminal J/E or inherits the old bound.
 motion=api.actual_object_motion(trial[:1523]);record('NEW-ONE-trial-ALL6-actual-object-motion-RETURN',motion)
 with np.load(S/'e037-joint-method-preflight-v1/NEW-original-objective-and-cone-layout-RETURN.npz',allow_pickle=False) as z:cones=[tuple(map(int,v)) for v in z['cones']]
 orthant=np.ones(NROW,dtype=np.bool_);cone_values=[]
 for st,wi in cones:
  orthant[st:st+wi]=False;v=F[st:st+wi];cone_values.append(float(-v[0]-np.linalg.norm(v[1:])))
 finite=finite_or_unknown(F,J,E,trial);max_aff=float(F[orthant].max(initial=-np.inf));min_cone=float(min(cone_values));strict=bool(np.all(F[NOBJ+NROB+582:NOBJ+NROB+776]<0))
 decision=dict(all_finite=finite,maximum_affine_F=max_aff,minimum_negative_F_cone_margin=min_cone,ALL1245_cone_margins=np.asarray(cone_values),ALL194_strict_depth=strict,
 exact_zero_phase=bool(trial[1522]==0),actual_normalized_phase=float(trial[1522]),ALL6_actual_motion=all(motion['passes']),motion_passes=motion['passes'],
 violated_affine_rows=int(np.count_nonzero(F[orthant]>0)),violated_cones=int(np.count_nonzero(np.asarray(cone_values)<0)),original_objective=float(.5*trial[:1523]@(rhsdata['P']@trial[:1523])+rhsdata['q']@trial[:1523]),
 terminal_numeric_J_E_NOT_used_for_another_update_or_claimed_outward=True,original_actual_training_integercells_NN_rank_median_quality_fulltriangles_and_fullpixel_heldout_POST_STILL_REQUIRED=True,
 independent_whole151source_URDF_robot_root_finger_motion_native_hand_object_support_ANDself_swept_motor_grasp_STILL_REQUIRED=True)
 record('NEW-ONE-trial-ALL-construction-decision-RETURN',decision)
 passed=finite and max_aff<=0 and min_cone>=0 and strict and decision['exact_zero_phase'] and all(motion['passes'])
 status='UNQUALIFIED_model_pass_PENDING_all_actual_source_native_gates' if passed else 'REJECTED_raw_joint_trial_before_actual_training_source'
except BaseException as e:
 record('NEW-actual-method-boundary-RETURN',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),raw_trial_sealed=point_sealed,complete_NEW_trial_query=full_query,missing_returns_NOT_reconstructed=True))
 save(D/'report.json',dict(status='UNRESOLVED_declared_method_limit' if isinstance(e,TimeoutError) else 'UNRESOLVED_actual_method_exception',wall_seconds=time.monotonic()-start,raw_trial_sealed=point_sealed,complete_NEW_trial_query=full_query,
 no_retry_no_raw_repair=True,no_actual_training_NN_raster_native_physics_GPU_PPO_query=True,no_qualification=True))
else:
 save(D/'report.json',dict(status=status,wall_seconds=time.monotonic()-start,steps=1,raw_trials=1,raw_trial_sealed=True,complete_NEW_trial_query=True,terminal_only=True,
 no_old_method_initial_query_or_solver_replay=True,no_actual_training_NN_raster_integercell_fullpixel_heldout_native_self_swept_motor_grasp_physics_GPU_PPO_query=True,
 original_F_components_gates_8self5object_UNCHANGED=True,no_reference_trajectory_task_policy_or_AUC_qualification=True))
