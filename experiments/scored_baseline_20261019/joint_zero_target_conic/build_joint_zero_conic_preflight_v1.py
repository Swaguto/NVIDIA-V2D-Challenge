"""NEW zero-target joint hard affine KKT maps on immutable SAVED inputs."""
from joint_zero_conic_v1 import *
from joint_saved_mean_bound_v1 import save
import shutil
D=S/'e037-joint-zero-preflight-v1';I=S/'e037-joint-executable-interface-v1';G=S/'e037-joint-proximal-preflight-v1'
M=S/'e037-coupled-angle-model-v1';PF=S/'e037-joint-method-preflight-v1'
paths=[S/'e037-joint-zero-contracts-v1/report.json',C/'joint_zero_conic_v1.py',C/'joint-mean-proof-checkpoint-v1.json',C/'private-integrity-joint-mean-proof-v1.json',
 I/'NEW-object-retained-A-RETURN.npz',I/'NEW-robot-linear-A-RETURN.npz',I/'NEW-embedded-objective-P-RETURN.npz',I/'NEW-interface-constants-RETURN.npz',
 dynamic_file(),dynamic_file().with_suffix('.npz'),next(OLD.glob('*-NEW-step-01-ALL-F-RETURN.json')),next(OLD.glob('*-NEW-step-01-ALL-F-RETURN.npz')),
 G/'NEW-ALL-row-Schur-normal-RETURN.json',G/'NEW-ALL-row-Schur-normal-RETURN.npz',PF/'NEW-original-objective-and-cone-layout-RETURN.npz',M/'continuous-angle-variables.json']
save(D/'NEW-input-bindings-BEFORE.json',{str(p):sha(p) for p in paths})
assert json.loads(paths[0].read_text())['passed']
proof=json.loads((C/'joint-mean-proof-checkpoint-v1.json').read_text());assert proof['certificate_evaluation']['passed'] and proof['independent_integer_audit']['certificate_full']
variables=json.loads((M/'continuous-angle-variables.json').read_text());w0=np.zeros(NVAR);w0[:465]=[v['initial_F_angle_rad'] for v in variables]
old=typed(next(OLD.glob('*-NEW-step-01-NEW-input-BEFORE.json')))['w']
seal_return(D,'NEW-F-authority-and-exact-zero-slice-RETURN',dict(full_start=w0,free_start=select(w0),KEEP=KEEP,BASEKEEP=BASEKEEP,phase_index=PHASE,
 identity_with_saved_query=bool(w0.tobytes()==old.tobytes()),phase_is_formal_original_coordinate=True,zero_target_additional_equality=True,not_old_failed_vector_selection=True))
assert w0.tobytes()==old.tobytes() and w0[PHASE]==0.;del old
A=sparse.load_npz(I/'NEW-object-retained-A-RETURN.npz');R=sparse.load_npz(I/'NEW-robot-linear-A-RETURN.npz');P=sparse.load_npz(I/'NEW-embedded-objective-P-RETURN.npz')
with np.load(I/'NEW-interface-constants-RETURN.npz',allow_pickle=False) as z:ob=z['object_b'];rb=z['robot_b'];q=z['q']
raw=typed(dynamic_file());J=raw['J'];F=typed(next(OLD.glob('*-NEW-step-01-ALL-F-RETURN.json')))['F'];del raw
normal=typed(G/'NEW-ALL-row-Schur-normal-RETURN.json')
B=normal['base_J'][:,BASEKEEP].tocsc();Ds=normal['direction_J'];weight=normal['Schur_weight'];S0=normal['normal'][BASEKEEP][:,BASEKEEP]
seal_return(D,'NEW-zero-slice-Schur-map-RETURN',dict(normal=S0,base_J=B,direction_J=Ds,weight=weight,base_full_indices=np.flatnonzero(BASEKEEP),
 direction_full_indices=np.arange(1523,NVAR),all_pair_instances=np.arange(249603),normal_reused_candidate_BY_HASH_NOT_feasibility_certificate=True))
jw=J@w0;seal_return(D,'NEW-dynamic-affine-origin-action-RETURN',dict(value=jw))
db=jw-F[NOBJ+NROB:];seal_return(D,'NEW-dynamic-affine-RHS-RETURN',dict(value=db,definition='J_saved*w_F - F_saved_dynamic'))
b=np.r_[ob,rb,db];Pdiag=np.zeros(NFREE);Pdiag[:1522]=P.diagonal()[BASEKEEP];qfree=np.zeros(NFREE);qfree[:1522]=q[BASEKEEP]
with np.load(PF/'NEW-original-objective-and-cone-layout-RETURN.npz',allow_pickle=False) as z:cones=z['cones'].copy()
seal_return(D,'NEW-ALL-hard-rows-RHS-objective-cones-RETURN',dict(b=b,Pdiag=Pdiag,q=qfree,cones=cones,formal_phase_equality_RHS=0.,
 existing_numerical_rows=NROW,formal_rows_with_zero_target=NROW+1,original_phase_domain_rows_retained=True,original_Pq_conditioning_unchanged=True))
# Every original coefficient position is retained as either a free-column route
# or an exact algebraic zero term in the original required phase==0 slice.
oc=np.arange(465,1523);jc=np.arange(NVAR);ro=np.arange(465)
of=np.where(oc==PHASE,-1,oc-(oc>PHASE));jf=np.where(jc==PHASE,-1,jc-(jc>PHASE))
seal_return(D,'NEW-complete-zero-target-column-routes-RETURN',dict(object_full=oc,object_free=of,robot_full=ro,robot_free=ro,dynamic_full=jc,dynamic_free=jf,
 all_original_stored_positions=dict(object=A.nnz,robot=R.nnz,dynamic=J.nnz),phase_terms_are_exact_zero_NOT_unknown_or_deleted_constraints=True))
seal_return(D,'NEW-preflight-finiteness-and-resource-RETURN',dict(all_finite=finite_or_unknown(S0,B,Ds,weight,jw,db,b,Pdiag,qfree),free_disk_bytes=shutil.disk_usage(S).free,
 normal_bytes=S0.nbytes,full_row_bytes=NROW*8,free_vector_bytes=NFREE*8,updates=16,estimated_return_bytes=16*(8*NROW*8+12*NFREE*8+12*250455*8)+2*1024**3,
 no_dense_full_Jacobian=True,original_source_native_operators_separate=True))
assert finite_or_unknown(S0,B,Ds,weight,jw,db,b,Pdiag,qfree) and len(b)==NROW and J.nnz==933560 and len(cones)==1245
save(D/'NEW-fixed-method-declaration-RETURN.json',dict(method='NEW joint hard zero-target affine KKT Douglas-Rachford resolvent',updates=16,terminal_update_ONLY=16,
 method_seconds=150,outer_seconds=240,RSS_bytes=2*1024**3,disk_reserve_bytes=int(1.5*1024**3),unit_resolvent_metric=True,relaxation=1,
 full_variables=NVAR,free_variables=NFREE,original_phase_coordinate=PHASE,exact_affine_phase_zero_parameterization=True,phase_domain_and_all_existing_rows_unchanged=True,
 original_objective_Pq_unchanged=True,original_constraints_plus_already_required_zero_target=True,
 operator='M(x,y)=(P*x+A.T*y+q, -A*x+b), C=R^n times whole K; 0 in M+N_C',
 update='u=(I+M)^-1 z; v=projection_C(2u-z); z_next=z+v-u',
 structured_linear_system='I+P+A.T*A; exact algebraic elimination of ALL two-coordinate continuous pair blocks, no selected rows',
 normal_candidate_reused_BY_HASH_requires_NEW_zero_slice_binding=True,all_4810140_rows_1245_cones_249603_pairs=True,
 original_8114_robot_194_source_38_domains_region2_three_unknown_bands=True,ALL_compiled_vertices_correlated_ties_continuous_directions=True,
 fresh_F_start_dual_zero_NO_old_failed_warmstart=True,same_saved_UNQUALIFIED_wrist_middle_hypotheses=True,
 saved_F_query_and_mean_product_certificate_reused_BY_HASH_no_query_replay=True,mean_SEARCH_not_universal_directional_derivative=True,
 affine_hard_model_only_NO_native_or_source_identity=True,all_intermediate_returns_lossless_before_further_work=True,
 no_retry_restart_linefraction_best_intermediate_tuning_or_budget_extension=True,
 terminal_shadow_primal_only_requires_ALL_closed_model_rows_cones_exact_phase_zero_before_NEW_nonlinear_query=True,
 no_training_NN_raster_source_native_physics_GPU_unless_all_original_preceding_gates_pass=True,no_constructor_solver_or_checkpoint_qualification=True))
save(D/'report.json',dict(passed=True,original_rows=NROW,formal_rows=NROW+1,original_variables=NVAR,free_variables=NFREE,normal_entries=1522**2,
 original_object_coefficients=A.nnz,robot_terms=R.nnz,dynamic_coefficients=J.nnz,whole_cones=1245,pairs=249603,no_NEW_point_or_solver_or_episode_query=True))
