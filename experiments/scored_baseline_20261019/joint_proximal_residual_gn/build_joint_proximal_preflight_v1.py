"""NEW full-row proximal/GN Schur construction on SAVED F query, no query replay."""
from joint_proximal_gn_v1 import *
from joint_saved_mean_bound_v1 import save
D=S/'e037-joint-proximal-preflight-v1';I=S/'e037-joint-executable-interface-v1';M=S/'e037-coupled-angle-model-v1'
assert json.loads((S/'e037-joint-proximal-remaining-contracts-v1/report.json').read_text())['passed']
proof=C/'joint-mean-proof-checkpoint-v1.json';pc=json.loads(proof.read_text());assert pc['certificate_evaluation']['passed'] and pc['independent_integer_audit']['certificate_full']
pf=S/'e037-joint-method-preflight-v1';paths=[proof,C/'private-integrity-joint-mean-proof-v1.json',I/'NEW-object-retained-A-RETURN.npz',I/'NEW-robot-linear-A-RETURN.npz',I/'NEW-embedded-objective-P-RETURN.npz',I/'NEW-interface-constants-RETURN.npz',dynamic_file(),dynamic_file().with_suffix('.npz'),M/'continuous-angle-variables.json',C/'joint_proximal_gn_v1.py']
save(D/'NEW-input-bindings-BEFORE.json',{str(p):sha(p) for p in paths})
variables=json.loads((M/'continuous-angle-variables.json').read_text());w0=np.zeros(NVAR);w0[:465]=[v['initial_F_angle_rad'] for v in variables]
old_input=typed(next(OLD.glob('*-NEW-step-01-NEW-input-BEFORE.json')))['w']
seal_return(D,'NEW-F-authority-start-identity-RETURN',dict(fresh_F_start=w0,identity_with_saved_query=bool(w0.tobytes()==old_input.tobytes()),F_not_old_failed_vector_selection=True))
assert w0.tobytes()==old_input.tobytes();del old_input
A=sparse.load_npz(I/'NEW-object-retained-A-RETURN.npz');robot=sparse.load_npz(I/'NEW-robot-linear-A-RETURN.npz');P=sparse.load_npz(I/'NEW-embedded-objective-P-RETURN.npz')
with np.load(I/'NEW-interface-constants-RETURN.npz',allow_pickle=False) as z:q=z['q']
old=typed(dynamic_file());J=old['J'];F=typed(next(OLD.glob('*-NEW-step-01-ALL-F-RETURN.json')))['F']
with np.load(pf/'NEW-original-objective-and-cone-layout-RETURN.npz',allow_pickle=False) as z:cones=[tuple(map(int,v)) for v in z['cones']]
seal_return(D,'NEW-proximal-residual-input-BEFORE',dict(F=F,cones=cones,P=P,q=q,current_J_saved_by_hash=True))
residual,branches=product_projection(F,cones);seal_return(D,'NEW-ALL-row-Moreau-residual-RETURN',dict(value=residual,whole_cone_branches=branches))
normal=normal_schur(A,robot,P,J);seal_return(D,'NEW-ALL-row-Schur-normal-RETURN',normal)
rhs=rhs_schur(A,robot,P,q,w0,normal,residual);seal_return(D,'NEW-Schur-RHS-and-original-objective-RETURN',dict(rhs=rhs,P=P,q=q,w0=w0))
assert finite_or_unknown(residual,normal['normal'],normal['base_J'],normal['direction_J'],rhs) and np.all(normal['Schur_weight']>0) and np.all(normal['Schur_weight']<=1)
assert normal['normal'].shape==(1523,1523) and len(F)==4810140 and J.nnz==933560
# NEW complete elimination maps: every direction column belongs to one pair row;
# filtered pair rows stay allocated with identity2 direction blocks.
Dmat=normal['direction_J'].tocoo();owner=Dmat.row-852;expected=Dmat.col//2
seal_return(D,'NEW-ALL-direction-elimination-map-RETURN',dict(row=Dmat.row,col=Dmat.col,value=Dmat.data,owner=owner,expected_pair=expected,all249603pairs=np.arange(249603),all499206directions=np.arange(499206)))
assert np.array_equal(owner,expected)
save(D/'NEW-fixed-method-declaration-RETURN.json',dict(method='ONE full-row proximal-residual Gauss-Newton Schur step',steps=1,raw_trials=1,linefractions=0,restarts=0,warmstarts=0,
 method_seconds=150,outer_seconds=240,RSS_bytes=2*1024**3,disk_reserve_bytes=int(1.5*1024**3),original_Pq_and_phase_interval_unchanged=True,
 search_surrogate='original f linear/quadratic terms + .5||Moreau(F)+J*d||^2 + .5||d||^2; SEARCH only, not physical objective/gate',
 normal='I+P+J.TJ with ALL rows; Schur-eliminate each continuous two-coordinate direction block exactly in ideal algebra',
 every_row_retained_even_zero_residual=True,ALL1245_whole_cones=True,ALL249603_pair_scopes=True,ALLvertices_correlatedties_continuousdirections=True,
 mean_SEARCH_not_universal_directional_derivative_or_stationarity=True,initial_numeric_SEARCH_bound_reused_by_hash_at_exact_saved_F_query_identity=True,
 terminal_NEW_query_no_further_update_or_inherited_numeric_bound=True,original_phase_domain_analytic_quadratic_solution_NOT_terminal_command_repair=True,
 fresh_F_start_no_old_failed_vector_start=True,same_saved_UNQUALIFIED_wrist_middle_hypotheses=True,
 no_old_normal_constructor_method_source_geometry_solver_or_audit_replay=True,every_NEW_return_typed_lossless_BEFORE_checks_queries=True,
 terminal_requires_closed_model_exact_zero_phase_actual6motion_BEFORE_actual_training_NN_raster_and_ALLoriginal_gates=True,no_qualification_by_constructor_or_solver=True))
save(D/'report.json',dict(passed=True,ALL_rows=4810140,base_variables=1523,direction_variables=499206,full_variables=500729,dynamic_entries=933560,
 original_object_nonzeros=A.nnz,robot_nonzeros=robot.nnz,normal_entries=1523**2,whole_cones=1245,pairs=249603,no_NEW_episode_query_or_candidate_selected=True))
