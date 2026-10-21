"""NEW square-root-free epigraph and homogeneous embedding maps, no query."""
from joint_hsd_embedding_v1 import *
D=S/'e037-joint-hsd-preflight-v1';Z=S/'e037-joint-zero-preflight-v1';I=S/'e037-joint-executable-interface-v1'
paths=[S/'e037-joint-hsd-contracts-v1/report.json',Z/'NEW-zero-slice-Schur-map-RETURN.json',Z/'NEW-zero-slice-Schur-map-RETURN.npz',
 Z/'NEW-ALL-hard-rows-RHS-objective-cones-RETURN.json',Z/'NEW-ALL-hard-rows-RHS-objective-cones-RETURN.npz',Z/'NEW-F-authority-and-exact-zero-slice-RETURN.json',
 Z/'NEW-F-authority-and-exact-zero-slice-RETURN.npz',S/'e037-joint-zero-binding-v1/report.json',C/'joint-mean-proof-checkpoint-v1.json',C/'joint_hsd_embedding_v1.py']
save(D/'NEW-input-bindings-BEFORE.json',{str(p):sha(p) for p in paths});assert json.loads(paths[0].read_text())['passed']
old=typed(paths[3]);maps=typed(paths[1]);authority=typed(paths[5]);idx=np.flatnonzero(old['Pdiag']);p=old['Pdiag'][idx];ne=len(idx);nx=NF+ne;nr=NR+3*ne
eb=np.zeros((ne,3));eb[:,0]=p/2;eb[:,1]=-p/2
ri=np.arange(3*ne);ci=np.empty(3*ne,dtype=np.int64);ci[0::3]=NF+np.arange(ne);ci[1::3]=NF+np.arange(ne);ci[2::3]=idx
data=np.empty(3*ne);data[0::3]=-1;data[1::3]=-1;data[2::3]=-p
E=sparse.coo_matrix((data,(ri,ci)),shape=(3*ne,nx)).tocsc()
seal_return(D,'NEW-ALL-original-P-epigraph-map-RETURN',dict(idx=idx,p=p,epigraph_A=E,epigraph_b=eb.ravel(),new_rows=NR+ri,all906_epigraph_variables=NF+np.arange(ne),
 ideal_identity='(t+p/2)^2-(t-p/2)^2-(p*x)^2 = 2*p*t-p^2*x^2; p>0 iff t>=.5*p*x^2',no_sqrt_no_objective_change=True))
ps=p*p;seal_return(D,'NEW-epigraph-normal-square-RETURN',dict(value=ps))
modifier=ps-p;seal_return(D,'NEW-epigraph-normal-modifier-RETURN',dict(value=modifier))
normal=maps['normal'].copy();normal[idx,idx]+=modifier
seal_return(D,'NEW-embedded-Schur-normal-RETURN',dict(normal=normal,base_J=maps['base_J'],direction_J=maps['direction_J'],weight=maps['weight'],
 epigraph_normal_diagonal=3.,normal_definition='I + ALL original L.TL + epigraph A.TA; original P moved exactly to conic epigraph, not removed from objective'))
b=np.r_[old['b'],eb.ravel()];c=np.r_[old['q'],np.ones(ne)];cones=np.vstack([old['cones'],np.column_stack([NR+3*np.arange(ne),np.full(ne,3)])])
x0=np.r_[authority['free_start'],np.zeros(ne)]
seal_return(D,'NEW-embedding-constants-F-start-RETURN',dict(b=b,c=c,cones=cones,x0=x0,idx=idx,p=p,primal_variables=nx,cone_rows=nr,formal_rows_including_phase_equality=nr+1,
 fresh_F_start_not_old_failed_vector=True,original_formal_phase_coordinate=PHASE,phase_exact_zero_parameterization_unchanged=True))
seal_return(D,'NEW-complete-factored-Q-incidence-map-RETURN',dict(original_A_positions=32782275,new_epigraph_positions=3*ne,
 Q_blocks=[('primal','dual','A.T'),('dual','primal','-A'),('primal','tau','c'),('tau','primal','-c.T'),('dual','tau','b'),('tau','dual','-b.T')],
 original_rows_and_all_transpose_positions_reused_BY_HASH=True,all_original_P_positions=ne,original_pair_scopes=249603,direction_coords=499206,
 explicit_homogeneous_tau_and_kappa_auxiliaries_NOT_physical_phase=True,ALL_original_source_native_strict_flags_conditioning_unchanged=True))
assert ne==906 and nr==4812858 and nx==501634 and len(cones)==2151 and finite_or_unknown(normal,E,b,c,x0)
save(D/'NEW-fixed-method-declaration-RETURN.json',dict(method='NEW square-root-free original-objective homogeneous self-dual joint conic embedding',updates=8,terminal_update_ONLY=8,
 method_seconds=150,outer_seconds=240,RSS_bytes=2*1024**3,disk_reserve_bytes=int(1.5*1024**3),
 numerical_original_rows=NR,new_objective_rows=2718,ALL_whole_cones=2151,original_Pq_exact_ideal_epigraph_equivalence=True,all_original_phase_domain_rows_and_zero_target_slice=True,
 embedding='Q=[[0,A.T,c],[-A,0,b],[-c.T,-b.T,0]], v=Q*u, u in R^n x K x R+, v in {0} x K x R+',
 resolvent='one structured factor for H=I+A.TA; stable positive scalar denominator from whole-row squared residuals',
 stable_denominator='1+||wc||^2+||A*wc||^2+||wb||^2+||b-A*wb||^2; wb=H^-1*A.T*b, wc=H^-1*c',
 splitting='tilde=(I+Q)^-1(u+v); u_next=projection_C(tilde-v); v_next=v-tilde+u_next',
 start='fresh F-derived x, y=0,tau=1; v_x=0,v_y=0,kappa=1; no old iterate/dual/intermediate',
 terminal_recovery='terminal8 ONLY; if tau>0 finite, raw x/tau then exact formal phase-zero map; not old-return normalization/repair',
 ALL_rows_compiled_vertices_correlated_ties_continuous_directions_native_filters_retained=True,initial_saved_F_query_numeric_bound_BY_HASH_only=True,
 no_model_query_interface_constructor_or_old_solver_method_replay=True,no_dense_full_Q_or_Jacobian=True,
 no_rescale_clip_project_repair_of_raw_candidate=True,projection_is_algorithm_operator_only=True,no_retry_restart_warmstart_linefraction_best_or_parameter_sweep=True,
 every_NEW_return_typed_lossless_before_checks_further_queries=True,terminal_must_pass_ALLhard_affine_zero_phase_before_NEW_nonlinear_and_ALLoriginal_gates=True,
 tau_zero_or_finite_failure_NOT_source_family_or_task_exclusion=True,no_qualification_by_embedding_or_solver=True))
save(D/'report.json',dict(passed=True,NEW_epigraph_coefficients=E.nnz,original_P_positions=ne,primal_variables=nx,original_formal_variables=NVAR,original_rows=NR,
 embedded_rows=nr,formal_rows=nr+1,all_whole_cones=2151,normal_entries=1522**2,no_NEW_point_query_factor_solver_or_method_yet=True))
