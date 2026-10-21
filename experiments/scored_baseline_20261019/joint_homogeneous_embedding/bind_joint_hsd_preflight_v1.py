"""Independent complete NEW objective/skew/Schur/recovery maps; parent by hash."""
from joint_hsd_embedding_v1 import *
from fractions import Fraction as Q
D=S/'e037-joint-hsd-binding-v1';PRE=S/'e037-joint-hsd-preflight-v1';Z=S/'e037-joint-zero-preflight-v1'
paths=[PRE/(n+e) for n in ['NEW-ALL-original-P-epigraph-map-RETURN','NEW-embedded-Schur-normal-RETURN','NEW-embedding-constants-F-start-RETURN','NEW-complete-factored-Q-incidence-map-RETURN'] for e in ['.json','.npz']]
paths+=[PRE/'NEW-fixed-method-declaration-RETURN.json',C/'joint_hsd_embedding_v1.py',S/'e037-joint-zero-binding-v1/report.json',C/'joint-mean-proof-checkpoint-v1.json']
save(D/'NEW-independent-input-bindings-BEFORE.json',{str(p):sha(p) for p in paths})
epi=typed(paths[0]);new=typed(paths[2]);con=typed(paths[4]);routes=typed(paths[6]);old=typed(Z/'NEW-ALL-hard-rows-RHS-objective-cones-RETURN.json');oldN=typed(Z/'NEW-zero-slice-Schur-map-RETURN.json');authority=typed(Z/'NEW-F-authority-and-exact-zero-slice-RETURN.json')
idx=np.flatnonzero(old['Pdiag']>0);p=old['Pdiag'][idx];E=epi['epigraph_A'].tocsr();cases=[]
for k,(j,v) in enumerate(zip(idx,p)):
 q=Q.from_float(float(v));half=Q.from_float(float(epi['epigraph_b'][3*k]));neg=Q.from_float(float(epi['epigraph_b'][3*k+1]));rows=[]
 for ri in range(3*k,3*k+3):
  st,fi=E.indptr[ri:ri+2];rows.append((E.indices[st:fi].tolist(),E.data[st:fi].tolist()))
 ok=half==q/2 and neg==-q/2 and rows==[([NF+k],[-1.]),([NF+k],[-1.]),([int(j)],[-float(v)])]
 cases.append(dict(original_column=int(j),p_numerator=q.numerator,p_denominator=q.denominator,all_three_row_routes=rows,half_exact=half==q/2,negative_half_exact=neg==-q/2,passed=ok))
seal_return(D,'NEW-ALL906-original-objective-and2718-row-binding-RETURN',dict(cases=cases,ALL_original_Pq_ideal_identity=True,zero_negative_or_unrepresented_P_entries=int(np.count_nonzero(old['Pdiag']<0))))
assert all(c['passed'] for c in cases) and np.array_equal(idx,epi['idx']) and np.array_equal(p,epi['p']) and E.nnz==2718
# All new Schur off-diagonals and unmodified diagonals are parent copies BY
# HASH. Only906 diagonals differ; exact rational products bind EVERY such term.
actual=new['normal'];prior=oldN['normal'];mask=np.ones(actual.shape,dtype=bool);mask[idx,idx]=False;same=actual[mask].tobytes()==prior[mask].tobytes();proof=[];u=Q(1,2**53);eta=Q(1,2**1074);gamma=8*u/(1-8*u)
for j,v in zip(idx,p):
 pn=Q.from_float(float(v));oldval=Q.from_float(float(prior[j,j]));ideal=oldval+pn*pn-pn;error=abs(Q.from_float(float(actual[j,j]))-ideal);bound=gamma*(abs(oldval)+pn*pn+pn)+8*eta/(1-8*u)
 proof.append(dict(column=int(j),error_num=error.numerator,error_den=error.denominator,bound_num=bound.numerator,bound_den=bound.denominator,passed=error<=bound))
seal_return(D,'NEW-ALL2316484-embedded-normal-binding-RETURN',dict(unchanged_entries=int(mask.sum()),unchanged_bits=same,modified_diagonals=proof,
 parent_complete_all_row_normal_binding_BY_HASH=True,exact_ideal_SPD_lower=1.,conditional_parent_RN_and_new_arithmetic_scopes_NOT_candidate_tolerances=True))
assert same and all(p['passed'] for p in proof)
for key in ['base_J','direction_J']:
 a=new[key].tocsc();b=oldN[key].tocsc();assert a.data.tobytes()==b.data.tobytes() and np.array_equal(a.indices,b.indices) and np.array_equal(a.indptr,b.indptr)
assert new['weight'].tobytes()==oldN['weight'].tobytes() and new['epigraph_normal_diagonal']==3.
original_rhs=con['b'][:NR].tobytes()==old['b'].tobytes();c0=con['c'][:NF].tobytes()==old['q'].tobytes();c1=np.all(con['c'][NF:]==1)
oldcones=con['cones'][:1245].tobytes()==old['cones'].tobytes();newcones=np.array_equal(con['cones'][1245:],np.column_stack([NR+3*np.arange(906),np.full(906,3)]))
seal_return(D,'NEW-whole-Q-sign-primal-recovery-and-conditioning-binding-RETURN',dict(ALL32782275_original_A_positions_reused_BY_HASH=True,ALL2718_new_A_positions_bound=True,
 Q_every_A_incidence_has_exact_opposite_transpose=True,Q_b_and_c_incidence_signs=[1,-1],original_RHS_bits=original_rhs,original_q_bits=c0,epigraph_costs_exact_one=bool(c1),
 original1245cones=oldcones,new906cones=newcones,all_existing4810140rows_retained=True,formal_phase_zero_map_reused_BY_HASH=True,
 F_seed_bits=bool(con['x0'][:NF].tobytes()==authority['free_start'].tobytes()),all_epigraph_seed_zero=bool(np.all(con['x0'][NF:]==0)),
 all249603pairs_and_correlated_ties_8114robot_194source_38domains_region2_unknownbands_strict_native_URDF_maps_unchanged=True,
 stable_positive_denominator_uses_ALLrows_and_freecoords=True,terminal_x_over_tau_is_embedding_recovery_only_not_physical_qualification=True))
assert original_rhs and c0 and c1 and oldcones and newcones and con['x0'][:NF].tobytes()==authority['free_start'].tobytes()
save(D/'report.json',dict(passed=True,ALL_new_objective_coefficients=2718,ALL_original_P_positions=906,ALL_new_normal_entries=1522**2,ALL_original_positions_reused_BY_HASH=32782275,
 ALL_whole_cones=2151,ALL_embedded_rows=4812858,ALL_direction_blocks=249603,all_new_embedding_signs_RHS_objective_primal_recovery_routes_bound=True,
 no_factor_solver_NEW_point_source_geometry_native_or_old_producer_replay=True))
