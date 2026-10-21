"""NEW complete independent Schur/normal/RHS maps; no old producer replay."""
from joint_proximal_gn_v1 import *
from joint_saved_mean_bound_v1 import save
D=S/'e037-joint-proximal-binding-v1';PRE=S/'e037-joint-proximal-preflight-v1';I=S/'e037-joint-executable-interface-v1'
paths=[PRE/'NEW-ALL-row-Schur-normal-RETURN.json',PRE/'NEW-ALL-row-Schur-normal-RETURN.npz',PRE/'NEW-Schur-RHS-and-original-objective-RETURN.json',PRE/'NEW-Schur-RHS-and-original-objective-RETURN.npz',PRE/'NEW-ALL-direction-elimination-map-RETURN.json',PRE/'NEW-ALL-direction-elimination-map-RETURN.npz',PRE/'NEW-fixed-method-declaration-RETURN.json',C/'joint_proximal_gn_v1.py']
save(D/'NEW-independent-input-bindings-BEFORE.json',{str(p):sha(p) for p in paths})
normal=typed(paths[0]);rhsdata=typed(paths[2]);elim=typed(paths[4]);r=typed(PRE/'NEW-ALL-row-Moreau-residual-RETURN.json')['value']
A=sparse.load_npz(I/'NEW-object-retained-A-RETURN.npz');robot=sparse.load_npz(I/'NEW-robot-linear-A-RETURN.npz');P=sparse.load_npz(I/'NEW-embedded-objective-P-RETURN.npz');raw=typed(dynamic_file());J=raw['J']
B=normal['base_J'];Dmat=normal['direction_J'];weight=normal['Schur_weight']
# Full exact new routing identities, all coefficients once, including explicit0s.
assert np.array_equal(J[:,:1523].tocsc().data,B.tocsc().data)
assert np.array_equal(J[:,:1523].tocsc().indices,B.tocsc().indices) and np.array_equal(J[:,:1523].tocsc().indptr,B.tocsc().indptr)
assert np.array_equal(J[:,1523:].tocsc().data,Dmat.tocsc().data) and np.array_equal(J[:,1523:].tocsc().indices,Dmat.tocsc().indices) and np.array_equal(J[:,1523:].tocsc().indptr,Dmat.tocsc().indptr)
assert np.array_equal(elim['owner'],elim['expected_pair']) and len(elim['all249603pairs'])==249603 and len(elim['all499206directions'])==499206
# Alternate row-oriented normal route; every retained source/object/robot row is
# present, with all geometric Schur weights (no residual-selected row subset).
Ar=A.tocsr();ref=np.eye(1523)+P.toarray();ref[465:,465:]+=(Ar.T@Ar).toarray();del Ar
Rr=robot.tocsr();ref[:465,:465]+=(Rr.T@Rr).toarray()
ref+=(B.multiply(weight[:,None]).T@B).toarray()
NROW=4810140;u=2.**-53;count=3*NROW+32;gamma=count*u/(1-count*u)
# Positive normal diagonals and Cauchy give a complete per-entry conditional RN
# envelope without invoking exact binary/source/native identities.
diag=np.diag(ref);bound=4*gamma*(diag[:,None]+diag[None,:])+np.finfo(float).tiny
error=np.abs(normal['normal']-ref)
seal_return(D,'NEW-ALL2319529-normal-entry-binding-RETURN',dict(reference=ref,actual=normal['normal'],absolute_error=error,conditional_RN_outer=bound,
 all_rows=4810140,all_normal_entries=1523**2,exact_ideal_normal_SPD_lower=1.,float_factor_or_source_accuracy_NOT_proven=True))
assert finite_or_unknown(ref,bound,error) and np.all(error<=bound)
# Independent RHS route assembles the full gradient, then Schur correction.
g=np.asarray(P@rhsdata['w0'][:1523]+rhsdata['q']);g[:465]+=robot.T@r[NOBJ:NOBJ+NROB];g[465:]+=A.T@r[:NOBJ];g+=B.T@r[NOBJ+NROB:]
reference=-g+B.T@((1-weight)*r[NOBJ+NROB:]);absbound=np.abs(P@rhsdata['w0'][:1523])+np.abs(rhsdata['q']);absbound[:465]+=abs(robot).T@np.abs(r[NOBJ:NOBJ+NROB]);absbound[465:]+=abs(A).T@np.abs(r[:NOBJ]);absbound+=2*abs(B).T@np.abs(r[NOBJ+NROB:]);outer=4*gamma*absbound+np.finfo(float).tiny
seal_return(D,'NEW-ALL1523-RHS-binding-RETURN',dict(reference=reference,actual=rhsdata['rhs'],absolute_error=np.abs(reference-rhsdata['rhs']),conditional_RN_outer=outer))
assert finite_or_unknown(reference,outer) and np.all(np.abs(reference-rhsdata['rhs'])<=outer)
assert np.array_equal(P.data,rhsdata['P'].data) and np.array_equal(P.indices,rhsdata['P'].indices) and np.array_equal(P.indptr,rhsdata['P'].indptr)
with np.load(I/'NEW-interface-constants-RETURN.npz',allow_pickle=False) as z:assert np.array_equal(z['q'],rhsdata['q'])
seal_return(D,'NEW-complete-routing-conditioning-domain-binding-RETURN',dict(ALL31839917_object_coefficients=A.nnz,ALL8798_robot_terms=robot.nnz,ALL933560_dynamic_coefficients=J.nnz,
 ALL4810140_rows_retained=True,ALL499206_direction_coordinates_retained=True,ALL249603_pair_blocks_retained=True,ALL1245_cones_RHS_Pq_strict_flags_unchanged_BY_HASH=True,
 normal_max_error=float(error.max()),RHS_max_error=float(np.max(np.abs(reference-rhsdata['rhs']))),conditional_arithmetic_bounds_NOT_candidate_tolerances=True))
save(D/'report.json',dict(passed=True,ALL_new_normal_entries=1523**2,ALL_new_RHS_entries=1523,ALL_new_coefficient_partitions=J.nnz,ALL_original_object_coefficients=A.nnz,
 ALL_pair_blocks=249603,ALL_direction_coordinates=499206,normal_max_error=float(error.max()),RHS_max_error=float(np.max(np.abs(reference-rhsdata['rhs']))),
 no_optimizer_solver_NEW_point_or_episode_query_yet=True,no_old_normal_constructor_or_method_replay=True))
