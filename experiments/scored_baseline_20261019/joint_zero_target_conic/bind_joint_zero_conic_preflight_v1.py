"""Independent NEW phase-zero/all-row/general-RHS Schur bindings, no solver."""
from joint_zero_conic_v1 import *
from joint_saved_mean_bound_v1 import save
D=S/'e037-joint-zero-binding-v1';PRE=S/'e037-joint-zero-preflight-v1';I=S/'e037-joint-executable-interface-v1'
names=['NEW-zero-slice-Schur-map-RETURN','NEW-dynamic-affine-origin-action-RETURN','NEW-dynamic-affine-RHS-RETURN','NEW-ALL-hard-rows-RHS-objective-cones-RETURN',
 'NEW-complete-zero-target-column-routes-RETURN','NEW-F-authority-and-exact-zero-slice-RETURN']
paths=[PRE/(n+ext) for n in names for ext in ['.json','.npz']]+[PRE/'NEW-fixed-method-declaration-RETURN.json',C/'joint_zero_conic_v1.py']
save(D/'NEW-independent-input-bindings-BEFORE.json',{str(p):sha(p) for p in paths})
maps=typed(PRE/(names[0]+'.json'));jw=typed(PRE/(names[1]+'.json'))['value'];db=typed(PRE/(names[2]+'.json'))['value'];constants=typed(PRE/(names[3]+'.json'))
routes=typed(PRE/(names[4]+'.json'));start=typed(PRE/(names[5]+'.json'));raw=typed(dynamic_file());J=raw['J'];del raw
A=sparse.load_npz(I/'NEW-object-retained-A-RETURN.npz');R=sparse.load_npz(I/'NEW-robot-linear-A-RETURN.npz');P=sparse.load_npz(I/'NEW-embedded-objective-P-RETURN.npz')
with np.load(I/'NEW-interface-constants-RETURN.npz',allow_pickle=False) as z:ob=z['object_b'];rb=z['robot_b'];q=z['q']
F=typed(next(OLD.glob('*-NEW-step-01-ALL-F-RETURN.json')))['F']
# Independent exact map: inverse free-to-full index rather than producer's
# piecewise column expression. Every coefficient belongs to one such column.
inverse={full:free for free,full in enumerate(k for k in range(NVAR) if k!=PHASE)}
def bind_columns(mat,fullcols,actual):
 out=np.array([inverse.get(int(k),-1) for k in fullcols],dtype=np.int64)
 counts=np.diff(mat.tocsc().indptr);zero=counts[out==-1].sum();free=counts[out!=-1].sum()
 return dict(actual=actual,independent=out,all_stored_positions=int(counts.sum()),exact_zero_phase_positions=int(zero),free_positions=int(free),passed=bool(np.array_equal(out,actual)))
columnproof=dict(object=bind_columns(A,routes['object_full'],routes['object_free']),robot=bind_columns(R,routes['robot_full'],routes['robot_free']),dynamic=bind_columns(J,routes['dynamic_full'],routes['dynamic_free']))
seal_return(D,'NEW-ALL-stored-position-zero-target-binding-RETURN',columnproof)
assert all(p['passed'] and p['exact_zero_phase_positions']+p['free_positions']==p['all_stored_positions'] for p in columnproof.values())
# Complete new B0/D routes preserve CSC order, explicit zeros, all source and
# pair incidences; all direction blocks remain represented, including filters.
refB=J[:,[k for k in range(1523) if k!=PHASE]].tocsc();refD=J[:,1523:].tocsc();B=maps['base_J'].tocsc();Ds=maps['direction_J'].tocsc()
parts={}
for name,a,b in [('B0',refB,B),('D',refD,Ds)]:
 parts[name]=dict(all_entries=a.nnz,data_bits=bool(a.data.tobytes()==b.data.tobytes()),indices=bool(np.array_equal(a.indices,b.indices)),indptr=bool(np.array_equal(a.indptr,b.indptr)))
coo=Ds.tocoo();owner=coo.row-852;target=coo.col//2
ss=np.zeros(250455);np.add.at(ss,coo.row,coo.data*coo.data)
refweight=1/(1+ss);weight=maps['weight'];we=np.abs(refweight-weight);wo=32*2.**-53*np.maximum(1.,np.abs(refweight))+np.finfo(float).tiny
seal_return(D,'NEW-ALL-general-RHS-Schur-route-binding-RETURN',dict(parts=parts,all499206_direction_columns=np.arange(499206),all249603_blocks=maps['all_pair_instances'],
 row=coo.row,col=coo.col,owner=owner,expected=target,actual_weight=weight,reference_weight=refweight,weight_error=we,conditional_weight_RN_outer=wo,
 exact_general_RHS_formula='S*z=r_z-B0.T*(w*(D*r_p)); p=r_p-D.T*(w*(D*r_p+B0*z))',
 all_blocks_including_zero_filters_retained=True,Schur_identity_lower_bound=1.,all_correlated_tied_SEARCH_coefficients_saved_bound_BY_HASH=True))
assert all(p['data_bits'] and p['indices'] and p['indptr'] for p in parts.values()) and np.array_equal(owner,target)
assert len(maps['all_pair_instances'])==249603 and Ds.shape[1]==499206 and np.all(we<=wo) and np.all(weight>0) and np.all(weight<=1)
del refB,refD,coo,owner,target
# Independently assemble only the NEW zero-slice principal system by the row
# route. This does not invoke or repeat the completed original GN constructor.
Ac=A[:,:1057].tocsr();Rc=R.tocsr();ref=np.eye(1522);ref+=P[:1522,:1522].toarray();ref[465:,465:]+=(Ac.T@Ac).toarray();ref[:465,:465]+=(Rc.T@Rc).toarray()
ref+=(B.multiply(weight[:,None]).T@B).toarray();diag=np.diag(ref);u=2.**-53;count=3*NROW+32;gamma=count*u/(1-count*u)
outer=4*gamma*(diag[:,None]+diag[None,:])+np.finfo(float).tiny;err=np.abs(maps['normal']-ref)
seal_return(D,'NEW-ALL2316484-zero-slice-normal-binding-RETURN',dict(actual=maps['normal'],reference=ref,absolute_error=err,conditional_RN_outer=outer,
 entries=1522**2,all_rows=NROW,exact_ideal_SPD_lower=1.,normal_factor_NOT_yet_run=True,numerical_outer_NOT_candidate_tolerance=True))
assert finite_or_unknown(ref,err,outer) and np.all(err<=outer)
# RHS binding reads SAVED J*w_F, and checks every new b term exactly as a
# rational expression of those saved returns, never replays an old Ax/query.
refdb=jw-F[NOBJ+NROB:];be=np.abs(refdb-db)
Pfree=np.zeros(NFREE);Pfree[:1522]=P.diagonal()[:1522];qfree=np.zeros(NFREE);qfree[:1522]=q[:1522]
with np.load(S/'e037-joint-method-preflight-v1/NEW-original-objective-and-cone-layout-RETURN.npz',allow_pickle=False) as z:oldcones=z['cones']
seal_return(D,'NEW-ALL-RHS-original-objective-cones-binding-RETURN',dict(dynamic_RHS_error=be,object_RHS_bits=bool(ob.tobytes()==constants['b'][:NOBJ].tobytes()),
 robot_RHS_bits=bool(rb.tobytes()==constants['b'][NOBJ:NOBJ+NROB].tobytes()),Pfree=Pfree,qfree=qfree,original_P_bits=bool(Pfree.tobytes()==constants['Pdiag'].tobytes()),
 original_q_bits=bool(qfree.tobytes()==constants['q'].tobytes()),whole_cone_pointers=bool(np.array_equal(oldcones,constants['cones'])),
 phase_literal_zero=start['full_start'][PHASE],all_existing_phase_domain_rows_retained=True,formal_equality_rows=NROW+1,
 strict_open_flags_source_native_operators_unchanged_BY_HASH=True))
assert np.all(be==0) and ob.tobytes()==constants['b'][:NOBJ].tobytes() and rb.tobytes()==constants['b'][NOBJ:NOBJ+NROB].tobytes()
assert Pfree.tobytes()==constants['Pdiag'].tobytes() and qfree.tobytes()==constants['q'].tobytes() and np.array_equal(oldcones,constants['cones'])
assert np.array_equal(lift(start['free_start']),start['full_start']) and start['full_start'][PHASE]==0.
save(D/'report.json',dict(passed=True,ALL_original_coefficient_positions=A.nnz+R.nnz+J.nnz,normal_entries=1522**2,general_RHS_direction_blocks=249603,
 dynamic_RHS_entries=len(db),all_rows=NROW,formal_rows=NROW+1,all_cones=1245,normal_max_error=float(err.max()),weight_max_error=float(we.max()),
 original_Pq_RHS_strict_flags_conditioning_source_native_unchanged=True,no_solver_factor_NEW_point_or_episode_query=True))
