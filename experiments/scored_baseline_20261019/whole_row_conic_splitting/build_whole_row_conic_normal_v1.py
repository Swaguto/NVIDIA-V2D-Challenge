"""NEW normal-system constructor from frozen coefficients; no optimization."""
from whole_row_conic_split_v1 import *
from scipy.linalg import cho_factor
from fractions import Fraction as F
O=N
A,b,q,P,meta=load_frozen()
dump(O/'NEW-normal-construction-BEFORE.json',dict(frozen_layout_sha256=sha(I/'NEW-structured-layout-RETURN.json'),rows=A.shape[0],columns=A.shape[1],stored_entries=A.nnz,rho=1,normal='P + A.T A',all_rows=True,no_diagonal_loading_no_row_column_scaling=True,source_original_constraints_UNCHANGED=True))
# This NEW normal product is distinct from old model/constructor/solves.
G=(A.T@A).toarray()+P.toarray();save(O/'NEW-ALL-row-normal-RETURN.npz',normal=G)
assert np.isfinite(G).all()
L,lower=cho_factor(G,lower=True,overwrite_a=False,check_finite=True)
save(O/'NEW-Cholesky-constructor-RETURN.npz',factor=L,lower=np.array(lower))
witness=[]
for col,row in coordinate_witnesses(A,meta):
 ar=A[row];nonzero=[(int(j),float(v)) for j,v in zip(ar.indices,ar.data) if v!=0]
 witness.append(dict(column=col,row=row,nonzero=nonzero))
dump(O/'NEW-coordinate-SPD-witness-RETURN.json',witness)
assert len(witness)==1058 and sorted(w['column'] for w in witness)==list(range(1058))
assert all(len(w['nonzero'])==1 and w['nonzero'][0][0]==w['column'] and w['nonzero'][0][1]!=0 for w in witness)
minsq=min(F.from_float(w['nonzero'][0][1])**2 for w in witness)
# Exact factored normal maps retain every CSR coefficient/row pointer BY HASH.
normal_map=dict(schema='v2d.NEW-whole-row-normal-map.v1',rows=A.shape[0],columns=A.shape[1],rho=1,all_row_term='sum over ALL r A[r,i]*A[r,j]',P_prefix_BY_HASH=sha(M/'NEW-canonical-P-RETURN.npz'),input_data_sha256=sha(I/'NEW-CSR-data-RETURN.npy'),input_indices_sha256=sha(I/'NEW-CSR-indices-RETURN.npy'),input_indptr_sha256=sha(I/'NEW-CSR-indptr-RETURN.npy'),cones=meta['cones'],strict_pointers=meta['strict_frame_pointers'],exact_SPD_lower_bound=str(minsq),normal_float_is_approximate=True,coordinate_rows_are_proof_subset_NOT_constraint_drop=True,all_other_rows_retained=True)
dump(O/'NEW-complete-factored-normal-map-RETURN.json',normal_map)
dump(O/'report.json',dict(schema='v2d.NEW-whole-row-conic-normal-construction.v1',rows=A.shape[0],columns=A.shape[1],stored_coefficients=A.nnz,normal_shape=list(G.shape),ALL900SOC4_151SOC5=True,coordinate_witnesses=1058,exact_SPD_lower_bound=str(minsq),factor_return_saved=True,normal_factor_NOT_episode_solve=True,no_source_or_model_replay=True))
