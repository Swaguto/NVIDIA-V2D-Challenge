"""NEW certificate maps/bounds only. No matrix/solver/source regeneration."""
from exact_terminal_dual_bound_v1 import *
O=P;A=load_csr()
with np.load(M/'NEW-canonical-vectors-RETURN.npz') as z:b=z['b']
bounds,proof=constructor_bounds(A,b)
dump(O/'NEW-exact-rational-constructor-bounds-RETURN.json',[dict(column=j,lower=x[0],upper=x[1],magnitude=max(abs(x[0]),abs(x[1]))) for j,x in enumerate(bounds)])
dump(O/'NEW-bound-origin-necessity-proofs-RETURN.json',proof)
with np.load(M/'NEW-canonical-A-RETURN.npz') as z:ptr=z['indptr'];shape=z['shape'];entries=len(z['data'])
meta=json.loads((I/'NEW-structured-layout-RETURN.json').read_text());dual=E/'ONE-actual-raw-terminal-RETURN.npz'
plan=dict(schema='v2d.NEW-exact-terminal-dual-map.v1',rows=ROWS,columns=COLS,stored_coefficients=entries,phase_column=1057,phase_units='saved normalized physical_rho/r; physical lower bound is exact radius times normalized lower bound',ALL_column_functionals=[dict(column=j,CSC_start=int(ptr[j]),CSC_end=int(ptr[j+1]),formula='sum ALL saved A.data[p] * y[A.indices[p]]; exact zeros typed algebraic0') for j in range(COLS)],by_functional=dict(row_start=0,row_end=ROWS,formula='sum ALL b[r]*y[r]'),orthant_defect=dict(row_start=0,row_end=NN,formula='sum max(-y[r],0)*(abs(b[r])+sum ALL abs(A[r,j])*M[j])'),SOC_defect=[dict(start=a,size=k,formula='max(ceil96bit_norm(y_tail)-y_top,0)*(abs(b_top)+sum ALL abs(A_top,j)*M[j])') for a,k,_ in meta['cones']],strict_pointers=meta['strict_frame_pointers'],bounds_file_sha256=sha(O/'NEW-exact-rational-constructor-bounds-RETURN.json'),proof_file_sha256=sha(O/'NEW-bound-origin-necessity-proofs-RETURN.json'),inputs={str(p):sha(p) for p in [M/'NEW-canonical-A-RETURN.npz',M/'NEW-canonical-vectors-RETURN.npz',dual,I/'NEW-CSR-data-RETURN.npy',I/'NEW-CSR-indices-RETURN.npy',I/'NEW-CSR-indptr-RETURN.npy']},dual_selection='SOLE original raw terminal step32 u, ALL values unchanged',root_bits=96,exact_arithmetic_bit_cap=8192,no_objective_model_gate_or_raw_dual_change=True,no_source_accuracy_or_family_exclusion_claim=True)
dump(O/'NEW-complete-certificate-map-RETURN.json',plan)
assert tuple(shape)==(ROWS,COLS) and ptr[0]==0 and ptr[-1]==32306054 and len(proof)==COLS
# One elementary coordinate proof per variable. Other constraints remain present.
assert all(lo<=hi for lo,hi in bounds)
dump(O/'report.json',dict(schema='v2d.NEW-exact-terminal-dual-map-construction.v1',rational_necessary_bounds=COLS,origin_proofs=len(proof),ALL_CSC_coefficient_functional_spans=entries,all_by_rows=ROWS,all_orthant_rows=NN,whole_SOC4=900,whole_SOC5=151,unused_float_bound_file_NOT_used=True,no_numeric_dual_bound_evaluation_yet=True))
