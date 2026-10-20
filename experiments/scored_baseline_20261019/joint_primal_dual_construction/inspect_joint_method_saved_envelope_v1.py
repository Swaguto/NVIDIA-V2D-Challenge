"""NEW exact saved sparse-envelope defect scope; no operators or method replay."""
from pathlib import Path
from fractions import Fraction
import json,hashlib,numpy as np
from scipy import sparse
from joint_lossless_returns_v1 import seal_return
from joint_constraint_interface_v1 import S,C,save,sha
O=S/'e037-joint-primal-dual-evaluation-v1';D=S/'e037-joint-method-envelope-scope-v1'
p=next(O.glob('*-ALL-dynamic-F-J-envelope-RETURN.json'));doc=json.loads(p.read_text())
save(D/'NEW-saved-input-bindings-RETURN.json',{str(q):sha(q) for q in [p,p.with_suffix('.npz'),O/'report.json',O/'guard.json',S/'e037-joint-primal-dual-return-audit-v1/report.json']})
with np.load(p.with_suffix('.npz'),allow_pickle=False) as z:
 def array(t):
  a=z[t['key']];assert hashlib.sha256(a.tobytes()).hexdigest()==t['bytes_sha256'];return a
 def matrix(t):return sparse.csc_matrix((array(t['data']),array(t['indices']),array(t['indptr'])),shape=t['shape'])
 items={k['value']:v for k,v in doc['tree']['items']};J=matrix(items['J']);E=matrix(items['absolute_tie_envelope'])
assert np.array_equal(J.indptr,E.indptr) and np.array_equal(J.indices,E.indices)
bad=np.flatnonzero(np.abs(J.data)>E.data);columns=np.searchsorted(J.indptr,bad,side='right')-1;rows=J.indices[bad]
seal_return(D,'NEW-ALL-actual-envelope-defects-RETURN',dict(data_positions=bad,rows=rows,columns=columns,J=J.data[bad],envelope=E.data[bad],
 stored_nnz=J.nnz,shape=J.shape,all_J_E_finite=bool(np.isfinite(J.data).all() and np.isfinite(E.data).all())))
ratios=[];gaps=[];steps=[]
for k in bad:
 a=Fraction(float(abs(J.data[k])));b=Fraction(float(E.data[k]));gaps.append(a-b)
 ulp=Fraction(float(np.spacing(abs(J.data[k]))));steps.append((a-b)/ulp if ulp else Fraction(0));ratios.append(a/b if b else None)
result=dict(passed=True,actual_defect_count=len(bad),ALL_saved_dynamic_coefficients=J.nnz,all_inputs_finite=bool(np.isfinite(J.data).all() and np.isfinite(E.data).all()),
 defect_rows_min=int(rows.min()) if len(rows) else None,defect_rows_max=int(rows.max()) if len(rows) else None,
 source_or_depth_domain_defects=int(np.count_nonzero(rows<852)),pair_defects=int(np.count_nonzero(rows>=852)),
 exact_max_gap=str(max(gaps,default=Fraction(0))),max_gap_float=float(max(gaps,default=Fraction(0))),exact_max_ULP_fraction=str(max(steps,default=Fraction(0))),
 current_method_stopped_BEFORE_metrics_projection_adjoint_update=True,no_old_return_repair_or_method_retry=True,
 saved_numeric_envelope_is_NOT_outward_for_ALL_actual_SEARCH_coefficients=True,root_cause_beyond_saved_arithmetic_SCOPE_NOT_proven=True,
 no_candidate_tolerance_or_source_native_gate_changed=True,no_candidate_qualification=True)
seal_return(D,'NEW-exact-defect-scope-RETURN',result)
assert len(bad)>0 and result['source_or_depth_domain_defects']==0 and result['all_inputs_finite']
save(D/'report.json',result)
