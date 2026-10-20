"""Only TWO NEW overflow-precondition contracts; old12 reused by hash."""
from whole_row_conic_runtime_v1 import *
from fractions import Fraction as F
O=S/'e037-whole-row-conic-range-contracts-v1'
a=guarded_projection(np.array([0.,float(2**501),0.,0.]),0,[(0,4,'new_range')]);dump(O/'NEW-outside-range-RETURN.json',a)
assert a['status']=='UNRESOLVED_projection_norm_range' and 4*F(2)**1000 < F.from_float(np.finfo(float).max)
b=guarded_projection(np.array([0.,float(2**499),0.,0.]),0,[(0,4,'new_range')]);save(O/'NEW-safe-range-actual-RETURN.npz',value=b['value']);dump(O/'NEW-safe-range-meta-RETURN.json',{k:v for k,v in b.items() if k!='value'})
assert b['status']=='projected_ALL_product_coordinates' and np.array_equal(b['value'],[float(2**498),float(2**498),0.,0.])
dump(O/'report.json',dict(schema='v2d.NEW-whole-row-conic-range-contracts.v1',NEW_contracts=2,passed=True,old12controls_NOT_replayed=True,projection_formula_parameters_UNCHANGED=True,precondition_NOT_original_acceptance_gate=True,finite_norm_scope_conditional=True))
