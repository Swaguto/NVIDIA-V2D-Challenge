"""Conditional actual NEW zero-slice linear SEARCH return guard, not feasibility."""
from fractions import Fraction as Q
import numpy as np
def linear_return_guard(normal,solution,rhs,record,tag):
 n=len(rhs);k=10*n+32;g=Q(k,2**53-k);upper=float(g)
 if Q.from_float(upper)<g:upper=np.nextafter(upper,np.inf)
 product=normal@solution;record(tag+'-normal-solution-product-RETURN',dict(value=product))
 residual=product-rhs;record(tag+'-actual-linear-residual-RETURN',dict(value=residual))
 absolute_product=np.abs(normal)@np.abs(solution);record(tag+'-absolute-linear-product-RETURN',dict(value=absolute_product))
 outer=2*upper*(absolute_product+np.abs(rhs))+np.finfo(float).tiny
 result=dict(residual=residual,conditional_RN_outer=outer,all_within=bool(np.all(np.abs(residual)<=outer)),operation_allowance=k,
  gamma_rational_numerator=g.numerator,gamma_rational_denominator=g.denominator,gamma_outward_float=upper,
  finite_RN_no_overflow_condition=True,linear_SEARCH_return_guard_NOT_candidate_tolerance_or_feasibility=True,no_global_binary_identity=True)
 record(tag+'-conditional-linear-return-guard-RETURN',result)
 return result
