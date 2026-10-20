"""NEW-request phasor coefficients with durable intermediate returns.
Old22 temporal polynomials are copied by hash and NEVER passed here.
"""
from fractions import Fraction as F
import math
import half_angle_temporal_algebra_v1 as h

def serial(v):
 if isinstance(v,F):return h.rec(v)
 if isinstance(v,dict):return{k:serial(a)for k,a in v.items()}
 if isinstance(v,(list,tuple)):return list(map(serial,v))
 return v

def build(request,angle_bounds,half_bounds,emit):
 names=request['variable_names'];weights=request['variable_weights'];guard=h.unrec(request['necessary_arithmetic_conversion_guard_rad']);lo=h.unrec(request['folded_ORIGINAL_lower_rad'])-guard;hi=h.unrec(request['folded_ORIGINAL_upper_rad'])+guard;box=[angle_bounds[n]for n in names];ub=[half_bounds[n]for n in names];qrange=h.linear_range(weights,box);chart=all(-3<F(a)<=F(b)<3 for a,b in box);branch=all(abs(v-end)<6 for v in qrange for end in[lo,hi]);status='certified_finite_angle_branch'if chart and branch else'UNRESOLVED_angle_branch_or_half_angle_pole';result=dict(weights=weights,angle_combination_range=[h.rec(v)for v in qrange],requested_interval=[h.rec(lo),h.rec(hi)],angle_branch=status,squared_phasor_modulus_denominator_lower_bound=1,positive_denominator_identity='prod((1+u_j^2)^abs(k_j)) >=1',tangent_difference_denominator_assumed_positive=False,boundaries=[]);emit(dict(kind='AFTER_NEW_request_branch_RETURN',RETURN=result))
 if not chart or not branch:return result
 phasor=h.phasor(weights);emit(dict(kind='AFTER_NEW_request_phasor_RETURN',RETURN=[dict(power=list(e),real=a,imag=b)for e,(a,b)in sorted(phasor.items())]))
 for name,threshold,direction in[('lower',lo,1),('upper',hi,-1)]:
  emit(dict(kind='BEFORE_NEW_request_trig_enclosure',boundary=name,threshold=h.rec(threshold)));trig=h.sincos(threshold/2);emit(dict(kind='AFTER_NEW_request_trig_enclosure_RETURN',boundary=name,RETURN=serial(trig)));rows=[];uniform=F(0)
  for e,(real,imag)in sorted(phasor.items()):
   coefficient=h.scale(h.subtract(h.scale(trig['cos'],imag),h.scale(trig['sin'],real)),direction);midpoint=sum(coefficient)/2;error=(coefficient[1]-coefficient[0])/2;monomial=math.prod(max(abs(F(a)),abs(F(b)))**power for power,(a,b)in zip(e,ub,strict=True));uniform+=error*monomial;row=dict(exponent=list(e),coefficient_lower=h.rec(coefficient[0]),coefficient_upper=h.rec(coefficient[1]),midpoint=h.rec(midpoint));emit(dict(kind='AFTER_NEW_request_coefficient_interval_RETURN',boundary=name,RETURN=row,domain_absolute_monomial_bound=h.rec(monomial),cumulative_uniform_error=h.rec(uniform)));rows.append(row)
  boundary=dict(name=name,sense='>=0',threshold=h.rec(threshold),trig_enclosure=serial(trig),coefficients=rows,uniform_coefficient_error=h.rec(uniform),necessary_midpoint_polynomial_lower_threshold=h.rec(-uniform));emit(dict(kind='AFTER_NEW_request_boundary_RETURN',RETURN=boundary));result['boundaries'].append(boundary)
 emit(dict(kind='AFTER_NEW_full_request_polynomial_RETURN',RETURN=result));return result
