"""Independent local-step coefficient maps, plus trusted private runtime cache."""
from fractions import Fraction as F
import math
import numpy as np
from scipy import sparse
import sequential_camera_conic_v1 as method

class CachedAngleRuntime(method.SavedAngleModel):
 def prepare_initial(self,emit):
  self.cached_initial_view=super().snapshot(self.x0,emit,'NEW-initial-runtime-binding',initial=True)
 def snapshot(self,x,emit,label,initial=False):
  if initial and hasattr(self,'cached_initial_view'):
   if not np.array_equal(x,self.x0):raise ValueError('unmodified F required for cached initial return')
   emit(dict(kind='AFTER_initial_runtime_view_REUSED_BY_HASH',label=label,source_initial_F_only=True,no_actual_URDF_derivative_or_merit_recomputed=True));return self.cached_initial_view
  return super().snapshot(x,emit,label,initial)

def exact_expected_step(system,view):
 """Separate scalar/rational implementation, no step_problem call.

 This binds only the LOCAL affine predictor and numerical packing. It does not
 make the nonlinear problem convex or certify source/robot/task infeasibility.
 """
 n=system.n;entries={};rhs=[];origins=[];magnitudes={};rhs_magnitudes=[]
 def offer(terms,b,origin,coefficient_magnitudes=None,rhs_magnitude=None):
  i=len(rhs)
  for j,v in terms.items():
   if v:entries[(i,j)]=F(v)
   magnitudes[(i,j)]=F(coefficient_magnitudes[j])if coefficient_magnitudes is not None and j in coefficient_magnitudes else abs(F(v))
  rhs.append(F(b));rhs_magnitudes.append(abs(F(b))if rhs_magnitude is None else F(rhs_magnitude));origins.append(origin)
 x=list(map(F,view['x']))
 for i,terms in enumerate(system.exact_terms):
  a={j:v for j,v in terms};a[n]=-F(float(system.linear_scale[i]));b=F(float(system.b[i]))-sum((v*x[j]for j,v in terms),F(0));offer(a,b,dict(kind='original_linear_predictor',original_row=i,original_saved_exact_bound_separately_preserved=True),rhs_magnitude=abs(F(float(system.b[i])))+sum((abs(v*x[j])for j,v in terms),F(0)))
 for i,j in enumerate(system.half_indices):
  du=F(float(view['half_derivatives'][i]));u=F(float(view['half_values'][i]));scale=F(float(method.CONFIG['halfangle_scale']));offer({j:-du,n:-scale},u-F(float(system.half_lo[i])),dict(kind='retained_half_lower',domain=i));offer({j:du,n:-scale},F(float(system.half_hi[i]))-u,dict(kind='retained_half_upper',domain=i))
 for source in view['source']:
  offer({**{j:-F(float(source['J'][2,k]))for k,j in enumerate(source['columns'])},n:-F(float(method.CONFIG['depth_scale_m']))},F(float(source['C'][2]))-F(float(.01)),dict(kind='strict_depth_local_predictor',factor_id=source['id'],actual_gate_still_STRICT=True))
 offer({n:F(-1)},F(0),dict(kind='rho_nonnegative'))
 for j in range(n):
  for sign in[-1,1]:offer({j:F(sign)},F(float(method.CONFIG['max_step_rad'])),dict(kind='algorithm_trust_bound',column=j,sign=sign))
 nonnegative_rows=len(rhs);blocks=[]
 for i,source in enumerate(view['source']):
  C=list(map(lambda v:F(float(v)),source['C']));r=F(float(source['radius']));J=source['J'];B=source['B'];columns=source['columns'];start=len(rhs)
  offer({**{j:-r*F(float(J[2,k]))for k,j in enumerate(columns)},n:-F(float(system.source_scale[i]))},r*C[2],dict(kind='SOC_head',factor_id=source['id']))
  for a in range(2):offer({j:-sum((F(float(B[a,t]))*F(float(J[t,k]))for t in range(3)),F(0))for k,j in enumerate(columns)},sum((F(float(B[a,t]))*C[t]for t in range(3)),F(0)),dict(kind='SOC_tail',factor_id=source['id'],camera_plane_row=a),coefficient_magnitudes={j:sum((abs(F(float(B[a,t]))*F(float(J[t,k])))for t in range(3)),F(0))for k,j in enumerate(columns)},rhs_magnitude=sum((abs(F(float(B[a,t]))*C[t])for t in range(3)),F(0)))
  blocks.append(dict(factor_id=source['id'],start=start,dimension=3))
 return dict(entries=entries,rhs=rhs,origins=origins,coefficient_operation_magnitudes=magnitudes,RHS_operation_magnitudes=rhs_magnitudes,nonnegative_rows=nonnegative_rows,SOC_blocks=blocks,shape=(len(rhs),n+1))

def rational_record(v):return dict(numerator=str(v.numerator),denominator=str(v.denominator))
def validate_actual_step(payload,expected,system):
 """Check every actual coefficient and RHS against rational local formulas."""
 rec=payload['A'];A=sparse.csc_matrix((rec['data'],rec['indices'],rec['indptr']),shape=rec['shape']).tocoo();actual={(int(i),int(j)):F(float(v))for i,j,v in zip(A.row,A.col,A.data,strict=True)if v};keys=set(actual)|set(expected['entries']);cert=[];maximum=F(0);eps=F(1,2**52)
 if tuple(rec['shape'])!=expected['shape']or payload['nonnegative_rows']!=expected['nonnegative_rows']or len(payload['SOC_blocks'])!=len(expected['SOC_blocks']):raise ValueError('canonical dimension or cone ordering mismatch')
 for key in sorted(keys):
  want=expected['entries'].get(key,F(0));got=actual.get(key,F(0));error=abs(got-want);guard=32*eps*(1+expected['coefficient_operation_magnitudes'].get(key,abs(want)));maximum=max(maximum,error);cert.append(dict(row=key[0],column=key[1],exact_expected=rational_record(want),actual_minus_expected=rational_record(got-want),numerical_packing_guard=rational_record(guard),original_gate_NOT_changed=True))
  if error>guard:raise ValueError('local canonical coefficient binding failed')
 rhscert=[];rhsmax=F(0)
 for i,(got,want)in enumerate(zip(payload['b'],expected['rhs'],strict=True)):
  got=F(float(got));error=abs(got-want);guard=64*eps*(1+expected['RHS_operation_magnitudes'][i]);rhsmax=max(rhsmax,error);rhscert.append(dict(row=i,exact_expected=rational_record(want),actual_minus_expected=rational_record(got-want),numerical_packing_guard=rational_record(guard)))
  if error>guard:raise ValueError('local canonical RHS binding failed')
 P=payload['P'];P=sparse.csc_matrix((P['data'],P['indices'],P['indptr']),shape=P['shape']);diagonal=np.r_[np.full(system.n,method.CONFIG['damping']),0.]
 if P.shape!=(system.n+1,system.n+1)or(P-sparse.diags(diagonal)).nnz or not np.array_equal(payload['q'],np.r_[np.zeros(system.n),1.]):raise ValueError('quadratic objective binding failed')
 for actualblock,block in zip(payload['SOC_blocks'],expected['SOC_blocks'],strict=True):
  if actualblock['factor_id']!=block['factor_id']or actualblock['canonical_start_row']!=block['start']or actualblock['dimension']!=3:raise ValueError('SOC block binding failed')
 return dict(all_NEW_actual_canonical_coefficients_RHS_objective_cones_bound=True,columns=system.n+1,rows=expected['shape'][0],nonnegative_rows=expected['nonnegative_rows'],SOC_cones=len(expected['SOC_blocks']),coefficient_identities=len(cert),RHS_identities=len(rhscert),maximum_exact_coefficient_error=rational_record(maximum),maximum_exact_RHS_error=rational_record(rhsmax),all_coefficient_certificates=cert,all_RHS_certificates=rhscert,necessary_LOCAL_numerical_packing_diagnostics_NOT_original_candidate_tolerances=True,no_nonlinear_original_family_or_task_exclusion=True)
