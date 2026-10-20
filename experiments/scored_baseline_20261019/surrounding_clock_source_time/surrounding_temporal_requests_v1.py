"""NEW boundary/fixed-term substitution for original full-clock finger gates.
Fixed F values are a restricted construction choice, never measured source truth.
"""
from fractions import Fraction as F
import half_angle_temporal_algebra_v1 as h

def signature(row):
 r=row.get('request',row);return(r['kind_of_constraint'],r['component'],tuple(r['clocks']),tuple(r['weights']))

def requests(original,fixed,timestamps,frames,velocity_limits,emit):
 original=[[F(v)for v in row]for row in original];fixed=[[F(v)for v in row]for row in fixed];ts=list(map(F,timestamps));frames=set(frames);n=len(ts)
 if n<3 or len(original)!=n or len(fixed)!=n or any(len(v)!=len(velocity_limits)for v in original+fixed):raise ValueError('explicit full clock/component shape required')
 if not frames or any(type(v)is not int or not 0<=v<n for v in frames):raise ValueError('invalid active frame')
 if any(b<=a for a,b in zip(ts,ts[1:])):raise ValueError('positive actual timestamps required')
 tolerance=F(float(1e-7));cap=F(float(.05))+tolerance;guard=F(1,10**14);result=[]
 def add(kind,component,clock,weights,lower,upper,origin):
  if not any(t in frames for t in clock):return
  fixed_terms=[dict(frame=t,weight=w,angle=h.rec(fixed[t][component]),contribution=h.rec(w*fixed[t][component]))for t,w in zip(clock,weights,strict=True)if t not in frames];constant=sum((h.unrec(v['contribution'])for v in fixed_terms),F(0));active=[(t,w)for t,w in zip(clock,weights,strict=True)if t in frames];r=dict(kind_of_constraint=kind,component=component,clocks=list(clock),weights=list(weights),ORIGINAL_lower_rad=h.rec(lower),ORIGINAL_upper_rad=h.rec(upper),exact_binary_input_origin=origin,fixed_outside_terms=fixed_terms,fixed_sum=h.rec(constant),variable_clocks=[t for t,w in active],variable_weights=[w for t,w in active],variable_names=[f'u_{t}_{component}'for t,w in active],folded_ORIGINAL_lower_rad=h.rec(lower-constant),folded_ORIGINAL_upper_rad=h.rec(upper-constant),necessary_arithmetic_conversion_guard_rad=h.rec(guard),ORIGINAL_motion_tolerance_rad=h.rec(tolerance),no_original_gate_changed=True,fixed_F_boundary_is_restricted_construction_NOT_observation=True)
  emit(dict(kind='AFTER_NEW_fullclock_boundary_request_RETURN',RETURN=r));result.append(r)
 for component,vmax in enumerate(velocity_limits):
  vmax=F(vmax)
  if vmax<=0:raise ValueError('positive actual URDF velocity required')
  for t in range(1,n):
   base=original[t][component]-original[t-1][component];add('original_adjacent_added',component,[t,t-1],[1,-1],base-cap,base+cap,dict(original_angles_rad=[h.rec(original[t][component]),h.rec(original[t-1][component])],original_added_cap_rad=h.rec(F(float(.05))),original_motion_tolerance_rad=h.rec(tolerance)))
   dt=ts[t]-ts[t-1];bound=(vmax+tolerance)*dt;add('actual_URDF_velocity',component,[t,t-1],[1,-1],-bound,bound,dict(actual_URDF_velocity_rad_s=h.rec(vmax),timestamps_s=[h.rec(ts[t]),h.rec(ts[t-1])],positive_delta_time_s=h.rec(dt),original_velocity_tolerance_rad_s=h.rec(tolerance)))
  for t in range(2,n):
   base=original[t][component]-2*original[t-1][component]+original[t-2][component];add('original_second_added',component,[t,t-1,t-2],[1,-2,1],base-cap,base+cap,dict(original_angles_rad=[h.rec(original[k][component])for k in[t,t-1,t-2]],original_added_cap_rad=h.rec(F(float(.05))),original_motion_tolerance_rad=h.rec(tolerance)))
 return result

def new_polynomial_request(request,angle_bounds,half_bounds,emit):
 names=request['variable_names'];weights=request['variable_weights'];guard=h.unrec(request['necessary_arithmetic_conversion_guard_rad']);lo=h.unrec(request['folded_ORIGINAL_lower_rad']);hi=h.unrec(request['folded_ORIGINAL_upper_rad']);emit(dict(kind='BEFORE_NEW_surrounding_clock_polynomial',request=request,variable_angle_bounds=[[h.rec(v)for v in angle_bounds[n]]for n in names],variable_half_angle_bounds=[[h.rec(v)for v in half_bounds[n]]for n in names]));result=h.build(weights,[angle_bounds[n]for n in names],[half_bounds[n]for n in names],lo-guard,hi+guard);emit(dict(kind='AFTER_NEW_surrounding_clock_polynomial_RETURN',RETURN=result));return result
