"""Independent SAVED-return audit, no Ax/adjoint/solve/projector/query replay."""
from joint_saved_mean_bound_v1 import S,C,typed,sha,save
from joint_lossless_returns_v1 import seal_return
from fractions import Fraction as Q
from pathlib import Path
import numpy as np,json,time
D=S/'e037-joint-zero-return-audit-v1';E=S/'e037-joint-zero-evaluation-v1';PRE=S/'e037-joint-zero-preflight-v1'
NROW=4810140;NOBJ=4551571;NROB=8114;NFREE=500728;PHASE=1522;NDYN=250455
start=time.monotonic();files=sorted(E.glob('*.json'));save(D/'NEW-SAVED-input-bindings-BEFORE.json',{str(p):sha(p) for p in files})
array_entries=0;typed_returns=0
def allfinite(value):
 global array_entries
 if isinstance(value,np.ndarray):
  array_entries+=value.size
  return value.dtype.kind not in 'fc' or bool(np.isfinite(value).all())
 if isinstance(value,dict):return all(allfinite(k) and allfinite(v) for k,v in value.items())
 if isinstance(value,(list,tuple)):return all(allfinite(v) for v in value)
 if isinstance(value,(float,np.floating)):return bool(np.isfinite(value))
 return True
for p in files:
 if 'array_file' in json.loads(p.read_text()):
  returned=typed(p);finite=allfinite(returned)
  if not finite:raise ValueError('nonfinite SAVED return '+p.name)
  typed_returns+=1;del returned
seal_return(D,'NEW-ALL-SAVED-typed-bits-finiteness-RETURN',dict(typed_returns=typed_returns,array_entries=array_entries,all_finite=True,all_NPZ_bit_hashes_checked=True))
con=typed(PRE/'NEW-ALL-hard-rows-RHS-objective-cones-RETURN.json');b=con['b'];q=con['q'];P=con['Pdiag'];cones=[tuple(map(int,a)) for a in con['cones']]
def get(name):
 paths=list(E.glob('*-'+name+'.json'))
 if len(paths)!=1:raise ValueError((name,len(paths)))
 return typed(paths[0])
initial=get('NEW-method-input-BEFORE');zx=initial['zx'];zy=initial['zy'];del initial
orthant=np.ones(NROW,dtype=np.bool_)
for st,wi in cones:orthant[st:st+wi]=False
summary=[];total_row_identities=0;total_branch_failures=0;total_projected_SOC_defects=0;max_linear_ratio=0.
# An independent scalar exact-rational projector relation witness: saved norm
# inputs are checked by squared RN allowances, without calling any norm/projector.
u=Q(1,2**53);eta=Q(1,2**1074)
def frac(x):return Q.from_float(float(x))
def near(actual,ideal,operations=32):
 error=abs(frac(actual)-ideal);gamma=operations*u/(1-operations*u)
 return error<=gamma*abs(ideal)+operations*eta
for step in range(1,17):
 tag=f'NEW-update-{step:02d}';inp=get(tag+'-input-BEFORE');assert inp['zx'].tobytes()==zx.tobytes() and inp['zy'].tobytes()==zy.tobytes();del inp
 t=get(tag+'-dual-minus-RHS-RETURN')['value'];at=get(tag+'-ALL500728-adjoint-RETURN')['value'];rhs=get(tag+'-full-resolvent-RHS-RETURN')['value']
 # Relations between durable operator returns, not operator execution.
 error_t=np.abs(t.astype(np.longdouble)-(zy.astype(np.longdouble)-b.astype(np.longdouble)))
 bound_t=4*np.longdouble(2.**-53)*(np.abs(zy.astype(np.longdouble))+np.abs(b.astype(np.longdouble)))+np.longdouble(2.)**-1070
 assert np.all(error_t<=bound_t)
 error_rhs=np.abs(rhs.astype(np.longdouble)-(zx.astype(np.longdouble)-q.astype(np.longdouble)-at.astype(np.longdouble)))
 bound_rhs=8*np.longdouble(2.**-53)*(np.abs(zx.astype(np.longdouble))+np.abs(q.astype(np.longdouble))+np.abs(at.astype(np.longdouble)))+np.longdouble(2.)**-1070
 assert np.all(error_rhs<=bound_rhs);del error_t,bound_t,error_rhs,bound_rhs,at
 ux=get(tag+'-ALL-free-linear-resolvent-RETURN')['value'];z=get(tag+'-Schur-solve-RETURN')['value'];dp=get(tag+'-D-adjoint-backsub-RETURN')['value']
 assert ux[:1522].tobytes()==z.tobytes()
 errp=np.abs(ux[1522:].astype(np.longdouble)-(rhs[1522:].astype(np.longdouble)-dp.astype(np.longdouble)))
 guardp=4*np.longdouble(2.**-53)*(np.abs(rhs[1522:].astype(np.longdouble))+np.abs(dp.astype(np.longdouble)))+np.longdouble(2.)**-1070
 assert np.all(errp<=guardp);del z,dp,errp,guardp,rhs
 guard=get(tag+'-conditional-linear-return-guard-RETURN');prod=get(tag+'-normal-solution-product-RETURN')['value'];srhs=get(tag+'-Schur-RHS-RETURN')['value'];res=get(tag+'-actual-linear-residual-RETURN')['value']
 assert np.array_equal(res,guard['residual']) and np.all(np.isfinite(guard['conditional_RN_outer'])) and np.all(np.abs(res)<=guard['conditional_RN_outer'])
 assert all(near(res[k],frac(prod[k])-frac(srhs[k])) for k in range(1522))
 max_linear_ratio=max(max_linear_ratio,float(np.max(np.divide(np.abs(res),guard['conditional_RN_outer'],where=guard['conditional_RN_outer']>0))))
 del guard,prod,srhs,res
 ax=get(tag+'-ALL4810140-action-RETURN')['value'];a=get(tag+'-object-action-RETURN')['value'];r=get(tag+'-robot-action-RETURN')['value'];j=get(tag+'-dynamic-action-RETURN')['value']
 assert ax[:NOBJ].tobytes()==a.tobytes() and ax[NOBJ:NOBJ+NROB].tobytes()==r.tobytes() and ax[NOBJ+NROB:].tobytes()==j.tobytes();del a,r,j
 uy=get(tag+'-dual-resolvent-RETURN')['value'];reflection=get(tag+'-whole-K-reflection-BEFORE')['value'];pr=get(tag+'-whole-K-projection-RETURN');projection=pr['value'];update=get(tag+'-split-update-RETURN')
 # Extended-precision checks use outward binary64 operation allowances; they
 # certify only saved state algebra and are never candidate feasibility gates.
 def relation(value,terms,signs,multiplier=16):
  ideal=np.zeros(len(value),dtype=np.longdouble);magnitude=np.zeros(len(value),dtype=np.longdouble)
  for term,sign in zip(terms,signs):ideal+=np.longdouble(sign)*term.astype(np.longdouble);magnitude+=abs(sign)*np.abs(term.astype(np.longdouble))
  error=np.abs(value.astype(np.longdouble)-ideal);outer=multiplier*np.longdouble(2.**-53)*magnitude+np.longdouble(2.)**-1068
  return bool(np.all(error<=outer)),float(np.max(error))
 ru,e1=relation(uy,[t,ax],[1,1]);rr,e2=relation(reflection,[uy,zy],[2,-1]);rn,e3=relation(update['zy'],[zy,projection,uy],[1,1,-1])
 assert ru and rr and rn and update['zx'].tobytes()==ux.tobytes() and update['shadow_primal'].tobytes()==ux.tobytes()
 assert np.all(projection[orthant][reflection[orthant]<0]==0) and projection[orthant][reflection[orthant]>=0].tobytes()==reflection[orthant][reflection[orthant]>=0].tobytes()
 branch_failures=0;projection_relation_failures=0;norm_failures=0;defects=0
 for (st,wi),branch in zip(cones,pr['branches']):
  raw=reflection[st:st+wi];out=projection[st:st+wi];norm=branch['norm'];h=frac(raw[0]);n=frac(norm);tail=sum(frac(x)**2 for x in raw[1:])
  # Norm rounding is checked by an exact squared relation on SAVED inputs.
  allowance=Q(64,2**53)*(tail+n*n)+Q(64,2**1074)
  if abs(n*n-tail)>allowance:norm_failures+=1
  if branch['branch']=='inside':
   branch_failures+=int(h<0 or tail>h*h)
   projection_relation_failures+=int(out.tobytes()!=raw.tobytes())
  elif branch['branch']=='polar':
   branch_failures+=int(h>0 or tail>h*h)
   projection_relation_failures+=int(np.any(out!=0))
  elif branch['branch']=='middle':
   branch_failures+=int(n<=0 or n<=h or n<=-h)
   projection_relation_failures+=int(not near(out[0],(n+h)/2))
   for k in range(1,wi):projection_relation_failures+=int(not near(out[k],frac(out[0])*frac(raw[k])/n,64))
  else:raise ValueError('unknown recorded whole-cone branch')
  oh=frac(out[0]);defects+=int(oh<0 or oh*oh<sum(frac(x)**2 for x in out[1:]))
 record=dict(step=step,ALL_row_relations=5*NROW,ALL_free_relations=NFREE,ALL_whole_cones=len(cones),all_finite=all(np.isfinite(x).all() for x in [ux,t,ax,uy,reflection,projection,update['zy']]),
  norm_squared_relation_failures=norm_failures,exact_branch_condition_failures=branch_failures,projection_relation_failures=projection_relation_failures,
  exact_dyadic_projected_SOC_defects=defects,maximum_saved_state_identity_error=max(e1,e2,e3),numerical_outer_NOT_candidate_tolerance=True)
 seal_return(D,f'NEW-SAVED-step-{step:02d}-independent-binding-RETURN',record)
 assert record['all_finite'] and norm_failures==0 and projection_relation_failures==0
 total_row_identities+=5*NROW;total_branch_failures+=branch_failures;total_projected_SOC_defects+=defects;summary.append(record)
 zx,zy=update['zx'],update['zy'];del t,ax,uy,reflection,projection,pr,update,ux
terminal=get('NEW-terminal16-hard-affine-residual-RETURN');decision=get('NEW-terminal16-hard-affine-decision-RETURN');F=terminal['F'];full=terminal['unqualified_full_shadow'];final_u=get('NEW-update-16-ALL-free-linear-resolvent-RETURN')['value']
assert full[PHASE]==0 and np.r_[full[:PHASE],full[PHASE+1:]].tobytes()==final_u.tobytes()
lastax=get('NEW-update-16-ALL4810140-action-RETURN')['value'];ok,last_error=relation(F,[lastax,b],[1,-1]);assert ok
cone_margins=[];exact_cone_defects=0
for st,wi in cones:
 v=-F[st:st+wi];h=frac(v[0]);tail=sum(frac(x)**2 for x in v[1:]);exact_cone_defects+=int(h<0 or h*h<tail)
 # Bind saved decision margins to independent scalar math.hypot, not any
 # old native/source/model operator or a cone projection call.
 import math
 cone_margins.append(float(v[0]-math.hypot(*(float(x) for x in v[1:]))))
cm=np.asarray(cone_margins);outer=64*2.**-53*(1+np.abs(cm));assert np.all(np.abs(cm-decision['all1245_cone_margins'])<=outer)
groups=dict(object_nonnegative=int(np.count_nonzero((F[:NOBJ]>0)&orthant[:NOBJ])),robot_linear=int(np.count_nonzero(F[NOBJ:NOBJ+NROB]>0)),
 source_SOC_factors=int(sum(cm[1051:]<0)),object_SOC=int(sum(cm[:1051]<0)),strict_depth=int(np.count_nonzero(F[NOBJ+NROB+582:NOBJ+NROB+776]>=0)),
 halfdomain=int(np.count_nonzero(F[NOBJ+NROB+776:NOBJ+NROB+852]>0)),ideal_geometric_pair_witness_rows=int(np.count_nonzero(F[NOBJ+NROB+852:]>0)))
assert int(np.count_nonzero(F[orthant]>0))==decision['violated_affine_rows'] and int(sum(cm<0))==decision['violated_cones'] and decision['exact_zero_phase']
seal_return(D,'NEW-terminal16-independent-hard-rejection-RETURN',dict(groups=groups,violated_affine_rows=decision['violated_affine_rows'],violated_cones=decision['violated_cones'],exact_dyadic_terminal_SOC_defects=exact_cone_defects,
 actual_phase=float(full[PHASE]),terminal_identity=True,terminal_state_identity_error=last_error,all_raw_projections_and_split_states_unchanged=True,
 maximum_linear_residual_over_conditional_outer=max_linear_ratio,ALL1245cones_EACH16updates=True,no_NEW_nonlinear_source_native_query=True,
 mixed_model_units_NOT_native_metres_or_physical_failure_counts=True))
log=(E/'worker.log').read_text();warnings=log.count('RuntimeWarning:')
save(D/'report.json',dict(passed=True,wall_seconds=time.monotonic()-start,updates=16,terminal_ONLY=16,typed_returns=typed_returns,ALL_saved_array_entries=array_entries,all_row_state_identities=total_row_identities,all_whole_cone_checks=16*1245,
 exact_branch_condition_failures_RETAINED=total_branch_failures,exact_dyadic_projected_SOC_defects_RETAINED=total_projected_SOC_defects,exact_dyadic_terminal_SOC_defects=exact_cone_defects,
 terminal_violated_affine_rows=decision['violated_affine_rows'],terminal_violated_cones=decision['violated_cones'],group_failures=groups,phase=0.,actual_logged_matmul_warnings_RETAINED=warnings,
 checked_returns_finite_warning_cause_future_reliability_UNPROVEN=True,no_method_Ax_adjoint_factor_solve_projector_source_geometry_native_or_motion_operator_replay=True,
 no_candidate_reference_trajectory_task_policy_AUC_qualification=True))
