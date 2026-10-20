"""NEW bounded primal inequality restoration over unchanged SAVED polynomials.
Conditioning/merit/damping are algorithmic, not source confidence or gates.
Only exact unchanged-model feasibility can return a proposed-success vector.
"""
from fractions import Fraction as F
import math,time
import numpy as np
import conditional_moment_cones_v1 as c

CONFIG=dict(maximum_iterations=80,wall_limit_seconds=120.,damping=1e-6,
            trust_step_Linf_halfangle=.02,maximum_Armijo_backtracks=12,
            Armijo_coefficient=1e-4,interior_merit_target=1e-10,
            normalized_exact_check_TRIGGER_only=1e-8)

class PolynomialSystem:
 def __init__(self,records,names,domains,emit):
  if not records or not names or len(set(names))!=len(names):raise ValueError('explicit nonempty model and unique global variables required')
  self.names=list(names);self.rows=[];self.derivative_maps=[];self.scales=[];self.signs=[];self.strict=[];rowids=[];powers=[];coeff=[];drows=[];dcols=[];dpowers=[];dcoeff=[];self.dim=len(names)
  for i,r in enumerate(records):
   scope=r['scope']
   if len(set(scope))!=len(scope)or not set(scope).issubset(names)or r['sense']not in['>=','<=','>']:raise ValueError('invalid scope/sign')
   index=[names.index(n)for n in scope];terms=[];maps=[];bound=F(0)
   for term_index,item in enumerate(r['coefficients']):
    e=item['power'];v=c.dec(item['value'])
    if len(e)!=len(scope)or any(type(k)is not int or k<0 for k in e):raise ValueError('invalid saved power')
    if not v:raise ValueError('explicit zero coefficient is not a structural nonzero')
    f=float(v)
    if not math.isfinite(f)or f==0:raise ValueError('UNRESOLVED original nonzero float conversion')
    global_power=[0]*self.dim
    for j,k in zip(index,e):global_power[j]=k
    terms.append((tuple(e),v));rowids.append(i);powers.append(global_power);coeff.append(f);bound+=abs(v)*math.prod(max(abs(F(x))for x in domains[n])**k for n,k in zip(scope,e))
    for local,(column,k)in enumerate(zip(index,e)):
     if not k:continue
     derivative=global_power.copy();derivative[column]-=1;maps.append(dict(original_term_index=term_index,variable=scope[local],global_column=column,multiplicity=k,derivative_global_power=derivative));drows.append(i);dcols.append(column);dpowers.append(derivative);dcoeff.append(float(v*k))
   normalizer=bound if bound>0 else F(1);numeric=float(normalizer)
   if not math.isfinite(numeric)or numeric<=0:raise ValueError('UNRESOLVED nonpositive/nonfinite static normalization')
   if F(numeric)<normalizer:numeric=math.nextafter(numeric,math.inf)
   sign=-1 if r['sense']=='<='else 1;strict=r['sense']=='>';row=dict(id=r['id'],scope=scope,index=index,terms=terms,sign=sign,strict=strict)
   self.rows.append(row);self.derivative_maps.append(maps);self.scales.append(numeric);self.signs.append(sign);self.strict.append(strict)
   emit(dict(kind='AFTER_NEW_primal_row_constructor_RETURN',row=i,id=r['id'],original_scope=scope,global_columns=index,sense=r['sense'],strict=strict,static_exact_L1_bound=c.rec(bound),static_normalizer=c.rec(normalizer),numeric_outward_normalizer=numeric,derivative_index_maps=maps,original_coefficient_records_UNCHANGED=True,normalizer_bound_on_SAVED_domain_ONLY=True,conditioning_outside_domain_NOT_feasibility=True))
  self.rowids=np.array(rowids,int);self.powers=np.array(powers,int).reshape(-1,self.dim);self.coeff=np.array(coeff,float);self.drows=np.array(drows,int);self.dcols=np.array(dcols,int);self.dpowers=np.array(dpowers,int).reshape(-1,self.dim);self.dcoeff=np.array(dcoeff,float);self.scales=np.array(self.scales);self.signs=np.array(self.signs,float);self.strict=np.array(self.strict,bool);self.count=len(records)
  emit(dict(kind='AFTER_NEW_primal_system_constructor_RETURN',rows=self.count,variables=self.dim,original_terms=len(coeff),derivative_index_terms=len(dcoeff),no_source_polynomial_or_model_authority_regenerated=True))
 def evaluate(self,x,jacobian=False):
  x=np.asarray(x,float)
  if x.shape!=(self.dim,)or not np.isfinite(x).all():raise ValueError('finite full variable vector required')
  with np.errstate(over='raise',invalid='raise',divide='raise'):
   terms=self.coeff*np.prod(np.power(x[None,:],self.powers),axis=1);value=np.zeros(self.count);np.add.at(value,self.rowids,terms);value=value*self.signs/self.scales
   if not jacobian:return value
   derivatives=self.dcoeff*np.prod(np.power(x[None,:],self.dpowers),axis=1);J=np.zeros((self.count,self.dim));np.add.at(J,(self.drows,self.dcols),derivatives);J=J*self.signs[:,None]/self.scales[:,None];return value,J
 def exact_check(self,x,emit):
  point=[F(float(v))for v in x];result=[]
  for i,r in enumerate(self.rows):
   value=F(0)
   for e,v in r['terms']:
    for j,k in zip(r['index'],e):v*=point[j]**k
    value+=v
   value*=r['sign'];passed=value>0 if r['strict']else value>=0;record=dict(row=i,id=r['id'],actual_binary_point_signed_polynomial=c.rec(value),strict=r['strict'],passed=bool(passed));emit(dict(kind='AFTER_NEW_exact_terminal_candidate_row_RETURN',RETURN=record));result.append(record)
  out=dict(all_exact_model_inequalities_passed=all(v['passed']for v in result),failed_rows=[v['row']for v in result if not v['passed']],strict_depth_NEVER_accepted_at_zero=True,original_constraints_NOT_merit_threshold=True);emit(dict(kind='AFTER_NEW_exact_terminal_candidate_CHECK_RETURN',RETURN=out));return out

def merit(value):
 residual=np.minimum(np.asarray(value)-CONFIG['interior_merit_target'],0.);return .5*float(residual@residual),residual

def restore(system,initial,emit):
 start=time.monotonic();x=np.array(initial,float,copy=True);iterations=0;exactchecks=0;accepted=0;linear_calls=0;trials=0;status=None;initial_copy=x.copy()
 if x.shape!=(system.dim,)or not np.isfinite(x).all():raise ValueError('finite full initial vector')
 emit(dict(kind='BEFORE_ONE_NEW_primal_restoration',configuration=CONFIG,initial_vector=x,initial_NOT_clipped_or_declared_feasible=True,all_variables_continuous=True,no_best_intermediate_selection=True))
 def check(point,value,why):
  nonlocal exactchecks
  if np.min(value)<-CONFIG['normalized_exact_check_TRIGGER_only']:return False
  emit(dict(kind='BEFORE_NEW_exact_terminal_candidate_CHECK',reason=why,actual_vector=point,numerical_TRIGGER_NOT_acceptance=True));result=system.exact_check(point,emit);exactchecks+=1;return result['all_exact_model_inequalities_passed']
 for iteration in range(CONFIG['maximum_iterations']):
  iterations=iteration+1
  if time.monotonic()-start>=CONFIG['wall_limit_seconds']:status='UNRESOLVED_wall_cap';break
  emit(dict(kind='BEFORE_NEW_primal_iterate_evaluation',iteration=iteration,actual_vector=x))
  try:value,J=system.evaluate(x,True)
  except BaseException as e:emit(dict(kind='AFTER_NEW_primal_evaluation_EXCEPTION_RETURN',iteration=iteration,error=repr(e)));status='UNRESOLVED_nonfinite_or_invalid_evaluation';break
  phi,residual=merit(value);active=residual<0;emit(dict(kind='AFTER_NEW_primal_residual_Jacobian_RETURN',iteration=iteration,actual_vector=x,normalized_signed_values=value,normalized_merit_residual=residual,active_rows=np.flatnonzero(active),Jacobian=J,merit=phi))
  if not np.isfinite(np.r_[value,J.ravel(),phi]).all():status='UNRESOLVED_nonfinite_evaluation';break
  if check(x,value,'accepted_iterate'):status='PROPOSED_exact_saved_model_feasible';break
  gradient=J[active].T@residual[active];H=J[active].T@J[active]+CONFIG['damping']*np.eye(system.dim);rhs=-gradient;emit(dict(kind='BEFORE_NEW_damped_linear_constructor',iteration=iteration,normal_matrix=H,rhs=rhs,constant_algorithmic_damping=CONFIG['damping'],NOT_force_effort_or_source_confidence=True))
  try:raw_step=np.linalg.solve(H,rhs);linear_calls+=1;emit(dict(kind='AFTER_NEW_damped_linear_solve_RETURN',iteration=iteration,actual_raw_step=raw_step,linear_residual=H@raw_step-rhs))
  except BaseException as e:emit(dict(kind='AFTER_NEW_damped_linear_solve_EXCEPTION_RETURN',iteration=iteration,error=repr(e)));status='UNRESOLVED_linear_solve';break
  if not np.isfinite(raw_step).all():status='UNRESOLVED_nonfinite_linear_return';break
  maximum=float(abs(raw_step).max());factor=min(1.,CONFIG['trust_step_Linf_halfangle']/maximum)if maximum else 1.;step=raw_step*factor;slope=float(gradient@step);emit(dict(kind='AFTER_NEW_algorithmic_trust_step_RETURN',iteration=iteration,actual_raw_step_PRESERVED=raw_step,trust_scale=factor,proposed_search_step=step,merit_directional_derivative=slope,NOT_clipping_commands_or_model_boxes=True))
  if maximum==0 or not slope<0:status='UNRESOLVED_stationary_infeasible';break
  was_accepted=False
  for backtrack in range(CONFIG['maximum_Armijo_backtracks']):
   if time.monotonic()-start>=CONFIG['wall_limit_seconds']:status='UNRESOLVED_wall_cap';break
   alpha=2.**(-backtrack);trial=x+alpha*step;trials+=1;emit(dict(kind='BEFORE_NEW_primal_trial',iteration=iteration,backtrack=backtrack,alpha=alpha,proposed_vector=trial,no_box_or_return_clipping=True))
   try:vtrial=system.evaluate(trial);trialphi,trialres=merit(vtrial);emit(dict(kind='AFTER_NEW_primal_trial_evaluation_RETURN',iteration=iteration,backtrack=backtrack,normalized_signed_values=vtrial,normalized_merit_residual=trialres,merit=trialphi))
   except BaseException as e:emit(dict(kind='AFTER_NEW_primal_trial_EXCEPTION_RETURN',iteration=iteration,backtrack=backtrack,error=repr(e)));continue
   if not np.isfinite(np.r_[vtrial,trialphi]).all():emit(dict(kind='AFTER_NEW_primal_trial_REJECTED',reason='nonfinite',iteration=iteration,backtrack=backtrack));continue
   if check(trial,vtrial,'trial_before_merit_decision'):
    x=trial;status='PROPOSED_exact_saved_model_feasible';was_accepted=True;accepted+=1;emit(dict(kind='AFTER_NEW_primal_trial_EXACT_feasibility_RETURN',iteration=iteration,backtrack=backtrack,merit_did_NOT_redefine_feasibility=True));break
   threshold=phi+CONFIG['Armijo_coefficient']*alpha*slope;passed=trialphi<=threshold;emit(dict(kind='AFTER_NEW_Armijo_acceptance_RETURN',iteration=iteration,backtrack=backtrack,actual_merit=trialphi,Armijo_threshold=threshold,accepted=bool(passed)))
   if passed:x=trial;was_accepted=True;accepted+=1;break
  if status is not None:break
  if not was_accepted:status='UNRESOLVED_Armijo_budget';break
 if status is None:status='UNRESOLVED_iteration_cap'
 result=dict(status=status,actual_terminal_vector=x,initial_vector=initial_copy,iterations=iterations,accepted_steps=accepted,linear_solve_calls=linear_calls,trial_evaluations=trials,exact_terminal_checks=exactchecks,wall_seconds=time.monotonic()-start,configuration=CONFIG,actual_unsuccessful_terminal_return_REJECTED=status!='PROPOSED_exact_saved_model_feasible',NO_best_or_failed_intermediate_promoted=True,exact_saved_model_feasibility_NOT_trajectory_or_source_truth=True)
 emit(dict(kind='AFTER_ONE_NEW_primal_restoration_TERMINAL_RETURN',RETURN=result));return result
