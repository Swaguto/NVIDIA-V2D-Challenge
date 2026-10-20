"""Same contracted method; reuse SHA-bound preflight initial view/first matrices."""
from sequential_camera_conic_v1 import *

def construct_precomputed(system,initial_view,initial_problem,emit,private_site):
 start=time.monotonic();emit(dict(kind='BEFORE_ONE_NEW_sequential_camera_construction',configuration=CONFIG,initial_F_angles=system.x0,all_prior_failed_poses_UNUSED=True));x=system.x0.copy();view=initial_view;calls=0;checks=0;trials=0
 def end(status,iteration,extra=None):
  r=dict(status=status,major_iterations=iteration,subproblem_calls=calls,trial_evaluations=trials,exact_acceptance_checks=checks,actual_seconds=time.monotonic()-start,terminal_angles=x.copy(),terminal_merit=view['merit'],proposal_returned=status=='PROPOSED_actual_original_model_point',all_unsuccessful_terminal_and_intermediate_points_REJECTED=status!='PROPOSED_actual_original_model_point',extra=extra);emit(dict(kind='AFTER_ONE_NEW_sequential_camera_construction_RETURN',actual_RETURN=r));return r
 for iteration in range(CONFIG['major_iterations']):
  if time.monotonic()-start>=CONFIG['method_seconds']:return end('UNRESOLVED_method_time_cap',iteration)
  if view['merit']<=CONFIG['exact_check_trigger']:
   acceptance=system.exact_acceptance(view,emit,f'check-{iteration}');checks+=1
   if acceptance['all_original_model_checks_passed']:return end('PROPOSED_actual_original_model_point',iteration,acceptance)
  problem=initial_problem if iteration==0 else step_problem(system,view,emit,f'major-{iteration}');remaining=CONFIG['method_seconds']-(time.monotonic()-start)
  if remaining<=0:return end('UNRESOLVED_method_time_cap',iteration)
  result=solve_step(problem,emit,f'major-{iteration}',remaining,private_site);calls+=1
  if result['status']not in['Solved','AlmostSolved']:return end('UNRESOLVED_subproblem_status',iteration+1,dict(actual_status=result['status'],no_original_family_exclusion=True))
  actual=np.asarray(result['x'],dtype=float)
  if actual.shape!=(system.n+1,)or not np.isfinite(actual).all():return end('UNRESOLVED_nonfinite_step_RETURN',iteration+1)
  step=actual[:-1];rho=float(actual[-1]);predicted=view['merit']-rho;emit(dict(kind='AFTER_NEW_raw_predicted_reduction',iteration=iteration,actual_step=step,actual_rho=rho,predicted_reduction=predicted,raw_RETURN_NOT_clipped_or_repaired=True))
  if not predicted>0:return end('UNRESOLVED_no_predicted_descent',iteration+1)
  accepted=False
  for backtrack in range(CONFIG['backtracks']):
   alpha=2.**(-backtrack);trial=x+alpha*step;emit(dict(kind='BEFORE_NEW_unlinearized_Armijo_trial',iteration=iteration,backtrack=backtrack,alpha=alpha,proposed_angles=trial,raw_step=step))
   if float(abs(alpha*step).max(initial=0.))>CONFIG['max_step_rad']:emit(dict(kind='AFTER_NEW_unlinearized_Armijo_trial',iteration=iteration,backtrack=backtrack,accepted=False,reason='algorithm_trust_bound; rawstep_UNCHANGED'));continue
   try:trialview=system.snapshot(trial,emit,f'trial-{iteration}-{backtrack}')
   except (ValueError,FloatingPointError)as ex:emit(dict(kind='AFTER_NEW_unlinearized_Armijo_trial',iteration=iteration,backtrack=backtrack,accepted=False,reason=repr(ex)));continue
   trials+=1;bound=view['merit']-CONFIG['armijo']*alpha*predicted;accepted=trialview['merit']<=bound;emit(dict(kind='AFTER_NEW_unlinearized_Armijo_trial',iteration=iteration,backtrack=backtrack,actual_merit=trialview['merit'],original_merit=view['merit'],Armijo_bound=bound,accepted=accepted))
   if accepted:x=trial;view=trialview;break
  if not accepted:return end('UNRESOLVED_no_Armijo_step',iteration+1)
 if view['merit']<=CONFIG['exact_check_trigger']:
  acceptance=system.exact_acceptance(view,emit,'terminal-cap-check');checks+=1
  if acceptance['all_original_model_checks_passed']:return end('PROPOSED_actual_original_model_point',CONFIG['major_iterations'],acceptance)
 return end('UNRESOLVED_iteration_cap',CONFIG['major_iterations'])
