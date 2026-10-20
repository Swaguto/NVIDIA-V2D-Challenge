"""NEW saved-return audit of actual products/steps/decisions; NO method replay.
Runtime warnings remain retained; finite matching products do not prove their cause.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict,Counter
import datetime,hashlib,json,math,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-primal-saved-products-independent-v1';E=S/'e037-primal-restoration-evaluation-v1';sys.path.insert(0,str(S/'compute'))
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
read=lambda q:json.loads(Path(q).read_text());sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest();start=time.monotonic();J=(O/'all-independent-saved-products-steps-and-terminal-certificates.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
report=read(E/'report.json');raw=read(E/'ONE-actual-primal-restoration-RETURN.json');assert report['actual_method_status']==raw['status']=='UNRESOLVED_iteration_cap'and raw['iterations']==80 and raw['exact_terminal_checks']==0
for n,d in report['all_outputs_sha256'].items():assert sha(E/n)==d
journal=E/'ONE-primal-restoration-all-iterates-trials-and-returns.jsonl';events=[json.loads(v)for v in journal.read_text().splitlines()];cfg=raw['configuration'];n=len(raw['actual_terminal_vector']);assert n==38;iterates={};constructors={};solves={};trust={};trials={};trialpoints={};decisions=[];warnings=(E/'worker.log').read_text();warning_count=warnings.count('RuntimeWarning:');hist=[]
for r in events:
 kind=r['kind']
 if kind=='AFTER_NEW_primal_residual_Jacobian_RETURN':iterates[r['iteration']]=r
 elif kind=='BEFORE_NEW_damped_linear_constructor':constructors[r['iteration']]=r
 elif kind=='AFTER_NEW_damped_linear_solve_RETURN':solves[r['iteration']]=r
 elif kind=='AFTER_NEW_algorithmic_trust_step_RETURN':trust[r['iteration']]=r
 elif kind=='BEFORE_NEW_primal_trial':trialpoints[(r['iteration'],r['backtrack'])]=r
 elif kind=='AFTER_NEW_primal_trial_evaluation_RETURN':trials[(r['iteration'],r['backtrack'])]=r
 elif kind=='AFTER_NEW_Armijo_acceptance_RETURN':decisions.append(r)
assert len(iterates)==len(constructors)==len(solves)==len(trust)==80 and len(decisions)==len(trials)==len(trialpoints)==611;lam=F(cfg['damping']);eps=F(1,1<<52);max_Herr=max_berr=max_step_residual=max_phi_error=F(0);minimum_SPD_margin=None;products=0;symmetry_failures=[];accepted=[]
for iteration in range(80):
 state=iterates[iteration];values=list(map(F,state['normalized_signed_values']));residual=list(map(F,state['normalized_merit_residual']));Jac=[[F(v)for v in row]for row in state['Jacobian']];active=state['active_rows'];assert active==[i for i,v in enumerate(residual)if v<0];exactH=defaultdict(F);Habs=defaultdict(F);exactb=[F(0)]*n;babs=[F(0)]*n
 for row in active:
  nz=[(k,v)for k,v in enumerate(Jac[row])if v]
  for a,v in nz:
   exactb[a]-=v*residual[row];babs[a]+=abs(v*residual[row])
   for b,w in nz:exactH[a,b]+=v*w;Habs[a,b]+=abs(v*w)
 for k in range(n):exactH[k,k]+=lam;Habs[k,k]+=abs(lam)
 H=[[F(v)for v in row]for row in constructors[iteration]['normal_matrix']];rhs=list(map(F,constructors[iteration]['rhs']));rawstep=list(map(F,solves[iteration]['actual_raw_step']));stored_residual=list(map(F,solves[iteration]['linear_residual']));rho_rows=[F(0)]*n;all_symmetric=True;Eentries=[]
 for a in range(n):
  for b in range(n):
   error=H[a][b]-exactH[a,b];bound=Habs[a,b]*eps*64*(len(active)+4);products+=1;max_Herr=max(max_Herr,abs(error));rho_rows[a]+=abs(error)
   if H[a][b]!=H[b][a]:all_symmetric=False
   if error:Eentries.append(dict(i=a,j=b,exact_error=c.rec(error),diagnostic_bound=c.rec(bound)))
   assert abs(error)<=bound
  error=rhs[a]-exactb[a];bound=babs[a]*eps*64*(len(active)+4);max_berr=max(max_berr,abs(error));assert abs(error)<=bound
 rho=max(rho_rows);margin=lam-rho;minimum_SPD_margin=margin if minimum_SPD_margin is None else min(minimum_SPD_margin,margin)
 if not all_symmetric:symmetry_failures.append(iteration)
 assert all_symmetric and margin>0
 # Exact quadratic-form lower certificate: H=J_active^T J_active+lambdaI+E, ||E||2<=max rowabs(E)<lambda.
 exact_linear=[sum(H[a][b]*rawstep[b]for b in range(n))-rhs[a]for a in range(n)];observed=[abs(v-w)for v,w in zip(stored_residual,exact_linear)];max_step_residual=max(max_step_residual,max(map(abs,exact_linear)));linear_bounds=[(sum(abs(H[a][b]*rawstep[b])for b in range(n))+abs(rhs[a]))*eps*64*(n+4)for a in range(n)];assert all(v<=b for v,b in zip(observed,linear_bounds));phi=sum(v*v for v in residual)/2;phi_error=abs(F(state['merit'])-phi);max_phi_error=max(max_phi_error,phi_error);assert phi_error<=phi*eps*64*(len(residual)+4)
 T=trust[iteration];assert list(map(F,T['actual_raw_step_PRESERVED']))==rawstep;factor=F(T['trust_scale']);step=list(map(F,T['proposed_search_step']));cap=F(cfg['trust_step_Linf_halfangle']);assert 0<factor<=1
 for a in range(n):assert abs(step[a]-rawstep[a]*factor)<=abs(rawstep[a]*factor)*eps*4
 assert max(map(abs,step))<=F(math.nextafter(cfg['trust_step_Linf_halfangle'],math.inf));gradient=[-v for v in exactb];direction=sum(v*w for v,w in zip(gradient,step));slope=F(T['merit_directional_derivative']);slope_bound=sum(abs(v*w)for v,w in zip(gradient,step))*eps*64*(len(active)+n+4)+sum(abs((exactb[a]-rhs[a])*step[a])for a in range(n));assert abs(slope-direction)<=slope_bound and slope<0
 emit(dict(kind='AFTER_NEW_independent_actual_product_step_certificate_RETURN',iteration=iteration,active_rows=active,normal_matrix_exact_errors=Eentries,normal_error_spectral_rowbound=c.rec(rho),positive_quadratic_form_margin=c.rec(margin),actual_H_exactly_symmetric=True,actual_rhs_errors=[c.rec(rhs[a]-exactb[a])for a in range(n)],raw_linear_system_residual=[c.rec(v)for v in exact_linear],stored_linear_residual_error=[c.rec(v)for v in observed],actual_merit_error=c.rec(phi_error),actual_direction_exact=c.rec(direction),saved_direction=slope.__float__(),no_product_or_step_replaced=True))
 hist.append(dict(iteration=iteration,merit=state['merit'],minimum_normalized_signed_value=min(state['normalized_signed_values']),active_count=len(active)))
# Every saved decision is an ordinary comparison of ACTUAL returned merits/thresholds.
by_iteration=defaultdict(list)
for d in decisions:
 key=(d['iteration'],d['backtrack']);trial=trials[key];proposal=trialpoints[key];assert d['actual_merit']==trial['merit']and bool(d['accepted'])==(d['actual_merit']<=d['Armijo_threshold']);by_iteration[d['iteration']].append(d);x=list(map(F,iterates[d['iteration']]['actual_vector']));step=list(map(F,trust[d['iteration']]['proposed_search_step']));alpha=F(proposal['alpha']);actual=list(map(F,proposal['proposed_vector']));assert alpha==F(1,1<<d['backtrack'])
 for a in range(n):assert abs(actual[a]-(x[a]+alpha*step[a]))<=(abs(x[a])+abs(alpha*step[a]))*eps*4
 emit(dict(kind='AFTER_NEW_independent_saved_trial_decision_RETURN',iteration=d['iteration'],backtrack=d['backtrack'],accepted=d['accepted'],actual_merit=d['actual_merit'],actual_threshold=d['Armijo_threshold'],proposal_NOT_clipped=True))
for iteration,ds in by_iteration.items():
 chosen=[d for d in ds if d['accepted']];assert len(chosen)==1 and chosen[0]is ds[-1];d=chosen[0];point=trialpoints[iteration,d['backtrack']]['proposed_vector'];target=iterates[iteration+1]['actual_vector']if iteration<79 else raw['actual_terminal_vector'];assert point==target;accepted.append(d)
assert len(accepted)==80;rows=read(S/'e037-surrounding-clock-model-v1/exact-query-constraint-records.json');last=iterates[79];violating=[dict(row=i,id=rows[i]['id'],normalized_signed_value=v,source_or_time_NOT_physical_force=True)for i,v in enumerate(last['normalized_signed_values'])if v<0];assert len(violating)==6;save('saved-terminal-violations-and-merit-history.json',dict(history=hist,violating_rows=violating,actual_status=raw['status'],all_terminal_and_intermediate_vectors_REJECTED=True,no_best_point_or_failed_return_promoted=True));J.close();out=dict(schema='v2d.primal-saved-products-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_saved_evidence_checks_passed=True,no_primal_method_polynomial_derivative_linear_solve_or_old_audit_replayed=True,saved_iterates_checked=80,saved_trial_acceptance_decisions_checked=611,saved_normal_matrix_entry_products_checked=products,all80_actual_rounded_normal_matrices_exactly_symmetric_and_certified_positive_quadratic_form=True,minimum_positive_quadratic_form_margin=c.rec(minimum_SPD_margin),maximum_normal_matrix_error=c.rec(max_Herr),maximum_rhs_error=c.rec(max_berr),maximum_actual_raw_linear_system_residual=c.rec(max_step_residual),maximum_merit_product_error=c.rec(max_phi_error),RuntimeWarning_occurrences_RETAINED=warning_count,warning_cause_UNPROVEN=True,finite_product_agreement_NOT_future_runtime_proof=True,actual_terminal_method_status=raw['status'],actual_terminal_model_violations=violating,merit_history_first_last=[hist[0],hist[-1]],NO_candidate_or_family_exclusion_or_trajectory_qualification=True,new_method_runs=0,new_linear_solves=0,new_native_position_checks=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:(v.get('approx')if isinstance(v,dict)and'approx'in v else v)for k,v in out.items()if k not in['all_outputs_sha256']}))
