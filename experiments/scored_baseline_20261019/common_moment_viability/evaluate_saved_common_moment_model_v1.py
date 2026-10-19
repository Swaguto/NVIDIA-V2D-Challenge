"""One NEW declared common-marginal phase-I LP, never a physical pose selector."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-common-moment-model-v1';sys.path.insert(0,str(S/'compute'))
import bernstein_common_moments_v1 as m
import correlated_bezier_clipping_v1 as b
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-common-moment-factor-model-and-solver-returns.jsonl').open('w')
contract=read(S/'e037-common-moment-contract-v1/report.json');assert contract['all_controls_passed']and contract['helper_sha256']==sha(S/'compute/bernstein_common_moments_v1.py')
I=S/'e037-source-time-clipping-binding-v1/correlated-source-time-clipping-inputs.json';P=S/'e037-correlated-source-time-clipping-v1/all-correlated-domain-proposals-coefficients-clips-and-outcomes.jsonl';D=P.parent/'correlated-domain-contraction-result.json'
assert sha(I)==read(I.parent/'report.json')['all_outputs_sha256'][I.name]and sha(P)==read(P.parent/'report.json')['all_outputs_sha256'][P.name];inputs=read(I);domains={k:[b.dec(v)for v in r]for k,r in read(D)['final_domains'].items()};last={}
for row in map(json.loads,P.open()):
    if row['kind']=='AFTER_NEW_correlated_factor_contraction':last[row['factor']]=row
factors=[];representation=[]
for old in inputs['factors']:
    state=last[old['id']];bounds=[[b.dec(v)for v in r]for r in state['grid_bounds']];target=[domains[v]for v in old['variables']];ranges=[((a-c)/(d-c),(z-c)/(d-c))for(a,z),(c,d)in zip(target,bounds,strict=True)]
    g.journal(j,dict(kind='BEFORE_NEW_common_degree_factor_representation',factor=old['id'],target_bounds=[[b.rec(v)for v in r]for r in target],saved_last_state=state))
    grid=b.lower_round(b.restrict(b.decode(state['lower_grid']),old['degree'],ranges));elevated=m.elevate(grid,old['degree']);factor=dict(id=old['id'],variables=old['variables'],grid=elevated);factors.append(factor)
    record=dict(id=old['id'],variables=old['variables'],old_degree=old['degree'],target_bounds=[[b.rec(v)for v in r]for r in target],degree4_grid=b.encode(elevated),pre_elevation_grid=b.encode(grid))
    g.journal(j,dict(kind='AFTER_NEW_common_degree_factor_representation',record=record));representation.append(record)
(O/'common-degree-factor-representations.json').write_text(json.dumps(g.plain(dict(factors=representation,domains={k:[b.rec(v)for v in r]for k,r in domains.items()},source_clipping_journal_sha256=sha(P))),allow_nan=False,indent=2)+'\n')
g.journal(j,dict(kind='BEFORE_NEW_common_marginal_model',variables=list(domains),factor_count=len(factors),square_anchor_set=['0','1/4','1/2','3/4','1'],polynomial_degree=4,one_episode_phase_one_LP=True,LP_time_limit_seconds=120))
model=m.build(list(domains),factors);model.update(source_input_sha256=sha(I),saved_clipping_journal_sha256=sha(P),final_domain_file_sha256=sha(D),original_root_source_finger_tolerances_unchanged=True,
    scope='Necessary lifted model within inherited region2/fixed unqualified wrist; all source interiors/unknown-event intersections retained; source depth lower bound reused, not inferred measured truth')
(O/'common-moment-LP-model.json').write_text(json.dumps(g.plain(model),allow_nan=False,indent=2)+'\n');g.journal(j,dict(kind='AFTER_NEW_common_marginal_model_RETURN',columns=len(model['columns']),equalities=len(model['equalities']),inequalities=len(model['inequalities']),model_sha256=sha(O/'common-moment-LP-model.json')))
assert len(model['columns'])<=1200 and len(model['inequalities'])<=6000 and len(model['equalities'])<=600
g.journal(j,dict(kind='BEFORE_NEW_episode_phase_one_LP',model_sha256=sha(O/'common-moment-LP-model.json'),time_limit_seconds=120,no_solver_restart_or_gate_changes=True))
result=m.solve_phase_one(model,time_limit=120)
(O/'phase-one-solver-RETURN.json').write_text(json.dumps(g.plain(result),allow_nan=False,indent=2)+'\n');g.journal(j,dict(kind='AFTER_NEW_episode_phase_one_solver_RETURN',return_sha256=sha(O/'phase-one-solver-RETURN.json'),status=result['status'],success=result['success'],objective=result['objective']))
certificate=m.finite_box_dual(model,result);(O/'finite-box-dual-certificate.json').write_text(json.dumps(g.plain(certificate),allow_nan=False,indent=2)+'\n');g.journal(j,dict(kind='AFTER_NEW_exact_dual_RETURN',classification=certificate['classification'],strict_gap=certificate.get('strict_gap')))
primal_check=None
if result['primal']is not None:
    x=[F(v)for v in result['primal'][:-1]];eq=[sum(m.dec(v['value'])*x[v['column']]for v in r['coefficients'])-m.dec(r['bound'])for r in model['equalities']];ineq=[sum(m.dec(v['value'])*x[v['column']]for v in r['coefficients'])-m.dec(r['bound'])for r in model['inequalities']]
    primal_check=dict(maximum_exact_binary_primal_equality_residual=m.rec(max(map(abs,eq))),maximum_exact_binary_primal_inequality_residual=m.rec(max(ineq)),moments_in_unit_box=all(0<=v<=1 for v in x),LP_diagnostic_residual_NOT_original_candidate_gate=True)
    g.journal(j,dict(kind='AFTER_NEW_primal_residual_diagnostic',record=primal_check))
classification='conditional_saved_polynomial_model_EXCLUDED'if certificate['classification']=='exact_saved_LP_model_INFEASIBILITY_certificate'else('numerical_relaxation_feasible_NOT_a_pose'if result['success']and result['objective']<=1e-7 else'UNRESOLVED_relaxation_or_dual')
j.close();out=dict(schema='v2d.common-moment-model-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/bernstein_common_moments_v1.py'),episode_phase_one_LPs=1,variables=len(model['columns']),equality_rows=len(model['equalities']),inequality_rows=len(model['inequalities']),phase_one_status=result['status'],phase_one_objective=result['objective'],classification=classification,dual_classification=certificate['classification'],exact_dual_gap=certificate.get('strict_gap'),primal_diagnostic=primal_check,
    no_pose_barycenter_or_reference_selected=True,relaxed_feasibility_NOT_source_truth_native_grasp_or_path=True,no_old_LP_clipping_source_graph_or_optimizer_repeated=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,current_verified_reference_and_middle_component_unchanged=True,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
