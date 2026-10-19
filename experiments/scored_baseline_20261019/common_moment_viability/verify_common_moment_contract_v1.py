"""New deterministic embedding, false-positive and dual controls before episode LP."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from math import comb,prod
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-common-moment-contract-v1';sys.path.insert(0,str(S/'compute'))
import bernstein_common_moments_v1 as m
import correlated_bezier_clipping_v1 as b
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();j=(O/'all-common-moment-contracts-and-solver-returns.jsonl').open('w');controls=[]
def check(name,work):
    g.journal(j,dict(kind='BEFORE_NEW_common_moment_control',name=name));v=work();g.journal(j,dict(kind='AFTER_NEW_common_moment_control',name=name,result=v));controls.append(dict(name=name,passed=v['passed']));assert v['passed']
def evaluate(grid,D,t):return sum(v*prod(F(comb(D,i))*x**i*(1-x)**(D-i)for i,x in zip(k,t,strict=True))for k,v in grid.items())
def elevate_control():
    n=0
    for degree in[(1,1),(1,2,1),(2,3)]:
        grid={k:F(sum((i+1)*v for i,v in enumerate(k))-2,7)for k in product(*(range(d+1)for d in degree))};e=m.elevate(grid,degree)
        for t in product([F(1,5),F(3,4)],repeat=len(degree)):
            actual=sum(v*prod(F(comb(d,i))*x**i*(1-x)**(d-i)for d,i,x in zip(degree,k,t,strict=True))for k,v in grid.items());assert evaluate(e,4,t)==actual;n+=1
    return dict(passed=True,exact_values=n)
check('multiaxis_degree_elevation_identity',elevate_control)
def embedding():
    definitions=[dict(id='ab',variables=['a','b'],power={(0,0):F(-4,5),(1,0):F(1),(0,1):F(1)}),dict(id='bc',variables=['b','c'],power={(0,0):F(1,10),(1,0):F(1),(0,1):F(-1)}),dict(id='abc',variables=['a','b','c'],power={(0,0,0):F(-1,5),(1,0,0):F(1),(0,1,0):F(-2),(0,0,1):F(1)})]
    factors=[dict(id=f['id'],variables=f['variables'],grid=b.power_to_bernstein(f['power'],[(F(0),F(1))]*len(f['variables']),(4,)*len(f['variables'])))for f in definitions];model=m.build(['a','b','c'],factors);t=dict(a=F(1,5),b=F(2,5),c=F(7,10));tables={}
    for col in model['columns']:
        scope=tuple(col['scope'])
        if scope not in tables:tables[scope]=m.deterministic_weights([t[v]for v in scope])
    x=[tables[tuple(v['scope'])][tuple(v['index'])]for v in model['columns']];eq=[sum(m.dec(c['value'])*x[c['column']]for c in row['coefficients'])-m.dec(row['bound'])for row in model['equalities']];ineq=[sum(m.dec(c['value'])*x[c['column']]for c in row['coefficients'])-m.dec(row['bound'])for row in model['inequalities']]
    return dict(passed=not any(eq)and max(ineq)<=0,equality_rows=len(eq),inequality_rows=len(ineq),maximum_inequality=m.rec(max(ineq)))
check('deterministic_source_time_embedding_shared_pair_marginals',embedding)
def squares():
    checks=0
    for p in m.positive_tests():
        for x in[F(i,17)for i in range(18)]:
            direct=sum(v*x**k for k,v in p['power'].items());value=sum(v*comb(4,k)*x**k*(1-x)**(4-k)for k,v in p['Bernstein'].items());assert direct==value and value>=0;checks+=1
    return dict(passed=True,positive_polynomials=9,exact_values=checks)
check('conditioned_positive_polynomial_identities',squares)
def pseudo_cycle():
    factors=[]
    for a,b0 in[('a','b'),('b','c'),('a','c')]:
        for sign in[1,-1]:factors.append(dict(id=a+b0+str(sign),variables=[a,b0],grid={(i,k):sign*(F(i+k,4)-1)for i,k in product(range(5),repeat=2)}))
    model=m.build(['a','b','c'],factors);x=[]
    for v in model['columns']:
        k=v['index'];x.append(F(1,2)if(len(k)==1 and k[0]in[0,4])or(len(k)==2 and k in[[0,4],[4,0]])else F(0))
    eq=all(sum(m.dec(c['value'])*x[c['column']]for c in row['coefficients'])==m.dec(row['bound'])for row in model['equalities']);ineq=all(sum(m.dec(c['value'])*x[c['column']]for c in row['coefficients'])<=m.dec(row['bound'])for row in model['inequalities']);no_binary_assignment=all(any(t[i]+t[k]!=1 for i,k in[(0,1),(1,2),(0,2)])for t in product([0,1],repeat=3))
    return dict(passed=eq and ineq and no_binary_assignment,relaxed_tables_consistent=True,binary_marginals_at_endpoints_only=True,no_deterministic_binary_pose=True,feasible_relaxation_NOT_constructive=True)
check('odd_cycle_relaxed_table_false_positive_retained',pseudo_cycle)
def scalar_model(feasible):
    def row(name,a,b0):return dict(name=name,coefficients=[dict(column=0,value=m.rec(F(a)))],bound=m.rec(F(b0)))
    return dict(columns=[dict(scope=['x'],index=[0])],equalities=[],inequalities=[row('upper',1,F(2,5)if feasible else F(1,5)),row('lower',-1,0 if feasible else F(-4,5))])
def lp_control(feasible):
    model=scalar_model(feasible);g.journal(j,dict(kind='BEFORE_NEW_tiny_phase_one_control',feasible=feasible,model=model));r=m.solve_phase_one(model,time_limit=10);g.journal(j,dict(kind='AFTER_NEW_tiny_phase_one_solver_RETURN',feasible=feasible,result=r));certificate=m.finite_box_dual(model,r)
    passed=r['success']and((r['objective']==0)if feasible else(certificate['classification']=='exact_saved_LP_model_INFEASIBILITY_certificate'))
    return dict(passed=passed,solver_return=r,dual_certificate=certificate)
check('new_phase_one_feasible_control',lambda:lp_control(True))
check('new_phase_one_infeasible_dual_control',lambda:lp_control(False))
def residual_controls():
    model=scalar_model(False);records=[]
    for alpha in[[1.,.5],[.5,1.]]:
        certificate=m.finite_box_dual(model,dict(success=True,ineqlin=dict(marginals=[-v for v in alpha]),eqlin=dict(marginals=[])));records.append(certificate)
    return dict(passed=all(v['classification']=='exact_saved_LP_model_INFEASIBILITY_certificate'for v in records),certificates=records)
check('finite_box_residual_correction_both_signs',residual_controls)
def inputs():
    errors=[]
    for call in[lambda:m.build(['a','a'],[]),lambda:m.build(['a'],[],D=3),lambda:m.deterministic_weights([F(-1)]),lambda:m.elevate({(0,):F(1)},(5,))]:
        try:call();errors.append(None)
        except ValueError as e:errors.append(str(e))
    return dict(passed=all(errors),errors=errors)
check('strict_research_input_contracts',inputs)
j.close();out=dict(schema='v2d.common-moment-contract-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/bernstein_common_moments_v1.py'),new_analytic_controls=controls,all_controls_passed=all(v['passed']for v in controls),tiny_NEW_analytic_control_LPs=2,episode_LPs=0,local_marginal_feasibility_NOT_a_pose=True,no_old_source_clipping_LP_graph_or_native_job_repeated=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
