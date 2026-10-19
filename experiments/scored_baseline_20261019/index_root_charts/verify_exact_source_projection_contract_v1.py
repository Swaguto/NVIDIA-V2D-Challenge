"""NEW independent exact projection/root contracts, no episode/model work."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-index-source-projection-contract-v1';sys.path.insert(0,str(S/'compute'))
import exact_source_projection_v1 as x
import thumb_conic_geometry_v1 as g
start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'all-exact-projection-contract-proposals-and-returns.jsonl').open('w');controls=[]
def require(name,passed,evidence):
    g.journal(j,dict(kind='AFTER_new_exact_projection_control',name=name,passed=bool(passed),evidence=evidence));controls.append(dict(name=name,passed=bool(passed)));assert passed,name
def roots(p,a,b,**kwargs):
    result=x.isolate(p,[a,b],**kwargs)
    assert len(result['roots'])+sum(v['open_real_root_count'] for v in result['unresolved'])==result['closed_interval_distinct_real_root_count']
    return result
g.journal(j,dict(kind='BEFORE_new_exact_projection_controls',families=['quadratic tangent','identically repeated','degree drop','multiple roots','complex roots','endpoint roots','zero width','continuous branch','determinant identity','constant derivative','isolation budget']))
p=x.bivariate([[0,-1],[],[1]]);r=x.resultant(p,x.derivative_u(p))
require('quadratic_tangency_resultant',r['coefficients']==[0,-1],dict(coefficients_hex=[hex(v) for v in r['coefficients']]))
p=x.bivariate([[0,0,1],[0,-2],[1]]);r=x.resultant(p,x.derivative_u(p))
require('identically_repeated_curve_is_unresolved',r['classification']=='identically_zero_resultant',dict(classification=r['classification']))
p=x.bivariate([[1],[1],[0,1]]);r=x.resultant(p,x.derivative_u(p));isolation=roots(r['coefficients'],x.F(-1,10),x.F(1,2))
require('degree_drop_event_is_not_removed',r['coefficients']==[0,-1,4] and isolation['closed_interval_distinct_real_root_count']==2,dict(resultant_hex=[hex(v) for v in r['coefficients']],isolation=isolation))
p=x.mul(x.mul([-x.F(1,5),1],[-x.F(1,5),1]),[x.F(1,10),1]);rr=roots(p,-1,1)
require('multiple_root_and_nonconstant_gcd_retained',rr['closed_interval_distinct_real_root_count']==2 and rr['nonconstant_gcd_repeated_factor_present'],rr)
rr=roots([1,0,1],-1,1);complex_roots=x.numerical_complex_roots([1,0,1])
require('complex_roots_not_filtered_as_real',rr['closed_interval_distinct_real_root_count']==0 and len(complex_roots['roots'])==2 and all(abs(v['imaginary'])>.9 for v in complex_roots['roots']),dict(exact=rr,numerical=complex_roots))
p=x.mul(x.mul([0,1],[-x.F(1,2),1]),[-1,1]);rr=roots(p,0,1)
require('open_closed_and_midpoint_endpoint_roots',rr['closed_interval_distinct_real_root_count']==3 and len(rr['roots'])==3,rr)
rr=roots(p,x.F(1,2),x.F(1,2))
require('fixed_parameter_root_interval',rr['closed_interval_distinct_real_root_count']==1 and len(rr['roots'])==1,rr)
p=x.bivariate([[0,-1],[],[1]]);u=x.specialize_v(p,x.F(1,4));rr=roots(u,0,2)
require('known_continuous_source_branch',rr['closed_interval_distinct_real_root_count']==1 and x.evaluate(u,x.F(1,4))<0 and x.evaluate(u,x.F(5,4))>0,rr)

def gaussian_det(matrix):
    a=[row.copy() for row in matrix];value=x.F(1)
    for k in range(len(a)):
        pivot=next((r for r in range(k,len(a)) if a[r][k]),None)
        if pivot is None:return x.F(0)
        if pivot!=k:a[k],a[pivot]=a[pivot],a[k];value=-value
        diag=a[k][k];value*=diag
        for r in range(k+1,len(a)):
            factor=a[r][k]/diag
            for col in range(k+1,len(a)):a[r][col]-=factor*a[k][col]
            a[r][k]=x.F(0)
    return value
p=x.bivariate([[1],[0,1],[],[1]]);q=x.derivative_u(p);r=x.resultant(p,q);matrix=x.sylvester(p,q);checks=[]
for v in [x.F(-2),x.F(-1,2),x.F(0),x.F(1,3),x.F(2)]:
    det=gaussian_det([[x.evaluate(poly,v) for poly in row] for row in matrix]);expected=x.evaluate([27,0,0,4],v);checks.append(dict(parameter=x.fraction_record(v),det=x.fraction_record(det),expected=x.fraction_record(expected),passed=det==expected))
require('independent_fraction_Gaussian_vs_polynomial_determinant',r['coefficients']==[27,0,0,4] and all(v['passed'] for v in checks),checks)
p=x.bivariate([[1]]);r=x.resultant(p,x.derivative_u(p))
require('zero_derivative_domain_explicit',r['classification']=='identically_zero_input_or_derivative',dict(classification=r['classification']))
rr=roots([0,-x.F(3,10),0,1],-1,1,node_cap=1)
require('root_isolation_budget_keeps_unknown_intervals',rr['classification']=='unresolved_exact_root_isolation_budget' and len(rr['unresolved'])==2,rr)
j.close();out=dict(schema='v2d.exact-source-projection-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/exact_source_projection_v1.py'),new_analytic_controls=controls,all_controls_passed=all(v['passed'] for v in controls),no_episode_source_native_physics_or_GPU_work=True,exact_arithmetic_scope='saved binary coefficient model, not upstream camera/source accuracy',all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
