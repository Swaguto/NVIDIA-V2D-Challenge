"""NEW exact real query/RETURN contracts; synthetic values are not source poses."""
from pathlib import Path
from fractions import Fraction as F
import ast,datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-direct-real-contract-v1';sys.path.insert(0,str(S/'compute'))
import direct_real_feasibility_v1 as q
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'all-NEW-exact-real-contract-returns.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(name,v):
 with(O/name).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
R=read(S/'compute/direct-real-arithmetic-runtime-inspection-v1.json');exe=Path(R['executable']);assert sha(exe)==R['sha256']and R['version_stdout']=='Z3 version 4.15.4 - 64 bit';checks=[];calls=0

def check(name,proposal,worker):
 emit(dict(kind='BEFORE_NEW_real_encoding_contract',name=name,proposal=proposal));result=worker();emit(dict(kind='AFTER_NEW_real_encoding_contract_RETURN',name=name,RETURN=result));checks.append(dict(name=name,RETURN=result))

def CLI(name,names,constraints,expected,with_values=True):
 global calls
 folder=O/name;folder.mkdir(mode=0o700);query=q.query(names,constraints,with_values);p=folder/'ONE-control-query.smt2';p.write_text(query);emit(dict(kind='AFTER_NEW_analytic_query_proposal_RETURN',name=name,query=query,sha256=sha(p)));r=q.run_once(exe,p,folder,5,emit);save(name+'-SUBPROCESS-RETURN.json',r);calls+=1;classified=q.classification(r['stdout']);emit(dict(kind='AFTER_NEW_analytic_subprocess_classification_RETURN',name=name,RETURN=classified));assert r['exit_code']==0 and r['guard_event']is None and classified['status']==expected
 return classified

def roundtrip():
 coeff={(0,0):F(-7,13),(2,1):F(1,1<<160),(1,2):F(-5,9)};expr=q.polynomial(coeff,['a','b']);tree=q.parse(expr)[0];returns=[]
 for a,b in [(F(0),F(0)),(F(-2,3),F(4,5)),(F(7,11),F(-9,17)),(F(1),F(1))]:
  direct=sum(v*a**i*b**j for(i,j),v in coeff.items());value=q.exact(tree,dict(a=a,b=b));returns.append(dict(a=c.rec(a),b=c.rec(b),direct=c.rec(direct),encoded=c.rec(value)));assert direct==value
 return dict(expression=expr,exact_evaluations=returns,tiny_nonzero_power_retained=True)
check('exact_rational_power_and_negative_sign_roundtrip',dict(new_encoding=True),roundtrip)

def rejected():
 cases=[({(-1,):F(1)},['a']),({(1,0):F(1)},['a']),({(1,1):F(1)},['a','a'])];result=[]
 for co,names in cases:
  try:q.polynomial(co,names)
  except ValueError as e:result.append(str(e))
  else:raise AssertionError('invalid power/scope accepted')
 return dict(explicit_rejections=result)
check('invalid_power_missing_axis_and_duplicate_alias_rejected',dict(cases=3),rejected)
check('SMT_exact_tiny_structural_nonzero_sign',dict(epsilon_denominator_bit=160),lambda:CLI('tiny',['a'],['(= a (- 1))','(> (* '+q.rational(F(1,1<<160))+' a) 0)'],'unsat',False))
check('SMT_strict_zero_depth_rejection',dict(depth_truth='strict_positive_required'),lambda:CLI('depth',['depth'],['(= depth 0)','(> depth 0)'],'unsat',False))
check('SMT_shared_variable_alias_identity',dict(no_independent_variable_copy=True),lambda:CLI('alias',['a','b'],['(not (= (* a a b) (* b a a)))'],'unsat',False))

def algebraic():
 out=CLI('algebraic',['r','s','v'],['(= (* r r) 2)','(> r 0)','(= s (- r))','(= v (/ 1 3))'],'sat');returned=out['forms'][1];assert {v[0]for v in returned}=={'r','s','v'};bounds={}
 for name,value in returned:
  emit(dict(kind='BEFORE_NEW_analytic_algebraic_parser',name=name,value=value));bounds[name]=q.algebraic_interval(value,lambda a:emit(dict(name=name,RETURN=a)))
 rlo,rhi=map(c.dec,[bounds['r']['lower'],bounds['r']['upper']]);slo,shi=map(c.dec,[bounds['s']['lower'],bounds['s']['upper']]);assert 0<rlo<=rhi and rlo*rlo<=2<=rhi*rhi and slo<=shi<0 and slo*slo>=2>=shi*shi and c.dec(bounds['v']['lower'])==c.dec(bounds['v']['upper'])==F(1,3)
 return dict(all_exact_algebraic_intervals=bounds,midpoints_NOT_candidates=True)
check('SMT_algebraic_positive_negative_and_rational_RETURN_intervals',dict(new_fixed96bit_return_isolation=True),algebraic)

def inner():
 # Saved h=P+G, |g-P|<=E<=G implies h-2G<=g<=h.
 P=F(3,7);G=F(1,1<<25);E=G/F(3);out=[]
 for gval in [P-E,P,P+E]:
  h=P+G;lower=h-2*G;out.append(dict(lower=c.rec(lower),g=c.rec(gval),h=c.rec(h)));assert lower<=gval<=h
 # h>=0 alone is not sufficient; lower>=0 is a sufficient numerical-model bound.
 P=-E/F(2);gval=P-E;h=P+G;assert h>0>gval and h-2*G<gval
 return dict(necessary_and_sufficient_distinguished=True,triangle_bound_cases=out,positive_upper_negative_true_counterexample=dict(h=c.rec(h),g=c.rec(gval)),no_candidate_gate_changed=True)
check('NEW_saved_upper_guard_to_sufficient_lower_implication',dict(uses_original_coefficient_uniform_guard=True),inner)

def status():
 out=[q.classification(s)for s in ['unknown\n','timeout\n','unsat\n(error "model is not available")\n']];assert [r['status']for r in out]==['unknown','UNRESOLVED_nonstandard_RETURN','unsat'];assert q.parse('(\"error with spaces\")')==[['"error with spaces"']];return out
check('unknown_timeout_and_unavailable_model_remain_explicit',dict(no_retry_or_exclusion=True),status)

def root_unavailable():
 out=[]
 for value in [['root-obj',['+',['^','x','2'],'1'],'1'],['root-obj',['^','x','33'],'1']]:
  try:q.algebraic_interval(value,lambda x:emit(dict(case='unavailable_root',RETURN=x)))
  except ValueError as e:out.append(str(e))
  else:raise AssertionError('unavailable root accepted')
 return dict(rejections=out,complex_roots_retained_as_count=True,parse_cap_NOT_old_resultant_cap=True)
check('nonreal_root_and_NEW_return_parse_cap_unresolved',dict(fixed_degree_cap=32,fixed_coefficient_bit_cap=4096),root_unavailable)

def domain():
 low,high=F(-7,5),F(9,4);t=F(2,7);u=low+(high-low)*t;expr='(+ '+q.rational(low)+' (* '+q.rational(high-low)+' t))';assert q.exact(q.parse(expr)[0],dict(t=t))==u;assert q.exact(q.parse('(+ 1 (^ u 2))')[0],dict(u=u))>=1
 return dict(affine_exact=c.rec(u),positive_chart_denominator_bound=1,finite_coordinates_NOT_measured_angles=True)
check('explicit_affine_domain_and_positive_finite_chart_encoding',dict(no_mandatory_source_seed=True),domain)
assert len(checks)==10 and calls==4;save('NEW-contract-certificates.json',checks);J.close();report=dict(schema='v2d.direct-real-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_contracts_passed=True,new_analytic_contracts=10,new_tiny_real_algebraic_solver_calls=calls,new_episode_solver_calls=0,runtime_sha256=R['sha256'],helper_sha256=sha(S/'compute/direct_real_feasibility_v1.py'),old_controls_source_polynomials_and_audits_NOT_replayed=True,solver_RETURN_intervals_NOT_robot_candidates=True,no_original_gate_or_authority_change=True,new_native_position_checks=0,new_GPU_jobs=0,all_outputs_sha256={str(p.relative_to(O)):sha(p)for p in O.rglob('*')if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',report);print(json.dumps({k:v for k,v in report.items()if k!='all_outputs_sha256'}))
