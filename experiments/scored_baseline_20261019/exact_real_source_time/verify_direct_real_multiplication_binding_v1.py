"""Independent complete multiplication-query coefficient identities; old source audit reused."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-direct-real-multiplication-binding-v1';P=S/'e037-direct-real-multiplication-query-v1';sys.path.insert(0,str(S/'compute'))
import direct_real_feasibility_v3 as q
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'all-NEW-multiplication-only-coefficient-identities.jsonl').open('w');emit=lambda v:g.journal(J,v)
def addition(a,b):
 z=dict(a)
 for k,v in b.items():z[k]=z.get(k,F(0))+v
 return{k:v for k,v in z.items()if v}
def product(a,b):
 z={}
 for k,v in a.items():
  for r,w in b.items():
   power=tuple(x+y for x,y in zip(k,r,strict=True));z[power]=z.get(power,F(0))+v*w
 return{k:v for k,v in z.items()if v}
def coefficients(e,names):
 zero=(0,)*len(names)
 if isinstance(e,str):
  if e in names:return {tuple(int(n==e)for n in names):F(1)}
  return{zero:F(e)}if F(e)else{}
 op,*args=e
 if op in['+','-']:
  vals=[coefficients(a,names)for a in args]
  if op=='-'and len(vals)==1:return{k:-v for k,v in vals[0].items()}
  z=vals[0]
  for p in vals[1:]:z=addition(z,p if op=='+'else{k:-v for k,v in p.items()})
  return z
 if op=='*':
  z={zero:F(1)}
  for a in args:z=product(z,coefficients(a,names))
  return z
 if op=='/':
  den=q.exact(args[1]);return{k:v/den for k,v in coefficients(args[0],names).items()}
 if op=='^':
  exponent=q.exact(args[1]);assert exponent.denominator==1 and 0<=exponent<=4;z={zero:F(1)};base=coefficients(args[0],names)
  for _ in range(int(exponent)):z=product(z,base)
  return z
 raise ValueError('nonpolynomial query term')
binding=read(P/'saved-polynomial-domain-authority-bindings.json');report=read(P/'report.json');assert report['NEW_actual_CLI_power_contract_passed']and sha(P/'ONE-saved-five-clock-QF_NRA-query.smt2')==binding['query_sha256']==report['query_sha256'];original=read(S/'e037-direct-real-model-v1/exact-query-constraint-records.json');rows=read(P/'exact-query-constraint-records.json');forms=q.parse((P/'ONE-saved-five-clock-QF_NRA-query.smt2').read_text());assertions=[v[1]for v in forms if isinstance(v,list)and v[0]=='assert'];declared=[v[1]for v in forms if isinstance(v,list)and v[0]=='declare-fun'];assert declared==binding['variables']and len(assertions)==len(rows)==len(original)==74;count=0
for r,old,expr in zip(rows,original,assertions,strict=True):
 for key in ['id','scope','coefficients','sense','origin']:assert r[key]==old[key]
 emit(dict(kind='BEFORE_NEW_complete_syntax_coefficient_identity',id=r['id'],parsed_assertion=expr));sense,body,rhs=expr;assert sense==r['sense']and q.exact(rhs)==0;actual=coefficients(body,r['scope']);expected={tuple(v['power']):c.dec(v['value'])for v in old['coefficients']};emit(dict(kind='AFTER_NEW_complete_syntax_coefficient_RETURN',id=r['id'],actual=[dict(power=list(k),value=c.rec(v))for k,v in sorted(actual.items())],expected=[dict(power=list(k),value=c.rec(v))for k,v in sorted(expected.items())]));assert actual==expected;count+=len(actual)
assert count==482;J.close();out=dict(schema='v2d.direct-real-multiplication-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_independent_bindings_passed=True,query_sha256=binding['query_sha256'],model_binding_sha256=sha(P/'saved-polynomial-domain-authority-bindings.json'),constraint_rows=74,independent_exact_coefficient_bindings=482,all_original_math_scope_coefficients_and_provenance_UNCHANGED=True,first_invalid_partial_query_RETAINED=True,first_complete_feasibility_evaluation_PENDING=True,no_old_source_or_domain_audit_replayed=True,new_solver_calls=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
with(O/'report.json').open('w')as f:f.write(json.dumps(out,indent=2)+'\n');f.flush();os.fsync(f.fileno())
print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))
