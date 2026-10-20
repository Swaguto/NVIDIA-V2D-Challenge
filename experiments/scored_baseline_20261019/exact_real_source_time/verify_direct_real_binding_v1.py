"""Independent exact syntax/coefficients/domains binding, not an old source audit."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-direct-real-binding-v1';P=S/'e037-direct-real-model-v1';sys.path.insert(0,str(S/'compute'))
import direct_real_feasibility_v2 as q
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'all-independent-query-binding-certificates.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
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
binding=read(P/'saved-polynomial-domain-authority-bindings.json');model=read(P/'report.json');assert model['ONE_query_construction']
for p,d in binding['inputs_sha256'].items():assert sha(p)==d
old_support=read(S/'e037-pointwise-support-independent-v1/report.json');assert old_support['all_independent_checks_passed'];emit(dict(kind='REUSED_saved_support_error_certificates_BY_HASH',sha256=sha(S/'e037-pointwise-support-independent-v1/report.json'),no_old_error_or_map_diagnosis_reexecuted=True))
query_path=P/'ONE-saved-five-clock-QF_NRA-query.smt2';assert sha(query_path)==binding['query_sha256'];records=read(P/'exact-query-constraint-records.json');assert sha(P/'exact-query-constraint-records.json')==binding['constraint_records_sha256'];forms=q.parse(query_path.read_text());declarations=[v[1]for v in forms if isinstance(v,list)and v[0]=='declare-fun'];assertions=[v[1]for v in forms if isinstance(v,list)and v[0]=='assert'];assert declarations==binding['variables']and len(declarations)==len(set(declarations))==10 and len(assertions)==len(records)==74
system=read(S/'e037-index-source-time-system-v1/coupled-source-time-polynomial-system.json');domains={k:list(map(c.dec,v))for k,v in binding['domains'].items()};original_domains=read(S/'e037-common-moment-model-v1/common-degree-factor-representations.json')['domains'];assert binding['domains']==original_domains;entries=0;rounded_zero=[]
for r,assertion in zip(records,assertions,strict=True):
 emit(dict(kind='BEFORE_NEW_query_row_binding',id=r['id'],parsed_assertion=assertion,source_origin=r['origin']));sense,expr,rhs=assertion;assert sense==r['sense']and q.exact(rhs)==0;declared={tuple(v['power']):c.dec(v['value'])for v in r['coefficients']};actual=coefficients(expr,r['scope']);origin=r['origin']
 if '_sufficient_lower'in r['id']:
  saved=read(origin['saved_factor_path']);assert sha(origin['saved_factor_path'])==origin['saved_factor_sha256'];assert saved['source_order']==r['scope']and[[c.dec(v)for v in b]for b in saved['original_half_angle_bounds']]==[domains[n]for n in r['scope']];u=saved['uniform_upper_binding'];G=c.dec(u['outward_uniform_guard']);E=c.dec(u['exact_uniform_error']);assert G>=E>=0;expected={tuple(v['power']):c.dec(v['value'])for v in u['upper_polynomial']};zero=(0,)*len(r['scope']);expected[zero]=expected.get(zero,F(0))-2*G;expected={k:v for k,v in expected.items()if v};rounded_zero.extend([dict(id=r['id'],power=v['power'])for v in u['coefficient_records']if v['rounded_center_zero_is_NOT_structural_zero']]);assert origin['uniform_upper_guard']==u['outward_uniform_guard']
 elif r['id'].startswith('image_')or r['id'].startswith('strict_depth_'):
  frame=int(r['id'].split('_')[-1]);source=next(v for v in system['source_constraints']if v['frame']==frame);p=source['source_record_COPIED_not_regenerated']['polynomial'];assert r['scope']==[f'u_{frame}_0',f'u_{frame}_1']and source['source_input_sha256']==origin['source_input_sha256'];expected={(i,j):F(v)for i,row in enumerate(p[origin['coefficient_key']])for j,v in enumerate(row)if v};assert p['denominator']==[[1.,0.,1.],[0.,0.,0.],[1.,0.,1.]]and p['minimum_depth']==.01
 else:
  assert len(r['scope'])==1;name=r['scope'][0];lo,hi=domains[name];expected={(1,):F(1),(0,):-lo}if r['id'].endswith('_lower')else{(1,):F(-1),(0,):hi};expected={k:v for k,v in expected.items()if v}
 emit(dict(kind='AFTER_NEW_query_row_coefficients_RETURN',id=r['id'],actual=[dict(power=list(k),value=c.rec(v))for k,v in sorted(actual.items())],expected=[dict(power=list(k),value=c.rec(v))for k,v in sorted(expected.items())]));assert actual==declared==expected;entries+=len(actual)
assert not rounded_zero;J.close();report=dict(schema='v2d.direct-real-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_independent_bindings_passed=True,query_sha256=sha(query_path),model_binding_sha256=sha(P/'saved-polynomial-domain-authority-bindings.json'),constraint_rows=74,independent_exact_coefficient_bindings=entries,continuous_variables=10,saved_temporal_upper_guard_to_lower_necessity_vs_sufficiency_proof_REUSED=True,temporal_tiny_structural_nonzero_coefficients_NOT_erased=True,positive_denominator_and_strict_depth_logic_bound=True,full_region2_and_all109_unknown_bands_retained=True,no_old_source_polynomials_controls_audits_or_solver_replayed=True,new_solver_calls=0,no_robot_candidate_or_gate_change=True,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',report);print(json.dumps({k:v for k,v in report.items()if k!='all_outputs_sha256'}))
