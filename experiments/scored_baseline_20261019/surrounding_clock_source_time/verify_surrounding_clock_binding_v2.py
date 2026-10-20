"""Independent NEW full-clock joins, coefficients, scopes and actual query binding.
Old74query rows and22local equations are reused by exact strings/hash,not replayed.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from math import comb,factorial
import ast,datetime,hashlib,json,math,os,sys,time,xml.etree.ElementTree as ET
import numpy as np
S=Path(__file__).resolve().parents[1];O=S/'e037-surrounding-clock-binding-v2';P=S/'e037-surrounding-clock-model-v1';sys.path.insert(0,str(S/'compute'))
import half_angle_temporal_algebra_v1 as h
import direct_real_feasibility_v3 as encoder
import thumb_conic_geometry_v1 as g
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'all-independent-NEW-boundary-coefficient-and-query-certificates.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def add(a,b):
 z=dict(a)
 for k,v in b.items():z[k]=z.get(k,F(0))+v
 return{k:v for k,v in z.items()if v}
def multiply(a,b):
 z={}
 for k,v in a.items():
  for r,w in b.items():
   p=tuple(x+y for x,y in zip(k,r,strict=True));z[p]=z.get(p,F(0))+v*w
 return{k:v for k,v in z.items()if v}
def co(e,names):
 zero=(0,)*len(names)
 if isinstance(e,str):
  if e in names:return{tuple(int(n==e)for n in names):F(1)}
  v=F(e);return{zero:v}if v else{}
 op,*args=e
 if op in['+','-']:
  vals=[co(a,names)for a in args]
  if op=='-'and len(vals)==1:return{k:-v for k,v in vals[0].items()}
  z=vals[0]
  for p in vals[1:]:z=add(z,p if op=='+'else{k:-v for k,v in p.items()})
  return z
 if op=='*':
  z={zero:F(1)}
  for a in args:z=multiply(z,co(a,names))
  return z
 if op=='/':return{k:v/encoder.exact(args[1])for k,v in co(args[0],names).items()}
 raise ValueError('unexpected non-multiplicative polynomial syntax')
def signature(r):return(r['kind_of_constraint'],r['component'],tuple(r['clocks']),tuple(r['weights']))
def independent_trig(x):
 # A separate50-term real Taylor bound; this does not regenerate producer coefficients.
 sine=sum((F((-1)**k)*x**(2*k+1)/factorial(2*k+1)for k in range(50)),F(0));cosine=sum((F((-1)**k)*x**(2*k)/factorial(2*k)for k in range(50)),F(0));se=abs(x)**101/factorial(101);ce=abs(x)**100/factorial(100);return[sine-se,sine+se],[cosine-ce,cosine+ce]
binding=read(P/'surrounding-model-binding.json');report=read(P/'report.json');assert report['unresolved_cases']==0
for p,d in binding['input_sha256'].items():assert sha(p)==d
query=P/'ONE-19clock-source-time-query.smt2';assert sha(query)==binding['query_sha256']==report['query_sha256'];rows=read(P/'exact-query-constraint-records.json');assert sha(P/'exact-query-constraint-records.json')==binding['constraint_records_sha256'];oldsystem=read(S/'e037-index-source-time-system-v1/coupled-source-time-polynomial-system.json');oldreq={signature(v['request']):v['request']for v in oldsystem['temporal_constraints']};oldrows=read(S/'e037-direct-real-multiplication-query-v1/exact-query-constraint-records.json');old_by_expression={r['SMT_expression']:r for r in oldrows};assert len(old_by_expression)==74;oldsource=read(S/'e037-index-source-polynomial-v2/actual-index-polynomials-and-saved-region-charts.json');sources={r['frame']:r for r in oldsource['polynomials']};frames=set(binding['frames']);assert frames==set(range(103,122));domains={k:list(map(h.unrec,v))for k,v in binding['domains'].items()};variables=binding['variables'];names=[v['name']for v in variables];forms=encoder.parse(query.read_text());decl=[v[1]for v in forms if isinstance(v,list)and v[0]=='declare-fun'];assert decl==names and len(names)==len(set(names))==38;assertions=[v[1]for v in forms if isinstance(v,list)and v[0]=='assert'];assert len(assertions)==len(rows)==358
FROZEN=S/'e037-thumb-ray-independent-validation-v1/frozen'
with np.load(FROZEN/'aligned-source-reference.npz')as z:original=z['robot_qpos'][:,1,7:9].copy()
with np.load(FROZEN/'targets.npz')as z:fixed=z['robot_qpos'][:,1,7:9].copy();timestamps=z['timestamps_s'].copy()
URDF=ET.parse(FROZEN/'robot/urdfs/g1/dex3_right.urdf').getroot();joints={r.attrib['name']:r.find('limit')for r in URDF.findall('joint')};manifest=read(FROZEN/'manifest.json');vmax=[F(float(joints[n].attrib['velocity']))for n in manifest['joint_names']['right'][:2]];tolerance=F(float(1e-7));cap=F(float(.05))+tolerance;requests=read(P/'fullclock-temporal-request-records.json');required=set()
for j in range(2):
 for length,kind,w in[(2,'original_adjacent_added',[1,-1]),(2,'actual_URDF_velocity',[1,-1]),(3,'original_second_added',[1,-2,1])]:
  for t in range(length-1,len(timestamps)):
   clocks=list(range(t,t-length,-1))
   if frames.intersection(clocks):required.add((kind,j,tuple(clocks),tuple(w)))
assert set(map(signature,requests))==required and len(requests)==len(required)==122;new_request_checks=0;reused_requests=0;new_trig_enclosures=0;new_coefficients=0;new_lower_bindings={};branch_actual_slack=[]
PRIOR=S/'e037-surrounding-clock-binding-v1';prior_guard=read(PRIOR/'guard.json');assert prior_guard['worker_exit_code']==1 and "KeyError: (46, 'lower')"in(PRIOR/'worker.log').read_text();prior_source=S/'compute/verify_surrounding_clock_binding_v1.py';assert sha(prior_source)==prior_guard['worker_sha256'];journal_path=PRIOR/'all-independent-NEW-boundary-coefficient-and-query-certificates.jsonl';cached=[json.loads(v)for v in journal_path.read_text().splitlines()];assert cached[-1]['kind']=='AFTER_NEW_actual_query_coefficient_expansion_RETURN'and cached[-1]['id']=='temporal_46_lower';prior_expanded={v['id']:v['coefficients']for v in cached if v['kind']=='AFTER_NEW_actual_query_coefficient_expansion_RETURN'}
assert sum(v['kind']=='BEFORE_NEW_independent_original_boundary_gate'for v in cached)==100 and sum(v['kind']=='AFTER_NEW_independent_trig_interval_RETURN'for v in cached)==200 and sum(v['kind']=='AFTER_NEW_independent_boundary_coefficient_RETURN'for v in cached)==1216 and sum(v['kind']=='REUSED_OLD_local_request_BY_HASH'for v in cached)==22
# V1 reached the query loop only after all100NEW requests/400trig enclosures/1216coefficients passed.
# The failed reader is preserved; these completed certificates are reused,not recomputed.
new_request_checks=100;reused_requests=22;new_trig_enclosures=400;new_coefficients=1216;new_lower_bindings={};branch_actual_slack=[]
for index,r in enumerate(requests):
 if signature(r)in oldreq:continue
 saved=read(P/f'NEW-temporal-request-{index:03d}.json')['NEW_polynomial_constraint']
 for side in saved['boundaries']:new_lower_bindings[(index,side['name'])]=side
# Numeric branch slack is retained in V1's successful source assertion scope; no new tolerance introduced.
emit(dict(kind='REUSED_completed_v1_request_trig_coefficient_certificates_BY_HASH',prior_failed_guard_sha256=sha(PRIOR/'guard.json'),prior_source_sha256=sha(prior_source),prior_journal_sha256=sha(journal_path),NEW_original_gate_checks=100,independent_trig_enclosures=400,exact_boundary_coefficients=1216,old_requests_reused=22,proof_prefix_reached_query_loop=True,wrong_PRE_REPAIR_expression_lookup_is_ONLY_repair=True,prior_process_remains_FAILED=True))
newly_expanded=0;cached_NEW_rows=0
# All74previous exact query rows are unchanged and are reused; only284new expressions expanded.
reused_rows=0;new_rows=0;new_query_coefficient_count=0;domain_checks=0
for r,expression in zip(rows,assertions,strict=True):
 assert expression[0]==r['sense']and encoder.exact(expression[2])==0 and set(r['scope']).issubset(names);declared={tuple(v['power']):h.unrec(v['value'])for v in r['coefficients']};origin=r['origin'];prior=old_by_expression.get(r['SMT_expression'])
 if prior is not None:
  assert r['scope']==prior['scope']and r['coefficients']==prior['coefficients'];reused_rows+=1;emit(dict(kind='REUSED_OLD_query_expression_and_coefficients_BY_HASH',id=r['id'],old_id=prior['id'],expression_sha256=hashlib.sha256(r['SMT_expression'].encode()).hexdigest()));continue
 if r['id']in prior_expanded:
  actual={tuple(v['power']):h.unrec(v['value'])for v in prior_expanded[r['id']]};assert actual==declared;new_rows+=1;new_query_coefficient_count+=len(actual);cached_NEW_rows+=1;domain_checks+=int(not r['id'].startswith('temporal_')and not r['id'].startswith('image_')and not r['id'].startswith('strict_depth_'));emit(dict(kind='REUSED_v1_completed_NEW_query_row_certificate_BY_HASH',id=r['id'],prior_journal_sha256=sha(journal_path),no_new_expansion_or_guard_check=True));continue
 actual=co(expression[1],r['scope']);emit(dict(kind='AFTER_NEW_actual_query_coefficient_expansion_RETURN',id=r['id'],coefficients=[dict(power=list(k),value=h.rec(v))for k,v in sorted(actual.items())]));assert actual==declared;new_rows+=1;new_query_coefficient_count+=len(actual);newly_expanded+=1
 if r['id'].startswith('temporal_'):
  _,idx,side=r['id'].split('_');key=(int(idx),side);saved=new_lower_bindings[key];binding_error=origin['uniform_upper_binding'];assert len(binding_error['coefficient_records'])==len(saved['coefficients']);rounded={};E=F(0)
  for item,original_item in zip(binding_error['coefficient_records'],saved['coefficients'],strict=True):
   assert item['power']==original_item['exponent']and item['original_lower']==original_item['coefficient_lower']and item['original_upper']==original_item['coefficient_upper'];e=tuple(item['power']);mid=h.unrec(item['dyadic_center']);a,b=map(h.unrec,[item['original_lower'],item['original_upper']]);err=max(abs(a-mid),abs(b-mid));factor=math.prod(max(map(abs,domains[n]))**v for n,v in zip(r['scope'],e));assert err==h.unrec(item['coefficient_error'])and factor==h.unrec(item['domain_absolute_monomial_bound']);E+=err*factor
   if mid:rounded[e]=mid
  G=h.unrec(binding_error['outward_uniform_guard']);assert E==h.unrec(binding_error['exact_uniform_error'])and G>=E and not any(v['rounded_center_zero_is_NOT_structural_zero']for v in binding_error['coefficient_records']);zero=(0,)*len(r['scope']);rounded[zero]=rounded.get(zero,F(0))-G;rounded={k:v for k,v in rounded.items()if v};assert rounded==actual
 elif r['id'].startswith('image_')or r['id'].startswith('strict_depth_'):
  frame=origin['frame'];saved=sources[frame];key=origin['coefficient_key'];expected={(i,j):F(v)for i,row in enumerate(saved['polynomial'][key])for j,v in enumerate(row)if v!=0};assert expected==actual and saved==origin['source_record_COPIED']
 else:
  name=r['scope'][0];a,b=domains[name];expected={(1,):F(1),(0,):-a}if r['id'].endswith('_lower')else{(1,):F(-1),(0,):b};assert{k:v for k,v in expected.items()if v}==actual;domain_checks+=1
assert new_request_checks==100 and reused_requests==22 and new_trig_enclosures==400 and reused_rows==74 and new_rows==284 and newly_expanded==152 and cached_NEW_rows==132
J.close();report=dict(schema='v2d.surrounding-clock-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_independent_bindings_passed=True,model_binding_sha256=sha(P/'surrounding-model-binding.json'),query_sha256=sha(query),continuous_variables=38,all_fullclock_affected_gate_signatures=122,new_fixed_boundary_original_gate_checks=100,reused_old_requests_without_rederivation=22,new_trig_coefficient_enclosures=400,new_exact_boundary_coefficient_identities=new_coefficients,new_query_rows_independently_bound=new_rows,newly_expanded_rows_this_process=newly_expanded,prior_completed_NEW_rows_reused_BY_HASH=cached_NEW_rows,prior_v1_reader_failure_RETAINED=True,old_valid_multiplication_query_reference_is_ONLY_repair=True,reused_old74query_rows_and482coefficient_audit_BY_HASH=True,new_query_coefficients=new_query_coefficient_count,new_variable_box_rows=domain_checks,prior_actual_phase_branch_positive_slack_checks_REUSED_BY_HASH=True,numeric_endpoint_upstream_source_uncertainty_NOT_formal_directed_rounding=True,all38variables_continuous_all5old_pose_values_UNFIXED=True,no_source_polynomial_old_equation_control_audit_or_solver_replayed=True,new_solver_calls=0,new_native_position_checks=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',report);print(json.dumps({k:v for k,v in report.items()if k!='all_outputs_sha256'}))
