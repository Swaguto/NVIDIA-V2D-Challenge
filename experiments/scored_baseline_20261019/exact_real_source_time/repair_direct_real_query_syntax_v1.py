"""Retain rejected partial query; prove multiplication-only syntax equivalent.
No old model producer, source polynomial, numerical gate or solver result repaired.
"""
from pathlib import Path
from fractions import Fraction as F
import ast,datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-direct-real-multiplication-query-v1';P=S/'e037-direct-real-model-v1';sys.path.insert(0,str(S/'compute'))
import direct_real_feasibility_v3 as q
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'all-syntax-control-and-query-identity-returns.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
runtime=read(S/'compute/direct-real-arithmetic-runtime-inspection-v1.json');old=read(S/'e037-direct-real-evaluation-v1/SOLE-exact-real-SUBPROCESS-RETURN.json');assert old['stdout'].count('unknown constant ^')==22 and old['exit_code']==1;emit(dict(kind='RETAINED_invalid_partial_query_RETURN_BY_HASH',sha256=sha(S/'e037-direct-real-evaluation-v1/SOLE-exact-real-SUBPROCESS-RETURN.json'),ignored_polynomial_rows=22,later_sat_INVALID_NOT_source_or_full_query_feasibility=True))
# This NEW CLI contract covers actual multivariate powers0..4 and rational/tiny signs.
co={(i,j):F((-1)**(i+j)*(i+2*j+1),((1<<130)if(i,j)==(4,4)else(3+i+j)))for i in range(5)for j in range(5)};poly=q.polynomial(co,['a','b']);expected=sum(v*F(2)**i*F(-3)**j for(i,j),v in co.items());query=q.query(['a','b'],['(= a 2)','(= b (- 3))','(not (= '+poly+' '+q.rational(expected)+'))'],False);control=O/'NEW-CLI-power-control';control.mkdir(mode=0o700);path=control/'ONE-control-query.smt2';path.write_text(query);emit(dict(kind='BEFORE_NEW_actual_CLI_multivariate_power_contract',query=query,expected=c.rec(expected)));result=q.run_once(runtime['executable'],path,control,5,emit);save('NEW-power-contract-SUBPROCESS-RETURN.json',result);assert result['exit_code']==0 and result['stdout'].strip()=='unsat'and not result['stderr'];emit(dict(kind='AFTER_NEW_actual_CLI_power_contract_PASS',all25multivariate_powers_zero_to_four=True,exact_rational_expected=c.rec(expected)))
model=read(P/'saved-polynomial-domain-authority-bindings.json');original=read(P/'exact-query-constraint-records.json');repaired=[];forms=[]
for row in original:
 co={tuple(v['power']):c.dec(v['value'])for v in row['coefficients']};body=q.polynomial(co,row['scope']);expr='('+row['sense']+' '+body+' 0)';parsed=q.parse(body)[0];assert '^'not in body
 # Independent exact evaluations check new multiplicative encoding against SAVED coefficients;
 # coefficient source/maps/errors stay bound by the completed482-coefficient audit BY HASH.
 samples=[]
 for n in range(3):
  values={name:F((n+1)*(k+2),7+n)for k,name in enumerate(row['scope'])};direct=sum(v*__import__('functools').reduce(lambda a,b:a*b,[values[name]**r for name,r in zip(row['scope'],power)],F(1))for power,v in co.items());encoded=q.exact(parsed,values);samples.append(dict(direct=c.rec(direct),encoded=c.rec(encoded)));assert direct==encoded
 out=dict(row,SMT_expression=expr,original_row_math_and_all_coefficients_UNCHANGED=True);repaired.append(out);forms.append(expr);emit(dict(kind='AFTER_NEW_query_syntax_identity_RETURN',id=row['id'],original_expression=row['SMT_expression'],multiplication_expression=expr,scope=row['scope'],sample_identity_returns=samples,all_original_coefficient_records_identical=True))
newquery=q.query(model['variables'],forms,True);assert len(repaired)==74 and '^'not in newquery;path=O/'ONE-saved-five-clock-QF_NRA-query.smt2';path.write_text(newquery);save('exact-query-constraint-records.json',repaired);binding=dict(model);binding.update(query_sha256=sha(path),constraint_records_sha256=sha(O/'exact-query-constraint-records.json'),original_model_binding_sha256=sha(P/'saved-polynomial-domain-authority-bindings.json'),syntax_repair_only_NO_source_model_regeneration=True,invalid_first_attempt_RETAINED=True,first_complete_feasibility_evaluation_STILL_PENDING=True);save('saved-polynomial-domain-authority-bindings.json',binding);J.close();report=dict(schema='v2d.direct-real-query-syntax-repair.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_actual_CLI_power_contract_passed=True,new_tiny_solver_calls=1,new_episode_solver_calls=0,original74row_math_482coefficient_bindings_UNCHANGED=True,three_NEW_encoding_identities_per_row=222,query_sha256=sha(path),model_binding_sha256=sha(O/'saved-polynomial-domain-authority-bindings.json'),actual_smt_power_parser_failure_retained=True,invalid_sat_NOT_accepted=True,all_outputs_sha256={str(p.relative_to(O)):sha(p)for p in O.rglob('*')if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',report);print(json.dumps({k:v for k,v in report.items()if k!='all_outputs_sha256'}))
