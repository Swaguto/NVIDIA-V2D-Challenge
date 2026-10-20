"""Compile once; independently bind NEW Jacobian/cache and exact arithmetic at F.
Source/time polynomial producers and prior numerical audits are NOT replayed.
"""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,math,os,sys,time
import numpy as np
S=Path(__file__).resolve().parents[1];O=S/'e037-primal-restoration-binding-v1';P=S/'e037-surrounding-clock-model-v1';sys.path.insert(0,str(S/'compute'))
import primal_inequality_restoration_v1 as p
import primal_saved_compilation_v1 as cache
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
read=lambda q:json.loads(Path(q).read_text());sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest();start=time.monotonic();J=(O/'all-primal-compilation-and-independent-derivative-certificates.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def direct(r,x):
 total=F(0)
 for t in r['coefficients']:
  v=c.dec(t['value'])
  for name,k in zip(r['scope'],t['power'],strict=True):v*=x[name]**k
  total+=v
 return total
contract_path=S/'e037-primal-restoration-contract-v1/report.json';contract=read(contract_path);assert contract['all_NEW_primal_contracts_passed']and contract['helper_sha256']==sha(S/'compute/primal_inequality_restoration_v1.py');original_binding=P/'surrounding-model-binding.json';model=read(original_binding);old_ind_path=S/'e037-surrounding-clock-binding-v2/report.json';old_ind=read(old_ind_path);assert old_ind['all_NEW_independent_bindings_passed']and old_ind['model_binding_sha256']==sha(original_binding);records_path=P/'exact-query-constraint-records.json';assert sha(records_path)==model['constraint_records_sha256'];records=read(records_path);names=[v['name']for v in model['variables']];domains={k:list(map(c.dec,v))for k,v in model['domains'].items()};assert len(records)==358 and len(names)==38
emit(dict(kind='BEFORE_NEW_primal_compilation',original_model_binding_sha256=sha(original_binding),original_constraint_records_sha256=sha(records_path),prior_source_time_domain_and_error_binding_REUSED_BY_HASH=sha(old_ind_path),no_original_coefficients_or_source_confidence_changed=True));system=p.PolynomialSystem(records,names,domains,emit);exactrows=[]
for r in system.rows:exactrows.append(dict(id=r['id'],scope=r['scope'],index=r['index'],terms=[dict(power=list(e),value=c.rec(v))for e,v in r['terms']],sign=r['sign'],strict=r['strict']))
save('exact-primal-row-bindings.json',exactrows);save('derivative-index-maps.json',system.derivative_maps);np.savez_compressed(O/'compiled-primal-system.npz',**{key:getattr(system,key)for key in['rowids','powers','coeff','drows','dcols','dpowers','dcoeff','scales','signs','strict']})
FROZEN=S/'e037-thumb-ray-independent-validation-v1/frozen';assert sha(FROZEN/'manifest.json')=='f438adf3da846ec1c76c31c7b6d6c12b46bea76004aa7745cfada504378da606'
with np.load(FROZEN/'targets.npz')as z:original_F=z['robot_qpos'].copy()
initial=[];initial_rows=[]
for name in names:
 _,frame,component=name.split('_');angle=float(original_F[int(frame),1,7+int(component)]);u=math.tan(angle/2);initial.append(u);lo,hi=domains[name];initial_rows.append(dict(variable=name,F_original_angle_rad=angle,actual_initial_halfangle=u,inside_saved_domain=bool(lo<=F(u)<=hi),initial_NOT_clipped_or_accepted=True))
initial=np.array(initial);np.save(O/'initial-F-halfangle-vector.npy',initial);save('initial-F-provenance-and-retained-domain-violations.json',initial_rows);files=['exact-primal-row-bindings.json','derivative-index-maps.json','compiled-primal-system.npz','initial-F-halfangle-vector.npy','initial-F-provenance-and-retained-domain-violations.json'];binding=dict(schema='v2d.cached-primal-compilation.v1',variables=names,rows=358,original_constraint_records_path=str(records_path),original_constraint_records_sha256=sha(records_path),original_source_time_model_binding_sha256=sha(original_binding),prior_independent_bindings_REUSED_BY_HASH=sha(old_ind_path),new_contracts_sha256=sha(contract_path),original_F_manifest_sha256=sha(FROZEN/'manifest.json'),algorithm_configuration=p.CONFIG,compiled_file_sha256={n:sha(O/n)for n in files},all_normalization_and_damping_ALGORITHMIC_not_candidate_gates_or_source_confidence=True,all38variables_continuous_all5oldSATvalues_UNUSED=True,initial_F_only_not_clipped=True);save('compiled-primal-binding.json',binding);emit(dict(kind='AFTER_NEW_cached_primal_compilation_RETURN',binding=binding));hydrated=cache.hydrate(O,emit)
for key in['rowids','powers','coeff','drows','dcols','dpowers','dcoeff','scales','signs','strict']:assert np.array_equal(getattr(system,key),getattr(hydrated,key))
value,Jacobian=hydrated.evaluate(initial,True);emit(dict(kind='AFTER_NEW_numeric_F_value_Jacobian_RETURN',initial_vector=initial,normalized_signed_values=value,Jacobian=Jacobian));point={n:F(float(v))for n,v in zip(names,initial)};stencil=F(1,64);derivative_checks=0;index_terms=0;value_errors=[];derivative_errors=[]
for i,(record,r)in enumerate(zip(records,hydrated.rows,strict=True)):
 assert record['id']==r['id']and record['scope']==r['scope']and r['index']==[names.index(n)for n in record['scope']];expected_terms=[(tuple(v['power']),c.dec(v['value']))for v in record['coefficients']];assert expected_terms==r['terms'];sign=-1 if record['sense']=='<='else 1;assert sign==r['sign']and(record['sense']=='>')==r['strict'];B=sum(abs(v)*math.prod(max(map(abs,domains[n]))**k for n,k in zip(record['scope'],e))for e,v in expected_terms);numeric_scale=F(float(hydrated.scales[i]));assert numeric_scale>0 and numeric_scale>=(B if B>0 else 1)
 # Exact independent degree-four stencil, using only original values at fourNEW shifted analytic points.
 expected_maps=[]
 for term_idx,(e,v)in enumerate(expected_terms):
  for local,k in enumerate(e):
   if k:
    globalpower=[0]*len(names)
    for name,power in zip(record['scope'],e):globalpower[names.index(name)]=power
    column=names.index(record['scope'][local]);globalpower[column]-=1;expected_maps.append(dict(original_term_index=term_idx,variable=record['scope'][local],global_column=column,multiplicity=k,derivative_global_power=globalpower))
 assert expected_maps==hydrated.derivative_maps[i];index_terms+=len(expected_maps)
 for name in record['scope']:
  assert max((e[record['scope'].index(name)]for e,v in expected_terms),default=0)<=4;f=[]
  for k in[-2,-1,1,2]:
   shifted=dict(point);shifted[name]+=k*stencil;f.append(direct(record,shifted))
  independent=(f[0]-8*f[1]+8*f[2]-f[3])/(12*stencil);mapped=F(0);magnitude=F(0)
  for m in expected_maps:
   if m['variable']!=name:continue
   term=expected_terms[m['original_term_index']][1]*m['multiplicity']
   for n,k in zip(names,m['derivative_global_power']):term*=point[n]**k
   mapped+=term;magnitude+=abs(term)
  emit(dict(kind='AFTER_NEW_independent_exact_derivative_stencil_RETURN',row=i,variable=name,four_point_values=[c.rec(v)for v in f],independent_derivative=c.rec(independent),saved_index_map_derivative=c.rec(mapped)));assert independent==mapped;derivative_checks+=1;exact=mapped*sign/numeric_scale;error=abs(F(float(Jacobian[i,names.index(name)]))-exact);rounding=magnitude/numeric_scale*F(64*(len(expected_terms)+len(names)+16),1<<52);emit(dict(kind='AFTER_NEW_numeric_derivative_representation_error_RETURN',row=i,variable=name,exact=c.rec(exact),actual=float(Jacobian[i,names.index(name)]),error=c.rec(error),declared_diagnostic_envelope=c.rec(rounding),NOT_original_candidate_tolerance=True));assert error<=rounding;derivative_errors.append(float(error))
 exact=direct(record,point)*sign/numeric_scale;error=abs(F(float(value[i]))-exact);magnitude=sum(abs(v)*math.prod(abs(point[n])**k for n,k in zip(record['scope'],e))for e,v in expected_terms);rounding=magnitude/numeric_scale*F(64*(len(expected_terms)+len(names)+16),1<<52);emit(dict(kind='AFTER_NEW_numeric_value_representation_error_RETURN',row=i,exact=c.rec(exact),actual=float(value[i]),error=c.rec(error),diagnostic_envelope=c.rec(rounding)));assert error<=rounding;value_errors.append(float(error))
initial_domain_bad=[r['variable']for r in initial_rows if not r['inside_saved_domain']];J.close();report=dict(schema='v2d.primal-restoration-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_independent_bindings_passed=True,original_saved_polynomial_rows=358,continuous_variables=38,compiled_binding_sha256=sha(O/'compiled-primal-binding.json'),static_normalizers_positive_and_original_signs_preserved=True,exact_derivative_stencil_identities=derivative_checks,exact_derivative_index_terms=index_terms,numeric_values_checked=358,numeric_derivatives_checked=derivative_checks,maximum_normalized_value_error=max(value_errors),maximum_normalized_derivative_error=max(derivative_errors),numeric_diagnostic_envelopes_NOT_original_candidate_tolerances=True,initial_F_domain_violations_RETAINED=initial_domain_bad,initial_F_NOT_clipped_or_accepted=True,all5oldSATpoints_UNUSED=True,all_original_source_polynomial_domain_clock_authority_and_gates_UNCHANGED=True,new_episode_restoration_calls=0,new_source_or_URDF_or_native_position_checks=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',report);print(json.dumps({k:v for k,v in report.items()if k!='all_outputs_sha256'}))
