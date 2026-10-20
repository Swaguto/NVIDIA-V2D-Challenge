"""NEW19clock continuous source/time construction; all5old poses remain free."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,math,os,sys,time,xml.etree.ElementTree as ET
import numpy as np
S=Path(__file__).resolve().parents[1];O=S/'e037-surrounding-clock-model-v1';C=S/'compute';sys.path.insert(0,str(C))
import surrounding_temporal_requests_v1 as boundary
import surrounding_temporal_polynomials_v1 as polynomial
import half_angle_temporal_algebra_v1 as h
import pointwise_support_moments_v1 as upper
import direct_real_feasibility_v3 as encoder
import thumb_conic_geometry_v1 as g
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'all-new-surrounding-clock-proposals-and-intermediate-returns.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
contract=read(S/'e037-surrounding-boundary-contract-v1/report.json');assert contract['all_NEW_boundary_contracts_passed']and contract['helper_sha256']==sha(C/'surrounding_temporal_requests_v1.py')
sourcepath=S/'e037-index-source-polynomial-v2/actual-index-polynomials-and-saved-region-charts.json';source=read(sourcepath);assert sha(sourcepath)==read(sourcepath.parent/'report.json')['all_outputs_sha256'][sourcepath.name];region=next(r for r in source['regions']if r['saved_node_id']==2);frames=[r['frame']for r in source['polynomials']];assert frames==list(range(103,122));charts={v['frame']:v['chart']for v in region['source_charts']};oldsystempath=S/'e037-index-source-time-system-v1/coupled-source-time-polynomial-system.json';oldsystem=read(oldsystempath);oldreq={boundary.signature(v):v for v in oldsystem['temporal_constraints']};assert len(oldreq)==22
FROZEN=S/'e037-thumb-ray-independent-validation-v1/frozen';assert sha(FROZEN/'manifest.json')=='f438adf3da846ec1c76c31c7b6d6c12b46bea76004aa7745cfada504378da606';manifest=read(FROZEN/'manifest.json')
for n,d in manifest['files'].items():assert sha(FROZEN/n)==d
with np.load(FROZEN/'aligned-source-reference.npz')as z:original=z['robot_qpos'][:,1,7:9].copy()
with np.load(FROZEN/'targets.npz')as z:fixed=z['robot_qpos'][:,1,7:9].copy();timestamps=z['timestamps_s'].copy()
rootmode=S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz';assert sha(rootmode)=='15a362bbf9eb6956ba487b175ce4d04e0ac42ecfa2e4d245c22799cca81c2932'
oldqueries=read(S/'e037-direct-real-model-v1/exact-query-constraint-records.json');oldlower={(int(r['id'].split('_')[1]),r['id'].split('_')[2]):r for r in oldqueries if '_sufficient_lower'in r['id']};assert len(oldlower)==44
representation=S/'e037-common-moment-model-v1/common-degree-factor-representations.json';core={k:list(map(h.unrec,v))for k,v in read(representation)['domains'].items()};box=np.array(region['box']);angle={};domains={};variables=[];inputs={str(p):sha(p)for p in[sourcepath,oldsystempath,representation,rootmode,S/'e037-direct-real-model-v1/exact-query-constraint-records.json',S/'e037-surrounding-boundary-contract-v1/report.json']}
for frame in frames:
 for component in range(2):
  name=f'u_{frame}_{component}';angle[name]=list(map(F,box[frame,component]));saved_half=charts[frame]['half_angle_bounds'][component]
  if name in core:domains[name]=core[name];kind='SAVED_local_necessary_clipped_domain_BY_HASH'
  else:domains[name]=[F(math.nextafter(saved_half[0]-1e-14,-math.inf)),F(math.nextafter(saved_half[1]+1e-14,math.inf))];kind='SAVED_region2_halfangle_endpoints_with_declared_1e14_outward_guard'
  row=dict(name=name,frame=frame,component=component,angle_bounds=[h.rec(v)for v in angle[name]],half_angle_bounds=[h.rec(v)for v in domains[name]],origin=kind,endpoint_guard_NECESSARY_ONLY_not_acceptance_tolerance=True,finite_chart=bool(-3<angle[name][0]<=angle[name][1]<3));emit(dict(kind='AFTER_NEW_surrounding_variable_RETURN',RETURN=row));variables.append(row)
fixed_outside=[]
for frame in range(len(fixed)):
 if frame in frames:continue
 passed=bool(((fixed[frame]>=box[frame,:,0])&(fixed[frame]<=box[frame,:,1])).all());record=dict(frame=frame,commands=fixed[frame],inside_inherited_region2=passed,restricted_construction_NOT_measured_anchor=True);emit(dict(kind='AFTER_NEW_fixed_outside_region_binding_RETURN',RETURN=record));fixed_outside.append(record)
assert all(v['inside_inherited_region2']for v in fixed_outside)
URDF=ET.parse(FROZEN/'robot/urdfs/g1/dex3_right.urdf').getroot();limit={r.attrib['name']:r.find('limit')for r in URDF.findall('joint')};velocity=[F(float(limit[n].attrib['velocity']))for n in manifest['joint_names']['right'][:2]];requests=boundary.requests(original,fixed,timestamps,frames,velocity,emit);assert len(requests)==122;records=[];newcount=reused=0;unresolved=[]
def offer(identifier,names,co,sense,origin):
 co={k:v for k,v in co.items()if v};record=dict(id=identifier,scope=names,coefficients=upper.polyrec(co),sense=sense,origin=origin);emit(dict(kind='BEFORE_NEW_surrounding_query_row',RETURN=record));record['SMT_expression']='('+sense+' '+encoder.polynomial(co,names)+' 0)';emit(dict(kind='AFTER_NEW_surrounding_query_row_RETURN',RETURN=record));records.append(record)
for variable in variables:
 name=variable['name'];lo,hi=domains[name];offer(name+'_lower',[name],{(1,):F(1),(0,):-lo},'>=',dict(variable_binding=variable));offer(name+'_upper',[name],{(1,):F(-1),(0,):hi},'>=',dict(variable_binding=variable))
for index,request in enumerate(requests):
 sig=boundary.signature(request);emit(dict(kind='BEFORE_NEW_surrounding_temporal_request_binding',index=index,request=request))
 if sig in oldreq:
  old=oldreq[sig];assert request['ORIGINAL_lower_rad']==old['request']['ORIGINAL_lower_rad']and request['ORIGINAL_upper_rad']==old['request']['ORIGINAL_upper_rad']and not request['fixed_outside_terms'];reused+=1
  record=dict(index=index,request=request,reused_old_request_id=old['request']['id'],old_system_sha256=sha(oldsystempath),polynomial_NOT_regenerated=True)
  for side in['lower','upper']:
   saved=oldlower[(old['request']['id'],side)];assert saved['scope']==request['variable_names'];co=upper.polydec(saved['coefficients']);offer(f'temporal_{index}_{side}',saved['scope'],co,saved['sense'],dict(reused_old_query_row_id=saved['id'],old_query_records_sha256=sha(S/'e037-direct-real-model-v1/exact-query-constraint-records.json'),old_request_id=old['request']['id'],no_old_polynomial_or_bound_recomputed=True))
 else:
  newcount+=1;result=polynomial.build(request,angle,domains,lambda v:emit(dict(request_index=index,**v)));record=dict(index=index,request=request,NEW_polynomial_constraint=result);save(f'NEW-temporal-request-{index:03d}.json',record)
  if result['angle_branch']!='certified_finite_angle_branch':unresolved.append(dict(index=index,reason=result['angle_branch']));continue
  for side in result['boundaries']:
   co,degrees,error=upper.upper_polynomial(side['coefficients'],[domains[n]for n in request['variable_names']],lambda v:emit(dict(request_index=index,boundary=side['name'],**v)));G=h.unrec(error['outward_uniform_guard']);zero=(0,)*len(request['variable_names']);co[zero]=co.get(zero,F(0))-2*G;emit(dict(kind='AFTER_NEW_sufficient_temporal_lower_RETURN',request_index=index,boundary=side['name'],lower_polynomial=upper.polyrec(co),binding=error))
   if any(v['rounded_center_zero_is_NOT_structural_zero']for v in error['coefficient_records']):unresolved.append(dict(index=index,boundary=side['name'],reason='nonzero_original_coefficient_rounded_zero_NOT_silently_erased'))
   origin=dict(new_request_record=f'NEW-temporal-request-{index:03d}.json',boundary=side['name'],uniform_upper_binding=error,exact_relation='lower=h-2G=P-G <= saved coefficient g',numerical_sufficient_model_NOT_original_candidate_tolerance=True);offer(f'temporal_{index}_{side["name"]}',request['variable_names'],co,'>=',origin)
 save(f'temporal-request-binding-{index:03d}.json',record)
assert reused==22 and newcount==100
for row in source['polynomials']:
 frame=row['frame'];names=[f'u_{frame}_0',f'u_{frame}_1'];saved=row['polynomial'];assert saved['positive_denominator_for_finite_chart']and saved['positive_depth_inequality_separate']and saved['minimum_depth']==.01
 for kind,key,sense in [('image','normalized_source_polynomial','<='),('strict_depth','normalized_depth_polynomial','>')]:
  co={(i,j):F(v)for i,r in enumerate(saved[key])for j,v in enumerate(r)if v!=0};offer(f'{kind}_{frame}',names,co,sense,dict(saved_source_polynomials_sha256=sha(sourcepath),frame=frame,coefficient_key=key,source_record_COPIED=row,ORIGINAL_image_tolerance_px=1e-6,depth_threshold_m=.01,positive_finite_chart_denominator_lower=1,NOT_source_accuracy=True))
assert len(variables)==38 and len(records)==358;query=encoder.query([v['name']for v in variables],[r['SMT_expression']for r in records],True);query_path=O/'ONE-19clock-source-time-query.smt2';query_path.write_text(query);emit(dict(kind='AFTER_NEW_full_surrounding_query_RETURN',rows=len(records),variables=38,query_sha256=sha(query_path),unresolved_cases=unresolved));save('exact-query-constraint-records.json',records);save('fullclock-temporal-request-records.json',requests);save('surrounding-model-binding.json',dict(schema='v2d.surrounding-model-binding.v1',input_sha256=inputs,frames=frames,variables=variables,domains={k:[h.rec(v)for v in a]for k,a in domains.items()},source_polynomials_copied=19,original_time_requests=122,old_time_requests_reused=22,NEW_time_requests=100,query_sha256=sha(query_path),constraint_records_sha256=sha(O/'exact-query-constraint-records.json'),fixed_outside_F_commands_CONSTRUCTION_only=fixed_outside,full_region2_COPIED=region,three_unknown109bands_COPIED=oldsystem['retained_109_implicit_source_evidence_COPIED'],no_local_rational_witness_fixed=True,unresolved=unresolved,no_gate_authority_model_root_profile_or_source_confidence_change=True));J.close();out=dict(schema='v2d.surrounding-clock-model.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),ONE_NEWsurrounding_query_construction=True,continuous_variables=38,source_frames=19,temporal_requests=122,new_temporal_requests=100,old_temporal_requests_reused_by_hash=22,old_temporal_polynomials_NOT_regenerated=44,new_temporal_polynomial_boundaries=200,constraints=358,unresolved_cases=len(unresolved),query_sha256=sha(query_path),new_solver_calls=0,new_native_position_checks=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))
