"""NEW frame-109 algebraic projection, not another source/pose sample bank."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-index-critical-projection-v1';sys.path.insert(0,str(S/'compute'))
import exact_source_projection_v1 as x
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
contract=json.loads((S/'e037-index-source-projection-contract-v1/report.json').read_text())
assert contract['all_controls_passed'] and contract['helper_sha256']==sha(S/'compute/exact_source_projection_v1.py')
source=S/'e037-index-source-polynomial-v2/actual-index-polynomials-and-saved-region-charts.json';assert sha(source)==json.loads((source.parent/'report.json').read_text())['all_outputs_sha256'][source.name]
verified=S/'e037-index-source-polynomial-independent-v1/independently-checked-source-region-records.json';assert sha(verified)==json.loads((verified.parent/'report.json').read_text())['all_outputs_sha256'][verified.name]
saved=json.loads(source.read_text());ids=[v['saved_node_id'] for v in json.loads(verified.read_text()) if not v['numerically_source_empty_frames']];assert len(ids)==7
record=next(v for v in saved['polynomials'] if v['frame']==109);F=x.bivariate(record['polynomial']['normalized_source_polynomial']);Fu=x.derivative_u(F)
sections={};mapping={}
for region in saved['regions']:
    if region['saved_node_id'] not in ids:continue
    chart=next(v['chart'] for v in region['source_charts'] if v['frame']==109)
    assert chart['depth_range']['lower']>0
    key=tuple(float(v).hex() for row in chart['half_angle_bounds'] for v in row)
    if key not in sections:sections[key]=dict(id=len(sections),half_angle_bounds=chart['half_angle_bounds'],angle_bounds=chart['angle_bounds'],positive_depth_bound_reused_by_hash=sha(source),saved_positive_depth_lower_bound=chart['depth_range']['lower'],full_outer_region_ids=[])
    sections[key]['full_outer_region_ids'].append(region['saved_node_id']);mapping[str(region['saved_node_id'])]=sections[key]['id']
sections=list(sections.values());assert len(sections)==3
vlo=min(v['half_angle_bounds'][1][0] for v in sections);vhi=max(v['half_angle_bounds'][1][1] for v in sections);bounds=[x.rational(vlo),x.rational(vhi)]
j=(O/'all-exact-projection-proposals-families-roots-and-unresolved-records.jsonl').open('w')
g.journal(j,dict(kind='NEW_projection_scope',frame=109,source_file_sha256=sha(source),independent_input_sha256=sha(verified),sections=sections,region_to_section=mapping,
    isolation_bounding_v_interval=[vlo,vhi],bounding_interval_NOT_convexified_feasible_union=True,source_depth_bounds_reused_not_recomputed=True))
families=[]
def offer(name,producer,kind,extra=None):
    g.journal(j,dict(kind='BEFORE_NEW_exact_projection_family',name=name,family_kind=kind,extra=extra))
    try:
        result=producer();p=result['coefficients']
        isolation=x.isolate(p,bounds,width=x.F(1,10**10),node_cap=2047)
        complex_roots=x.numerical_complex_roots(p)
        out=dict(name=name,family_kind=kind,projection_classification=result.get('classification'),
            integer_coefficients_hex=[hex(v) for v in p],degree=len(p)-1,maximum_coefficient_bits=max((abs(v).bit_length() for v in p),default=0),
            exact_real_isolation=isolation,numerical_complex_roots=complex_roots,extra=extra,
            exact_model_scope='exact saved binary source-polynomial coefficients, not upstream camera or source truth')
    except Exception as exc:
        out=dict(name=name,family_kind=kind,unresolved_exact_projection_error=type(exc).__name__,error_message=str(exc),extra=extra,
            full_v_scope_unresolved=True)
    g.journal(j,dict(kind='AFTER_NEW_exact_projection_family',record=out));families.append(out)
offer('source_multiple_proximal_root',lambda:x.resultant(F,Fu),'Sylvester_resultant_F_Fu')
offer('source_leading_proximal_coefficient',lambda:dict(coefficients=x.primitive(F[-1]),classification='saved_leading_coefficient'),'proximal_degree_drop')
boundaries={}
for section in sections:
    for side,u in zip(['lower','upper'],section['half_angle_bounds'][0],strict=True):
        key=float(u).hex()
        boundaries.setdefault(key,dict(u=u,section_boundaries=[]))['section_boundaries'].append(dict(section_id=section['id'],side=side))
for i,boundary in enumerate(boundaries.values()):
    offer('source_proximal_boundary_'+str(i),lambda u=boundary['u']:dict(coefficients=x.primitive(x.specialize_u(F,x.rational(u))),classification='exact_saved_proximal_bound_specialization'),
        'source_crosses_section_proximal_boundary',boundary)
j.close()
output=dict(frame=109,source_record=record,source_polynomial_input_sha256=sha(source),sections=sections,full_region_to_local_section=mapping,
    projection_v_bounds=[x.fraction_record(v) for v in bounds],families=families,
    reused_depth_certificate_scope='numerical saved Bernstein depth bounds stay positive; not regenerated or formal source truth')
(O/'source-projection-families-and-sections.json').write_text(json.dumps(g.plain(output),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.saved-index-critical-projection.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/exact_source_projection_v1.py'),
    frame=109,full_outer_regions_preserved=7,distinct_local_source_rectangles=3,projection_families=len(families),
    family_degrees={v['name']:v.get('degree') for v in families},
    exact_arithmetic_failures=[v['name'] for v in families if v.get('full_v_scope_unresolved')],
    unresolved_root_isolation_families=[v['name'] for v in families if v.get('exact_real_isolation',{}).get('unresolved')],
    identically_zero_or_degree_degenerate_families=[v['name'] for v in families if v.get('degree',0)<0],
    complete_isolation_for_saved_coefficient_model=all(not v.get('full_v_scope_unresolved') and not v.get('exact_real_isolation',{}).get('unresolved') and v.get('degree',-1)>=0 for v in families),
    no_old_source_polynomial_Bernstein_Taylor_LP_partition_or_graph_reexecution=True,
    all_complex_roots_and_endpoint_degree_drop_multiple_and_unknown_cases_retained=True,
    no_native_position_physics_GPU_PPO_or_source_target_mutation=True,current_verified_reference_and_middle_component_unchanged=True,
    no_continuous_or_global_source_motion_or_physical_task_qualification=True,
    all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
