"""NEW implicit source strips on event-free distal cells, frame 109 only.

The failed critical resultant is NOT recovered. Its role is replaced, on the
stated rectangles only, by saved common-zero exclusion charts. No trajectory
or native check is inferred from an isolated source strip.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-index-regular-source-branches-v1';sys.path.insert(0,str(S/'compute'))
import exact_source_projection_v1 as x
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();j=(O/'all-regular-cells-specializations-root-returns-and-source-strips.jsonl').open('w')
def fraction(v):return x.F(int(v['numerator_hex'],16),int(v['denominator_hex'],16))
P=S/'e037-index-critical-projection-v1';pp=P/'source-projection-families-and-sections.json';assert sha(pp)==json.loads((P/'report.json').read_text())['all_outputs_sha256'][pp.name];projection=json.loads(pp.read_text());F=x.bivariate(projection['source_record']['polynomial']['normalized_source_polynomial'])
B=S/'e037-index-critical-bernstein-v1';bp=B/'critical-chart-partitions-and-unresolved-events.json';assert sha(bp)==json.loads((B/'report.json').read_text())['all_outputs_sha256'][bp.name];charts=json.loads(bp.read_text())
# All saved leading/boundary root counts are reused, never executed again.
boundaries=[v for v in projection['families'] if v['family_kind']!='Sylvester_resultant_F_Fu']
assert len(boundaries)==4 and all(v['exact_real_isolation']['closed_interval_distinct_real_root_count']==0 and not v['exact_real_isolation']['unresolved'] for v in boundaries)
g.journal(j,dict(kind='reuse_saved_zero_leading_and_bound_crossing_counts',input_sha256=sha(pp),family_names=[v['name'] for v in boundaries],failed_critical_resultant_bypassed_only_on_saved_common_zero_charts=True))
sections=[];witnesses=[];specializations=0
def specialize(section_id,v,u_bounds,reason):
    global specializations
    p=x.primitive(x.specialize_v(F,v));g.journal(j,dict(kind='BEFORE_NEW_proximal_specialization_root_isolation',section_id=section_id,reason=reason,
        distal_parameter=x.fraction_record(v),proximal_bounds=[x.fraction_record(z) for z in u_bounds],integer_polynomial_hex=[hex(z) for z in p]))
    isolation=x.isolate(p,u_bounds,width=x.F(1,10**10),node_cap=511)
    g.journal(j,dict(kind='AFTER_NEW_proximal_root_isolation_RETURN',section_id=section_id,reason=reason,isolation=isolation))
    complex_roots=x.numerical_complex_roots(p);g.journal(j,dict(kind='AFTER_NEW_proximal_complex_root_RETURN',section_id=section_id,reason=reason,record=complex_roots))
    specializations+=1
    if isolation.get('unresolved') or not p:
        return dict(classification='unresolved_proximal_root_chart',isolation=isolation,numerical_complex_roots=complex_roots)
    bands=sorted([(fraction(r['lower']),fraction(r['upper'])) for r in isolation['roots']]);cursor=u_bounds[0];strips=[]
    intervals=[]
    for lo,hi in bands:
        if cursor<lo:intervals.append([cursor,lo])
        cursor=max(cursor,hi)
    if cursor<u_bounds[1]:intervals.append([cursor,u_bounds[1]])
    for lo,hi in intervals:
        u=(lo+hi)/2;value=x.evaluate(p,u);record=dict(lower=x.fraction_record(lo),upper=x.fraction_record(hi),representative_u=x.fraction_record(u),source_sign=1 if value>0 else -1 if value<0 else 0)
        if value<0:
            angles=[2*x.math.atan(float(u)),2*x.math.atan(float(v))]
            witness=dict(section_id=section_id,distal_parameter=x.fraction_record(v),proximal_parameter=x.fraction_record(u),angles_rad=angles,
                source_polynomial_sign=-1,scope='isolated ROBOT source witness, not measured angle/depth/contact or temporal proof')
            g.journal(j,dict(kind='NEW_source_strip_representative_proposal',record=witness));witnesses.append(witness)
        strips.append(record)
    positive=bool(not isolation['roots'] and strips and all(v['source_sign']>0 for v in strips))
    return dict(classification='source_empty_proximal_section_at_parameter' if positive else 'regular_proximal_source_strip_chart',
        integer_polynomial_hex=[hex(z) for z in p],distal_parameter=x.fraction_record(v),isolation=isolation,numerical_complex_roots=complex_roots,strips=strips,
        no_root_window_promoted_to_finite_width_source_interval=True)

for section,tree in zip(projection['sections'],charts['sections'],strict=True):
    assert section['id']==tree['section_id'];u_bounds=[x.rational(v) for v in section['half_angle_bounds'][0]];v_bounds=[x.rational(v) for v in section['half_angle_bounds'][1]]
    bands=sorted([(fraction(v['bounds'][1][0]),fraction(v['bounds'][1][1])) for v in tree['unresolved_event_boxes']]);merged=[]
    for lo,hi in bands:
        if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
        else:merged.append([lo,hi])
    cursor=v_bounds[0];cells=[]
    for lo,hi in merged:
        if cursor<lo:cells.append([cursor,lo])
        cursor=max(cursor,hi)
    if cursor<v_bounds[1]:cells.append([cursor,v_bounds[1]])
    lifted=[]
    for lower,upper in cells:
        midpoint=(lower+upper)/2
        g.journal(j,dict(kind='BEFORE_NEW_event_free_continuous_source_cell',section_id=section['id'],distal_interval=[x.fraction_record(lower),x.fraction_record(upper)],
            root_count_topology_scope='no F=Fu common zero in the full proximal rectangle on this open distal cell; leading/boundary events absent by saved counts'))
        rootchart=specialize(section['id'],midpoint,u_bounds,'regular_cell_representative')
        row=dict(distal_open_interval=[x.fraction_record(lower),x.fraction_record(upper)],representative_chart=rootchart,
            root_order_continuity_conditional_on_saved_frozen_coefficient_common_zero_cover=True,
            source_accuracy_and_whole_clock_temporal_qualification=False)
        g.journal(j,dict(kind='AFTER_NEW_event_free_continuous_source_cell',section_id=section['id'],record=row));lifted.append(row)
    endpointcharts=[]
    if not merged:
        for value in v_bounds:endpointcharts.append(specialize(section['id'],value,u_bounds,'closed_distal_endpoint'))
    empty=bool(not merged and lifted and all(v['representative_chart']['classification']=='source_empty_proximal_section_at_parameter' for v in lifted)
        and all(v['classification']=='source_empty_proximal_section_at_parameter' for v in endpointcharts))
    result=dict(section_id=section['id'],full_outer_region_ids=section['full_outer_region_ids'],bounds=section['half_angle_bounds'],
        unknown_common_zero_distal_bands=[[x.fraction_record(z) for z in row] for row in merged],regular_source_cells=lifted,closed_endpoint_charts=endpointcharts,
        entire_closed_local_source_rectangle_empty_in_saved_coefficient_model=empty,
        classification='source_empty_local_section' if empty else 'partly_resolved_implicit_source_branch_with_unknown_event_bands',
        no_real_source_or_temporal_or_native_qualification=True)
    g.journal(j,dict(kind='NEW_section_source_branch_summary',record=result));sections.append(result)
j.close();(O/'regular-source-cells-and-unknown-bands.json').write_text(json.dumps(g.plain(dict(frame=109,projection_input_sha256=sha(pp),common_zero_chart_sha256=sha(bp),sections=sections,isolated_robot_source_witnesses=witnesses)),allow_nan=False,indent=2)+'\n')
excluded=[z for v in sections if v['entire_closed_local_source_rectangle_empty_in_saved_coefficient_model'] for z in v['full_outer_region_ids']]
out=dict(schema='v2d.saved-index-regular-source-branches.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),
    frame=109,local_sections=len(sections),new_proximal_specializations=specializations,regular_source_cells=sum(len(v['regular_source_cells']) for v in sections),
    isolated_source_witnesses=len(witnesses),unknown_distal_event_bands=sum(len(v['unknown_common_zero_distal_bands']) for v in sections),
    source_empty_local_sections=[v['section_id'] for v in sections if v['entire_closed_local_source_rectangle_empty_in_saved_coefficient_model']],
    conditional_full_outer_regions_source_rejected=excluded,conditional_full_outer_regions_still_unresolved=[z for v in sections if not v['entire_closed_local_source_rectangle_empty_in_saved_coefficient_model'] for z in v['full_outer_region_ids']],
    source_coefficient_model_NOT_source_accuracy_or_global_task_impossibility=True,failed_resultant_not_reconstructed=True,
    original_boundary_family_Sturm_counts_reused_by_hash_not_reexecuted=True,
    current_verified_reference_middle_component_and_source_authority_unchanged=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,
    all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
