"""NEW F=Fu=0 exclusion charts; no failed resultant is reconstructed."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-index-critical-bernstein-v1';sys.path.insert(0,str(S/'compute'))
import saved_bernstein_critical_v1 as b
import thumb_conic_geometry_v1 as g
F=b.F;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();j=(O/'all-new-critical-chart-controls-proposals-and-bounds.jsonl').open('w');controls=[]
def require(name,passed,evidence):
    g.journal(j,dict(kind='new_saved_chart_exact_control',name=name,passed=bool(passed),evidence=evidence));controls.append(dict(name=name,passed=bool(passed)));assert passed,name
# (u+v)^2 on [0,1]^2, whose tensor Bernstein coefficients are independent
# closed-form values. A deliberate constant perturbation tests error retention.
p=[[0,0,1],[0,2,0],[1,0,0]];B=[[0,0,1],[0,F(1,2),2],[1,2,4]]
audit=b.audit_saved_chart(p,B,[[0,1],[0,1]])
require('independent_exact_basis_identity',audit['source_error']==0 and audit['derivative_error']==0,dict(error=b.encode_fraction(audit['source_error'])))
passed=True
for u,v in [(F(1,7),F(2,9)),(F(2,3),F(1,5))]:passed &= b.evaluate_bernstein(audit['derivative_coefficients'],u,v)==2*(u+v)
require('proximal_derivative_has_physical_coordinate_scale',passed,dict(derivative=b.encode_grid(audit['derivative_coefficients'])))
bad=[row.copy() for row in B];bad[0][0]+=F(1,1000);difference=b.audit_saved_chart(p,bad,[[0,1],[0,1]])
require('saved_coefficient_error_not_deleted',difference['source_error']>0 and difference['derivative_error']>0,dict(error=b.encode_fraction(difference['source_error'])))
passed=True
for axis in [0,1]:
    left,right=b.split(B,axis)
    for x,y in [(F(1,7),F(2,9)),(F(2,3),F(1,5))]:
        args=[x,y];args[axis]/=2;passed &= b.evaluate_bernstein(left,x,y)==b.evaluate_bernstein(B,*args)
        args=[x,y];args[axis]=(args[axis]+1)/2;passed &= b.evaluate_bernstein(right,x,y)==b.evaluate_bernstein(B,*args)
require('exact_De_Casteljau_both_children_and_axes',passed,dict(no_float_subdivision=True))

P=S/'e037-index-critical-projection-v1';path=P/'source-projection-families-and-sections.json';assert sha(path)==json.loads((P/'report.json').read_text())['all_outputs_sha256'][path.name];projection=json.loads(path.read_text())
old=S/'e037-index-source-polynomial-v2/actual-index-polynomials-and-saved-region-charts.json';assert sha(old)==projection['source_polynomial_input_sha256'];saved=json.loads(old.read_text());source=projection['source_record']['polynomial'];section_results=[]
for section in projection['sections']:
    region=next(v for v in saved['regions'] if v['saved_node_id']==section['full_outer_region_ids'][0]);chart=next(v['chart'] for v in region['source_charts'] if v['frame']==109)
    bounds=[[b.rational(v) for v in row] for row in section['half_angle_bounds']]
    g.journal(j,dict(kind='BEFORE_NEW_saved_coefficient_identity_audit',section_id=section['id'],source_hash=sha(old),section_bounds=[[b.encode_fraction(v) for v in row] for row in bounds]))
    audit=b.audit_saved_chart(source['normalized_source_polynomial'],chart['source_range']['bernstein_coefficients'],bounds)
    depth=b.audit_saved_chart(source['normalized_depth_polynomial'],chart['depth_range']['bernstein_coefficients'],bounds)
    assert depth['source_lower']>0
    g.journal(j,dict(kind='AFTER_NEW_saved_coefficient_identity_audit',section_id=section['id'],exact_source_difference=b.encode_grid(audit['exact_difference_power']),
        source_error_bound=b.encode_fraction(audit['source_error']),derivative_error_bound=b.encode_fraction(audit['derivative_error']),
        exact_depth_difference=b.encode_grid(depth['exact_difference_power']),depth_error_bound=b.encode_fraction(depth['source_error']),exact_frozen_polynomial_depth_lower=b.encode_fraction(depth['source_lower'])))
    queue=[dict(id=0,parent=None,depth=0,bounds=bounds,source=audit['source_coefficients'],derivative=audit['derivative_coefficients'])];nextid=1;nodes=[];uncertain=[]
    while queue:
        node=queue.pop();sb=node['source'];db=node['derivative'];flo=min(v for row in sb for v in row)-audit['source_error'];fhi=max(v for row in sb for v in row)+audit['source_error'];dlo=min(v for row in db for v in row)-audit['derivative_error'];dhi=max(v for row in db for v in row)+audit['derivative_error']
        g.journal(j,dict(kind='BEFORE_NEW_common_zero_chart_node',section_id=section['id'],id=node['id'],parent=node['parent'],depth=node['depth'],bounds=[[b.encode_fraction(v) for v in row] for row in node['bounds']],source_Bernstein=b.encode_grid(sb),derivative_Bernstein=b.encode_grid(db)))
        children=[]
        if flo>0 or fhi<0:classification='F_nonzero_no_source_boundary_tangency'
        elif dlo>0 or dhi<0:classification='Fu_nonzero_no_source_boundary_tangency'
        elif nextid+2<=127:
            classification='split_new_common_zero_chart';axis=node['depth']%2;middle=sum(node['bounds'][axis])/2
            sl,sr=b.split(sb,axis);dl,dr=b.split(db,axis)
            for k,(ss,dd) in enumerate([(sl,dl),(sr,dr)]):
                bb=[row.copy() for row in node['bounds']];bb[axis][1 if k==0 else 0]=middle
                children.append(dict(id=nextid+k,parent=node['id'],depth=node['depth']+1,bounds=bb,source=ss,derivative=dd))
            nextid+=2;queue.extend(reversed(children))
        else:classification='unresolved_possible_common_zero';uncertain.append(node)
        record=dict(id=node['id'],parent=node['parent'],depth=node['depth'],bounds=[[b.encode_fraction(v) for v in row] for row in node['bounds']],classification=classification,
            source_interval=[b.encode_fraction(flo),b.encode_fraction(fhi)],derivative_interval=[b.encode_fraction(dlo),b.encode_fraction(dhi)],child_ids=[v['id'] for v in children])
        nodes.append(record);g.journal(j,dict(kind='AFTER_NEW_common_zero_chart_node',section_id=section['id'],record=record))
    assert len(nodes)<=127
    section_results.append(dict(section_id=section['id'],full_outer_region_ids=section['full_outer_region_ids'],bounds=section['half_angle_bounds'],nodes=nodes,
        unresolved_event_boxes=[dict(id=v['id'],bounds=[[b.encode_fraction(z) for z in row] for row in v['bounds']]) for v in uncertain],
        coefficient_identity_scope='exact rational saved floating power/Bernstein coefficients; no trigonometric/source accuracy proof',
        source_error_bound=b.encode_fraction(audit['source_error']),derivative_error_bound=b.encode_fraction(audit['derivative_error']),positive_depth_lower=b.encode_fraction(depth['source_lower'])))
j.close();(O/'critical-chart-partitions-and-unresolved-events.json').write_text(json.dumps(g.plain(dict(frame=109,projection_input_sha256=sha(path),sections=section_results)),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.saved-index-common-zero-charts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),new_exact_analytic_controls=controls,
    frame=109,sections=len(section_results),nodes_per_section=[len(v['nodes']) for v in section_results],maximum_declared_nodes_per_section=127,
    unresolved_common_zero_boxes_per_section=[len(v['unresolved_event_boxes']) for v in section_results],
    failed_resultant_and_missing_coefficients_not_reconstructed=True,old_Bernstein_transforms_and_root_counts_not_reexecuted=True,
    source_depth_and_derivative_frozen_coefficient_bounds_exactly_audited=True,no_full_source_or_temporal_or_physical_candidate_qualified=True,
    source_authority_current_reference_and_middle_component_unchanged=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,
    all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
