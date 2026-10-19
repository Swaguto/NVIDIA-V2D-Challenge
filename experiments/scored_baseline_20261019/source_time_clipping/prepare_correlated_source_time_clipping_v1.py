"""NEW contractor controls and input binding; old source grids are reused."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import datetime,hashlib,json,math,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-source-time-clipping-binding-v1';sys.path.insert(0,str(S/'compute'))
import correlated_bezier_clipping_v1 as b
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-new-filled-source-clipping-controls-and-input-bindings.jsonl').open('w');controls=[]
def check(name,work):
    g.journal(j,dict(kind='BEFORE_NEW_clipping_control',name=name));v=work();g.journal(j,dict(kind='AFTER_NEW_clipping_control',name=name,result=v));controls.append(dict(name=name,passed=v['passed']));assert v['passed']
def simple():
    grid={(0,0):F(-3,5),(1,0):F(2,5),(0,1):F(2,5),(1,1):F(7,5)};r=b.clip_axis(grid,(1,1),0)
    return dict(passed=not r['empty']and b.dec(r['lower'])==0 and b.dec(r['upper'])==F(3,5),return_value=r)
check('filled_halfplane_interior_kept',simple)
check('strictly_positive_grid_empty',lambda:dict(passed=b.clip_axis({(0,):F(1),(1,):F(2)},(1,),0)['empty']))
def quadratic():
    grid=b.power_to_bernstein({(0,):F(21,100),(1,):F(-1),(2,):F(1)},[(F(0),F(1))],(2,));r=b.clip_axis(grid,(2,),0)
    return dict(passed=b.dec(r['lower'])<=F(3,10)<F(7,10)<=b.dec(r['upper']),return_value=r)
check('quadratic_two_boundaries_not_replaced_by_one_root',quadratic)
def disconnected():
    p={(0,):F(9,256)-F(1,10000),(1,):F(-3,8),(2,):F(11,8),(3,):F(-2),(4,):F(1)};grid=b.power_to_bernstein(p,[(F(0),F(1))],(4,));r=b.clip_axis(grid,(4,),0)
    return dict(passed=b.dec(r['lower'])<=F(1,4)<F(3,4)<=b.dec(r['upper']),return_value=r,interval_is_necessary_hull_not_connected_feasibility=True)
check('disconnected_filled_components_both_retained',disconnected)
def restriction():
    power={(0,0):F(2,7),(1,0):F(-3,2),(2,1):F(5,3),(1,2):F(-1,4)};grid=b.power_to_bernstein(power,[(F(0),F(1))]*2,(2,2));ranges=[(F(1,5),F(3,5)),(F(2,7),F(6,7))];sub=b.restrict(grid,(2,2),ranges);maximum=F(0)
    for point in product([F(0),F(1,3),F(1)],repeat=2):
        q=[lo+(hi-lo)*v for v,(lo,hi)in zip(point,ranges,strict=True)];direct=sum(v*math.prod(x**k for x,k in zip(q,e,strict=True))for e,v in power.items());maximum=max(maximum,abs(direct-b.evaluate(sub,(2,2),point)))
    return dict(passed=maximum==0,exact_error=b.rec(maximum))
check('two_variable_DeCasteljau_restriction_identity',restriction)
def temporal_shape():
    p={(1,2,0):F(-2,3),(0,1,1):F(5,11),(1,0,1):F(3,7)};bounds=[(F(-1,5),F(3,5)),(F(1,7),F(4,7)),(F(-2,9),F(2,9))];grid=b.power_to_bernstein(p,bounds,(1,2,1));errors=[]
    for t in product([F(1,4),F(3,4)],repeat=3):
        q=[a+(z-a)*v for v,(a,z)in zip(t,bounds,strict=True)];errors.append(sum(v*math.prod(x**k for x,k in zip(q,e,strict=True))for e,v in p.items())-b.evaluate(grid,(1,2,1),t))
    return dict(passed=not any(errors),independent_values=len(errors))
check('new_three_clock_temporal_basis_binding',temporal_shape)
def outward():
    v=F(1,3);lo=b.outer_quantize(v,(F(0),F(1)),'lower');hi=b.outer_quantize(v,(F(0),F(1)),'upper');grid={(0,):F(1,3),(1,):F(-7,11)};rounded=b.lower_round(grid)
    return dict(passed=lo<=v<=hi and all(rounded[k]<=z for k,z in grid.items()),bounds=[b.rec(lo),b.rec(hi)])
check('outward_domain_and_lower_coefficient_quantization',outward)
basis,norm=b.lagrange_basis_bound(4)
check('Lagrange_partition_and_Kronecker_values',lambda:dict(passed=all(sum(b.dec(v)*F(i,4)**k for k,v in enumerate(r['power']))==int(i==n)for n,r in enumerate(basis)for i in range(5)),Lebesgue_bound=b.rec(norm)))
P=S/'e037-index-source-time-system-v1/coupled-source-time-polynomial-system.json';assert sha(P)==read(P.parent/'report.json')['all_outputs_sha256'][P.name];system=read(P);variables={v['variable']:v for v in system['variables']};initial={k:[F(v)for v in row['half_angle_outer_bounds']]for k,row in variables.items()};source_by_frame={r['frame']:r for r in system['source_constraints']};factors=[]
old=S/'e037-index-critical-bernstein-v1/all-new-critical-chart-controls-proposals-and-bounds.jsonl';assert sha(old)==read(old.parent/'report.json')['all_outputs_sha256'][old.name];reused=next(v for v in map(json.loads,old.open())if v['kind']=='AFTER_NEW_saved_coefficient_identity_audit'and v['section_id']==0)
for row in system['full_correlated_outer_region_COPIED']['source_charts']:
    frame=row['frame']
    if frame not in range(107,112):continue
    chart=row['chart'];grid={(i,k):F(v)for i,r in enumerate(chart['source_range']['bernstein_coefficients'])for k,v in enumerate(r)};oldbounds=[[F(v)for v in r]for r in chart['half_angle_bounds']];record=dict(frame=frame,old_source_grid_reused_not_regenerated=True,source_input_sha256=sha(P),old_half_angle_bounds=chart['half_angle_bounds'])
    g.journal(j,dict(kind='BEFORE_NEW_reused_source_grid_lower_binding',record=record))
    if frame==109:
        error=b.dec(reused['source_error_bound']);record.update(error_bound=b.rec(error),error_reused_from_saved_critical_audit_by_sha256=sha(old))
    else:
        power=source_by_frame[frame]['source_record_COPIED_not_regenerated']['polynomial']['normalized_source_polynomial'];samples=[];maximum=F(0)
        for x,y in product([F(i,4)for i in range(5)],repeat=2):
            u=oldbounds[0][0]+(oldbounds[0][1]-oldbounds[0][0])*x;v=oldbounds[1][0]+(oldbounds[1][1]-oldbounds[1][0])*y
            difference=sum(F(z)*u**i*v**k for i,r in enumerate(power)for k,z in enumerate(r))-b.evaluate(grid,(4,4),(x,y));maximum=max(maximum,abs(difference));samples.append(dict(local=[b.rec(x),b.rec(y)],exact_difference=b.rec(difference)))
            g.journal(j,dict(kind='AFTER_NEW_source_grid_interpolation_defect',frame=frame,sample=samples[-1]))
        error=maximum*norm**2;record.update(error_bound=b.rec(error),tensor_interpolation_defects=samples,new_error_bound_method='25point exact interpolation and saved Lagrange supnorm, not old source transform or critical chart',basis_proof=basis,Lebesgue_bound=b.rec(norm))
    names=['u_'+str(frame)+'_'+str(i)for i in range(2)];bounds=[initial[k]for k in names];ranges=[((a-c)/(d-c),(z-c)/(d-c))for(a,z),(c,d)in zip(bounds,oldbounds,strict=True)];assert all(0<=a<=z<=1 for a,z in ranges)
    lower=b.lower_round(b.restrict({k:v-error for k,v in grid.items()},(4,4),ranges));factor=dict(id='source_'+str(frame),variables=names,degree=[4,4],initial_bounds=[[b.rec(v)for v in r]for r in bounds],lower_grid=b.encode(lower),binding=record,positive_depth_lower_bound_reused=chart['depth_range']['lower']);assert factor['positive_depth_lower_bound_reused']>0
    g.journal(j,dict(kind='AFTER_NEW_reused_source_grid_lower_binding',factor=factor));factors.append(factor)
for row in system['temporal_constraints']:
    poly=row['polynomial_constraint'];names=row['global_variables'];bounds=[initial[k]for k in names]
    for boundary in poly['boundaries']:
        name='time_'+str(row['request']['id'])+'_'+boundary['name'];g.journal(j,dict(kind='BEFORE_NEW_temporal_lower_grid',id=name,request=row['request']))
        power={tuple(v['exponent']):-b.dec(v['midpoint'])for v in boundary['coefficients']};degree=tuple(max(e[k]for e in power)for k in range(len(names)));error=b.dec(boundary['uniform_coefficient_error']);grid=b.power_to_bernstein(power,bounds,degree);lower=b.lower_round({k:v-error for k,v in grid.items()});factor=dict(id=name,variables=names,degree=list(degree),initial_bounds=[[b.rec(v)for v in r]for r in bounds],lower_grid=b.encode(lower),binding=dict(negated_midpoint_power=[dict(index=list(k),value=b.rec(v))for k,v in sorted(power.items())],coefficient_error=b.rec(error),request=row['request'],sense='lower polynomial<=0 necessary for original half-angle temporal gate'))
        g.journal(j,dict(kind='AFTER_NEW_temporal_lower_grid',factor=factor));factors.append(factor)
j.close();data=dict(schema='v2d.correlated-source-time-clipping-inputs.v1',source_time_system_sha256=sha(P),initial_domains={k:[b.rec(v)for v in r]for k,r in initial.items()},factors=factors,unknown_event_bands_retained=3,filled_source_interiors_NOT_boundary_roots_only=True,all_original_region2_correlations_and_root_hypothesis_retained=True,new_scope='24pass necessary correlated five-clock Bezier clipping; no splits, point seeds, LP, graph or optimizer',maximum_passes=24,coefficient_floor_bits=64,outward_domain_quantization_bits=32)
(O/'correlated-source-time-clipping-inputs.json').write_text(json.dumps(g.plain(data),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.correlated-source-time-clipping-binding-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/correlated_bezier_clipping_v1.py'),new_analytic_controls=controls,all_controls_passed=all(v['passed']for v in controls),old_source_grids_reused=5,new_exact_tensor_defect_values=100,old109coefficient_error_reused_by_hash=True,new_temporal_lower_grids=44,continuous_variables=10,no_source_polynomial_chart_root_tree_or_old_temporal_tests_repeated=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
