"""NEW continuous five-clock source/time algebra; no point bank or path solver."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,math,sys,time
import numpy as np
S=Path(__file__).resolve().parents[1];O=S/'e037-index-source-time-system-v1';sys.path.insert(0,str(S/'compute'))
import half_angle_temporal_algebra_v1 as h
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();read=lambda p:json.loads(Path(p).read_text());j=(O/'all-source-time-inputs-proposals-and-intermediate-returns.jsonl').open('w')
contract=read(S/'e037-half-angle-time-contract-v1/report.json');assert contract['all_controls_passed'] and contract['helper_sha256']==sha(S/'compute/half_angle_temporal_algebra_v1.py')
P=S/'e037-index-source-polynomial-v2/actual-index-polynomials-and-saved-region-charts.json';assert sha(P)==read(P.parent/'report.json')['all_outputs_sha256'][P.name];saved=read(P);region=next(v for v in saved['regions']if v['saved_node_id']==2)
L=S/'e037-index-regular-source-branches-v1/regular-source-cells-and-unknown-bands.json';assert sha(L)==read(L.parent/'report.json')['all_outputs_sha256'][L.name];lift=read(L);section=next(v for v in lift['sections']if v['full_outer_region_ids']==[2]);assert len(section['unknown_common_zero_distal_bands'])==3
FROZEN=S/'e037-thumb-ray-independent-validation-v1/frozen';assert sha(FROZEN/'manifest.json')=='f438adf3da846ec1c76c31c7b6d6c12b46bea76004aa7745cfada504378da606'
with np.load(FROZEN/'aligned-source-reference.npz')as z:original=z['robot_qpos'][:,1,7:9].copy()
with np.load(FROZEN/'targets.npz')as z:ts=z['timestamps_s'].copy()
frames=list(range(107,112));box=np.array(region['box']);records=[];variables=[];sources=[];arithmetic_guard=F(1,10**14);motion_tolerance=F(float(1e-7));added_cap=F(float(.05))
g.journal(j,dict(kind='BEFORE_NEW_five_clock_input_binding',region_id=2,source_sha256=sha(P),source_cells_sha256=sha(L),original_reference_sha256=sha(FROZEN/'aligned-source-reference.npz'),timestamp_sha256=sha(FROZEN/'targets.npz'),frames=frames,all_full_clock_correlations_preserved=region))
for frame in frames:
    for component in range(2):
        lo,hi=box[frame,component];u=[math.nextafter(math.tan(float(lo)/2)-1e-14,-math.inf),math.nextafter(math.tan(float(hi)/2)+1e-14,math.inf)]
        assert 2*math.atan(u[0])<=lo and 2*math.atan(u[1])>=hi
        row=dict(variable='u_'+str(frame)+'_'+str(component),frame=frame,joint_component=component,actual_angle_bounds_rad=[float(lo),float(hi)],half_angle_outer_bounds=u,
                 half_angle_endpoint_numeric_guard=1e-14,endpoint_guard_is_NECESSARY_ONLY_not_original_candidate_tolerance=True,finite_chart_no_half_angle_pole=bool(-3<lo<=hi<3))
        variables.append(row);g.journal(j,dict(kind='AFTER_NEW_variable_binding',record=row))
    source=next(v for v in saved['polynomials']if v['frame']==frame)
    row=dict(frame=frame,source_record_COPIED_not_regenerated=source,source_input_sha256=sha(P),
             source_inequality='source_polynomial(u_frame_0,u_frame_1)<=0',positive_depth_inequality='depth_polynomial(u_frame_0,u_frame_1)>0',
             finite_denominator='(1+u_frame_0^2)*(1+u_frame_1^2)>=1',original_image_tolerance_px=1e-6,
             source_accuracy_whole_clock_and_native_NOT_established=True)
    sources.append(row);g.journal(j,dict(kind='AFTER_NEW_source_binding_by_hash',record=row))
variable={(r['frame'],r['joint_component']):r for r in variables}
def offer(kind,component,clock,weights,lower,upper,origin):
    angle=[variable[(f,component)]['actual_angle_bounds_rad']for f in clock];ub=[variable[(f,component)]['half_angle_outer_bounds']for f in clock]
    request=dict(id=len(records),kind_of_constraint=kind,component=component,clocks=clock,weights=weights,ORIGINAL_lower_rad=h.rec(lower),ORIGINAL_upper_rad=h.rec(upper),
                 exact_binary_input_origin=origin,necessary_arithmetic_conversion_guard_rad=h.rec(arithmetic_guard),original_acceptance_tolerance_NOT_changed=True)
    g.journal(j,dict(kind='BEFORE_NEW_temporal_polynomial',request=request))
    result=h.build(weights,angle,ub,lower-arithmetic_guard,upper+arithmetic_guard)
    # Save the return before any assertion or work on the next constraint.
    row=dict(request=request,polynomial_constraint=result,global_variables=[variable[(f,component)]['variable']for f in clock])
    g.journal(j,dict(kind='AFTER_NEW_temporal_polynomial_return',record=row));records.append(row)
    assert result['angle_branch']=='certified_finite_angle_branch'
for component in range(2):
    for frame in frames[1:]:
        base=F(float(original[frame,component]))-F(float(original[frame-1,component]));cap=added_cap+motion_tolerance
        offer('original_adjacent_added',component,[frame,frame-1],[1,-1],base-cap,base+cap,dict(original_angles_rad=original[[frame,frame-1],component].tolist(),added_cap_rad=.05,original_motion_tolerance_rad=1e-7))
        dt=F(float(ts[frame]))-F(float(ts[frame-1]));assert dt>0;speed=(F(12)+motion_tolerance)*dt
        offer('actual_URDF_velocity',component,[frame,frame-1],[1,-1],-speed,speed,dict(actual_URDF_velocity_rad_s=12,original_velocity_tolerance_rad_s=1e-7,timestamps_s=ts[[frame,frame-1]].tolist(),positive_delta_time_s=h.rec(dt)))
    for frame in frames[2:]:
        base=F(float(original[frame,component]))-2*F(float(original[frame-1,component]))+F(float(original[frame-2,component]));cap=added_cap+motion_tolerance
        offer('original_second_added',component,[frame,frame-1,frame-2],[1,-2,1],base-cap,base+cap,dict(original_angles_rad=original[[frame,frame-1,frame-2],component].tolist(),added_cap_rad=.05,original_motion_tolerance_rad=1e-7))
j.close();system=dict(schema='v2d.index-five-clock-source-time-polynomial-system.v1',frames=frames,variables=variables,temporal_constraints=records,source_constraints=sources,
    full_correlated_outer_region_COPIED=region,retained_109_implicit_source_evidence_COPIED=section,original_reference_angles_rad=original[frames].tolist(),timestamps_s=ts[frames].tolist(),
    root_and_source_polynomials_remain_the_fixed_UNQUALIFIED_source_wrist_hypothesis=True,isolated_source_poses_NOT_mandatory_nodes=True,
    source_depth_angles_contacts_NOT_measured_truth=True,no_point_path_accepted_or_exclusion_claimed=True,
    uncertainty='Exact rational coefficient enclosures bind binary angle/timestamp inputs. Endpoint tan guard and upstream camera/trig/source coefficients are numerical; no formal global proof.',
    outside_107_to_111_temporal_and_all_source_rows_still_required=True)
(O/'coupled-source-time-polynomial-system.json').write_text(json.dumps(g.plain(system),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.index-source-time-system-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/half_angle_temporal_algebra_v1.py'),
    continuous_frames=frames,continuous_variables=len(variables),coupled_temporal_constraints=len(records),polynomial_inequalities=2*len(records),source_polynomials_copied_by_hash=5,
    all_angle_branches_and_phasor_denominators_certified=True,unknown_109_distal_event_bands_retained=3,all_151_clock_region_correlations_retained=True,
    original_tolerances_unchanged=True,necessary_conversion_guards_NOT_candidate_gates=True,no_old_producer_transform_isolator_graph_LP_or_native_repeated=True,
    new_candidate_or_source_motion_exclusion=False,current_verified_reference_and_middle_component_unchanged=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,
    all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
