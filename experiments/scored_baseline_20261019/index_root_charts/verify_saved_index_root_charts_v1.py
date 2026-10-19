"""NEW independent coefficient/tree/Sturm/URDF checks, no producer reruns."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,math,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-index-root-chart-independent-v1';sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from v2d_dex3_flashchord.kinematics import URDFKinematics
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();j=(O/'all-independent-root-chart-identities-certificates-and-URDF-records.jsonl').open('w')
def frac(v):return F(int(v['numerator_hex'],16),int(v['denominator_hex'],16))
def grid(v):return [[frac(z) for z in row] for row in v]
def power(p,u,v):return sum(F(float(z))*u**i*v**k for i,row in enumerate(p) for k,z in enumerate(row))
def e_power(p,u,v):return sum(z*u**i*v**k for i,row in enumerate(p) for k,z in enumerate(row))
def bernstein(B,u,v):
    n=len(B)-1;m=len(B[0])-1
    return sum(B[i][k]*math.comb(n,i)*u**i*(1-u)**(n-i)*math.comb(m,k)*v**k*(1-v)**(m-k) for i in range(n+1) for k in range(m+1))
def peval(p,x):
    value=F(0)
    for v in p[::-1]:value=value*x+v
    return value
def deriv(p):return [i*p[i] for i in range(1,len(p))]
def trim(p):
    p=p.copy()
    while p and p[-1]==0:p.pop()
    return p
def proportional(a,b):
    a=trim(list(map(F,a)));b=trim(list(map(F,b)))
    if not a or not b:return not a and not b
    if len(a)!=len(b):return False
    scale=a[-1]/b[-1]
    return scale>0 and all(v==scale*w for v,w in zip(a,b,strict=True))
def remainder(a,b):
    a=trim(list(map(F,a)));b=trim(list(map(F,b)))
    while a and len(a)>=len(b):
        degree=len(a)-len(b);ratio=a[-1]/b[-1]
        for k,v in enumerate(b):a[degree+k]-=ratio*v
        a=trim(a)
    return a
def side(p,x,direction):
    order=0
    while p:
        value=peval(p,x)
        if value:return (1 if value>0 else -1)*(-1 if direction<0 and order%2 else 1)
        p=deriv(p);order+=1
    return 0
def variation(seq,x,direction):
    values=[side(p,x,direction) for p in seq];values=[v for v in values if v]
    return sum(a!=b for a,b in zip(values,values[1:]))
def verify_sturm(record):
    seq=[[int(v,16) for v in p] for p in record['Sturm_sequence_hex']];p=[int(v,16) for v in record['integer_polynomial_hex']]
    assert proportional(seq[0],p)
    if len(seq)>1:assert proportional(seq[1],deriv(p))
    for i in range(len(seq)-2):assert proportional([-v for v in remainder(seq[i],seq[i+1])],seq[i+2])
    if len(seq)>1:assert not remainder(seq[-2],seq[-1])
    for node in record['all_root_count_nodes']:
        a,b=frac(node['lower']),frac(node['upper']);count=0 if a==b else variation(seq,a,1)-variation(seq,b,-1)
        assert count==node['open_real_root_count']
    assert len(record['roots'])+sum(v['open_real_root_count'] for v in record['unresolved'])==record['closed_interval_distinct_real_root_count']
    return len(record['all_root_count_nodes'])

P=S/'e037-index-critical-projection-v1';pp=P/'source-projection-families-and-sections.json';assert sha(pp)==json.loads((P/'report.json').read_text())['all_outputs_sha256'][pp.name];projection=json.loads(pp.read_text());source=projection['source_record']['polynomial']
B=S/'e037-index-critical-bernstein-v1';bp=B/'critical-chart-partitions-and-unresolved-events.json';assert sha(bp)==json.loads((B/'report.json').read_text())['all_outputs_sha256'][bp.name];trees=json.loads(bp.read_text());before={};audits={}
for line in (B/'all-new-critical-chart-controls-proposals-and-bounds.jsonl').open():
    row=json.loads(line)
    if row['kind']=='BEFORE_NEW_common_zero_chart_node':before[(row['section_id'],row['id'])]=row
    elif row['kind']=='AFTER_NEW_saved_coefficient_identity_audit':audits[row['section_id']]=row
identity_checks=0;subdivision_checks=0;node_checks=0
for section in trees['sections']:
    sid=section['section_id'];nodes={v['id']:v for v in section['nodes']};root=before[(sid,0)];rootB=grid(root['source_Bernstein']);rootD=grid(root['derivative_Bernstein']);E=grid(audits[sid]['exact_source_difference'])
    bounds=[[frac(z) for z in row] for row in root['bounds']];du=bounds[0][1]-bounds[0][0];dv=bounds[1][1]-bounds[1][0];error=frac(audits[sid]['source_error_bound']);derr=frac(audits[sid]['derivative_error_bound'])
    assert sum(abs(z) for row in E for z in row)==error
    assert sum(i*abs(E[i][k]) for i in range(1,len(E)) for k in range(len(E[0])))/du==derr
    # A full exact tensor interpolation grid verifies polynomial identity,
    # rather than repeating the earlier coefficient conversion.
    for a in [F(i,7) for i in range(1,6)]:
        for v in [F(i,11) for i in range(1,6)]:
            u0=bounds[0][0]+du*a;v0=bounds[1][0]+dv*v
            assert power(source['normalized_source_polynomial'],u0,v0)==bernstein(rootB,a,v)+e_power(E,a,v)
            identity_checks+=1
    for a in [F(i,7) for i in range(1,5)]:
        for v in [F(i,11) for i in range(1,6)]:
            u0=bounds[0][0]+du*a;v0=bounds[1][0]+dv*v
            direct=sum(i*F(float(z))*u0**(i-1)*v0**k for i,row in enumerate(source['normalized_source_polynomial']) if i for k,z in enumerate(row))
            Eder=sum(i*E[i][k]*a**(i-1)*v**k for i in range(1,len(E)) for k in range(len(E[0])))/du
            assert direct==bernstein(rootD,a,v)+Eder;identity_checks+=1
    for node in section['nodes']:
        original=before[(sid,node['id'])];sb=grid(original['source_Bernstein']);db=grid(original['derivative_Bernstein']);bb=[[frac(z) for z in row] for row in original['bounds']]
        assert [min(z for row in sb for z in row)-error,max(z for row in sb for z in row)+error]==[frac(z) for z in node['source_interval']]
        assert [min(z for row in db for z in row)-derr,max(z for row in db for z in row)+derr]==[frac(z) for z in node['derivative_interval']]
        if node['classification'].startswith('F_nonzero'):assert frac(node['source_interval'][0])>0 or frac(node['source_interval'][1])<0
        if node['classification'].startswith('Fu_nonzero'):assert frac(node['derivative_interval'][0])>0 or frac(node['derivative_interval'][1])<0
        if node['child_ids']:
            left,right=[before[(sid,k)] for k in node['child_ids']];lb=[[frac(z) for z in row] for row in left['bounds']];rb=[[frac(z) for z in row] for row in right['bounds']];axis=node['depth']%2;mid=sum(bb[axis])/2
            expectedleft=[r.copy() for r in bb];expectedright=[r.copy() for r in bb];expectedleft[axis][1]=mid;expectedright[axis][0]=mid;assert lb==expectedleft and rb==expectedright
            for child,childbounds in [(left,lb),(right,rb)]:
                for key in ['source_Bernstein','derivative_Bernstein']:
                    parentgrid=grid(original[key]);childgrid=grid(child[key]);n=len(childgrid)-1;m=len(childgrid[0])-1
                    for a in [F(i+1,n+2) for i in range(n+1)]:
                        for v in [F(i+1,m+2) for i in range(m+1)]:
                            uv=[a,v];absolute=[childbounds[k][0]+(childbounds[k][1]-childbounds[k][0])*uv[k] for k in range(2)];parentuv=[(absolute[k]-bb[k][0])/(bb[k][1]-bb[k][0]) for k in range(2)]
                            assert bernstein(childgrid,a,v)==bernstein(parentgrid,*parentuv);subdivision_checks+=1
        node_checks+=1
    assert len(nodes)==1+2*sum(bool(v['child_ids']) for v in nodes.values())
    g.journal(j,dict(kind='independent_exact_saved_chart_identity_and_cover_passed',section_id=sid,nodes=len(nodes),source_and_derivative_identity=True))
root_count_nodes=0
for family in projection['families']:
    if family['family_kind']=='Sylvester_resultant_F_Fu':assert family.get('full_v_scope_unresolved');continue
    root_count_nodes+=verify_sturm(family['exact_real_isolation'])
L=S/'e037-index-regular-source-branches-v1';lp=L/'regular-source-cells-and-unknown-bands.json';assert sha(lp)==json.loads((L/'report.json').read_text())['all_outputs_sha256'][lp.name];lift=json.loads(lp.read_text())
for section in lift['sections']:
    for chart in [v['representative_chart'] for v in section['regular_source_cells']]+section['closed_endpoint_charts']:
        root_count_nodes+=verify_sturm(chart['isolation']);v=frac(chart['distal_parameter']);coeff=[int(z,16) for z in chart['integer_polynomial_hex']]
        exact=[sum(F(float(z))*v**k for k,z in enumerate(row)) for row in source['normalized_source_polynomial']];assert proportional(exact,coeff)
    if section['entire_closed_local_source_rectangle_empty_in_saved_coefficient_model']:
        assert not section['unknown_common_zero_distal_bands'] and all(v['representative_chart']['classification']=='source_empty_proximal_section_at_parameter' for v in section['regular_source_cells'])
with np.load(S/'e037-saved-middle-source-component-v1/UNQUALIFIED-middle-source-motion-component.npz') as z:base=z['robot_qpos'].copy()
kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf');witnesschecks=[];camera=c.cameras[109];observed=np.array(projection['source_record']['pixel']);cap=projection['source_record']['ORIGINAL_cap_px']
for index,w in enumerate(lift['isolated_robot_source_witnesses']):
    candidate=base.copy();candidate[109,1,7:9]=w['angles_rad'];cmd=candidate[109,1]
    g.journal(j,dict(kind='BEFORE_NEW_independent_actual_URDF_source_witness',index=index,command=cmd,source=w))
    tip=kin.poses(cmd[:7],dict(zip(c.manifest['joint_names']['right'],cmd[7:],strict=True)))['right_index_tip'][:3,3];C=(tip@c.stage-camera[:3,3])@camera[:3,:3];error=float(np.linalg.norm(c.uv(C[None])[0]-observed));u,v=np.tan(np.array(w['angles_rad'])/2);D=(1+u*u)*(1+v*v)
    predicted=np.array([np.polynomial.polynomial.polyval2d(u,v,row) for row in source['camera_numerators']])/D;closure=float(abs(C-predicted).max());lo,hi=c.bounds(109);position=bool(((cmd[7:9]>=lo[:2]-1e-8)&(cmd[7:9]<=hi[:2]+1e-8)).all());passed=bool(position and error<=cap+1e-6 and closure<1e-8)
    delta=candidate[:,1,7:]-c.aligned[:,1,7:];speed=abs(np.diff(candidate[:,1,7:],axis=0))/np.diff(c.ts)[:,None];limits=np.array([12,12,12,12,6.857,12,12]);motion=dict(component_rad=float(abs(delta).max()),adjacent_added_rad=float(abs(np.diff(delta,axis=0)).max()),second_added_rad=float(abs(np.diff(delta,n=2,axis=0)).max()),maximum_velocity_excess_rad_s=float((speed-limits).max()))
    motionpass=bool(motion['component_rad']<=.25+1e-7 and motion['adjacent_added_rad']<=.05+1e-7 and motion['second_added_rad']<=.05+1e-7 and motion['maximum_velocity_excess_rad_s']<=1e-7)
    np.savez_compressed(O/('UNQUALIFIED-sparse-index-witness-'+str(index)+'.npz'),robot_qpos=candidate)
    record=dict(index=index,source_image_error_px=error,original_source_cap_px=cap,URDF_polynomial_camera_closure_m=closure,original_position_passed=position,isolated_source_URDF_passed=passed,sparse_original_finger_motion=motion,sparse_original_finger_motion_passed=motionpass,whole_clock_source_native_motor_grasp_unqualified=True)
    g.journal(j,dict(kind='AFTER_NEW_independent_actual_URDF_source_witness',record=record));witnesschecks.append(record);assert passed
j.close();(O/'independent-root-chart-verification.json').write_text(json.dumps(g.plain(dict(witness_checks=witnesschecks,local_section_source_exclusions=[v['section_id'] for v in lift['sections'] if v['entire_closed_local_source_rectangle_empty_in_saved_coefficient_model']])),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.independent-index-root-charts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),new_exact_tensor_coefficient_identity_values=identity_checks,new_exact_subdivision_identity_values=subdivision_checks,critical_chart_nodes_independently_checked=node_checks,saved_Sturm_root_count_certificate_nodes_checked=root_count_nodes,
    new_actual_URDF_source_witnesses=len(witnesschecks),all_isolated_source_URDF_checks_passed=all(v['isolated_source_URDF_passed'] for v in witnesschecks),sparse_original_finger_motion_passes=sum(v['sparse_original_finger_motion_passed'] for v in witnesschecks),
    six_conditional_outer_regions_source_rejected_in_saved_model=True,one_conditional_outer_region_remains_unresolved=True,
    no_old_projection_Sturm_isolation_Bernstein_chart_or_source_producer_reexecuted=True,no_missing_resultant_reconstructed=True,
    source_accuracy_and_full_temporal_native_motor_grasp_qualification=False,current_reference_middle_component_and_source_authority_unchanged=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,
    all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
