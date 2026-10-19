"""NEW saved-system checks against independent direct angles and timestamps.

Does not import the temporal algebra producer/helper or rerun source projections.
"""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,itertools,json,math,sys,time
import numpy as np
S=Path(__file__).resolve().parents[1];O=S/'e037-index-source-time-independent-v1';sys.path.insert(0,str(S/'compute'))
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();read=lambda p:json.loads(Path(p).read_text());j=(O/'all-independent-temporal-coefficients-branches-and-direct-angle-checks.jsonl').open('w')
def f(v):return F(int(v['numerator_hex'],16),int(v['denominator_hex'],16))
def series(x):
    # Separate longer rational Taylor series, not the producer enclosure code.
    sine=sum(((-1)**k*x**(2*k+1)/math.factorial(2*k+1)for k in range(48)),F(0));cosine=sum(((-1)**k*x**(2*k)/math.factorial(2*k)for k in range(48)),F(0))
    return sine,cosine,abs(x)**97/math.factorial(97),abs(x)**96/math.factorial(96)
def independent_coefficients(weights):
    result={(0,)*len(weights):complex(1)}
    for axis,k in enumerate(weights):
        factor={p:math.comb(abs(k),p)*(1j*(1 if k>0 else -1))**p for p in range(abs(k)+1)}
        out={}
        for e,z in result.items():
            for degree,w in factor.items():
                a=list(e);a[axis]+=degree;a=tuple(a);out[a]=out.get(a,0)+z*w
        result={e:z for e,z in out.items()if z!=0}
    return {e:(F(int(z.real)),F(int(z.imag)))for e,z in result.items()}
def interval_eval(boundary,point):
    lower=upper=F(0)
    for row in boundary['coefficients']:
        z=math.prod(F(v)**k for v,k in zip(point,row['exponent'],strict=True));a=f(row['coefficient_lower'])*z;b=f(row['coefficient_upper'])*z;lower+=min(a,b);upper+=max(a,b)
    return lower,upper
P=S/'e037-index-source-time-system-v1/coupled-source-time-polynomial-system.json';assert sha(P)==read(P.parent/'report.json')['all_outputs_sha256'][P.name];system=read(P)
source=read(S/'e037-index-source-polynomial-v2/actual-index-polynomials-and-saved-region-charts.json');assert system['full_correlated_outer_region_COPIED']==next(v for v in source['regions']if v['saved_node_id']==2)
for v in system['source_constraints']:assert v['source_record_COPIED_not_regenerated']==next(p for p in source['polynomials']if p['frame']==v['frame'])
with np.load(S/'e037-thumb-ray-independent-validation-v1/frozen/aligned-source-reference.npz')as z:original=z['robot_qpos'][:,1,7:9].copy()
with np.load(S/'e037-thumb-ray-independent-validation-v1/frozen/targets.npz')as z:ts=z['timestamps_s'].copy();frozen=z['robot_qpos'][:,1,7:9].copy()
variables={(v['frame'],v['joint_component']):v for v in system['variables']};rng=np.random.default_rng(19371);coefficient_checks=branch_checks=sample_checks=exact_decision_checks=near_boundary=0;phase_max=0.;frozen_failures=[];retained_rejects=0
for row in system['temporal_constraints']:
    req=row['request'];poly=row['polynomial_constraint'];weights=req['weights'];clocks=req['clocks'];component=req['component'];ranges=[variables[(c,component)]['actual_angle_bounds_rad']for c in clocks];ub=[variables[(c,component)]['half_angle_outer_bounds']for c in clocks]
    combos=[sum(k*F(v)for k,v in zip(weights,corner,strict=True))for corner in itertools.product(*ranges)];assert min(combos)==f(poly['angle_combination_range'][0]) and max(combos)==f(poly['angle_combination_range'][1]);assert all(abs(c-f(end))<6 for c in combos for end in poly['requested_interval']);branch_checks+=1
    pair_coeff=independent_coefficients(weights)
    for boundary in poly['boundaries']:
        x=f(boundary['threshold'])/2;sine,cosine,se,ce=series(x);direction=1 if boundary['name']=='lower'else -1
        saved={tuple(c['exponent']):c for c in boundary['coefficients']};assert set(saved)==set(pair_coeff)
        for e,(real,imag)in pair_coeff.items():
            center=direction*(cosine*imag-sine*real);error=abs(imag)*ce+abs(real)*se;c=saved[e]
            passed=f(c['coefficient_lower'])<=center-error<=center+error<=f(c['coefficient_upper'])
            # At an identically exact zero sin/cos value the enclosures coincide.
            g.journal(j,dict(kind='AFTER_NEW_independent_coefficient_check',constraint=req['id'],boundary=boundary['name'],exponent=e,passed=passed));assert passed;coefficient_checks+=1
    proposals=[list(v)for v in itertools.product(*ranges)]+[[float(rng.uniform(a,b))for a,b in ranges]for unused in range(24)]+[[float(frozen[c,component])for c in clocks]]
    # Explicit accepted/rejected points around the ORIGINAL floating gate.
    for endpoint in['ORIGINAL_lower_rad','ORIGINAL_upper_rad']:
        for side in[-1,1]:
            point=[float(original[c,component])for c in clocks];rest=sum(k*q for k,q in zip(weights[1:],point[1:],strict=True));point[0]=(float(f(req[endpoint]))+side*2e-12-rest)/weights[0]
            if all(a<=q<=b for q,(a,b)in zip(point,ranges,strict=True)):proposals.append(point)
    for n,angles in enumerate(proposals):
        g.journal(j,dict(kind='BEFORE_NEW_direct_angle_test_vector_NOT_candidate',constraint=req['id'],test=n,angles=angles))
        u=[math.tan(q/2)for q in angles];evaluations=[interval_eval(b,u)for b in poly['boundaries']];classification='rejected'if any(b<0 for a,b in evaluations)else('inside'if all(a>=0 for a,b in evaluations)else'UNRESOLVED_boundary')
        if req['kind_of_constraint']=='actual_URDF_velocity':
            dt=ts[clocks[0]]-ts[clocks[1]];direct=float(abs((angles[0]-angles[1])/dt));cap=12+1e-7;passed=direct<=cap;gate_margin=abs(cap-direct)*dt
        else:
            correction=[q-original[c,component]for q,c in zip(angles,clocks,strict=True)];direct=abs(float(sum(k*q for k,q in zip(weights,correction,strict=True))));cap=.05+1e-7;passed=direct<=cap;gate_margin=abs(cap-direct)
        Q=sum(k*q for k,q in zip(weights,angles,strict=True));phasor=math.prod((1+1j*(1 if k>0 else -1)*v)**abs(k)for k,v in zip(weights,u,strict=True));modulus=abs(phasor);closure=abs(phasor/modulus-complex(math.cos(Q/2),math.sin(Q/2)));phase_max=max(phase_max,closure)
        assert modulus>=1-1e-14
        if gate_margin>1e-12:
            consistent=(classification=='inside')==passed and classification!='UNRESOLVED_boundary';exact_decision_checks+=1
        else:consistent=not(passed and classification=='rejected');near_boundary+=1
        outcome=dict(kind='AFTER_NEW_direct_angle_test_vector_NOT_candidate',constraint=req['id'],test=n,angles=angles,half_angles=u,classification=classification,original_float_gate_passed=bool(passed),original_gate_margin_rad=gate_margin,original_float_value=direct,phase_closure=closure,consistent=bool(consistent))
        g.journal(j,outcome);assert consistent and closure<2e-15;sample_checks+=1;retained_rejects+=classification=='rejected'
        if n==len(list(itertools.product(*ranges)))+24 and not passed:frozen_failures.append(req['id'])
j.close();out=dict(schema='v2d.index-source-time-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),
    exact_coefficient_enclosure_checks=coefficient_checks,independent_angle_branch_vertex_checks=branch_checks,new_direct_angle_timestamp_test_vectors=sample_checks,nonboundary_gate_equivalence_checks=exact_decision_checks,near_boundary_cases_retained=near_boundary,
    intentionally_rejected_test_vectors_retained=retained_rejects,maximum_half_angle_phase_closure=phase_max,frozen_local_temporal_gate_failures=frozen_failures,all_independent_checks_passed=True,
    five_saved_source_records_and_full_correlated_region_reused_exactly=True,unknown_109_distal_event_bands_retained=3,
    test_vectors_NOT_source_URDF_native_or_path_candidates=True,no_old_source_projection_root_chart_LP_graph_native_repeated=True,
    no_new_source_motion_path_or_exclusion_qualified=True,current_verified_reference_and_middle_component_unchanged=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,
    all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
