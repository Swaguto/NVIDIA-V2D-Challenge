"""NEW mechanics-independent temporal algebra contracts, not old tests rerun."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,math,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-half-angle-time-contract-v1';sys.path.insert(0,str(S/'compute'))
import half_angle_temporal_algebra_v1 as h
import thumb_conic_geometry_v1 as g
start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'all-new-contract-proposals-and-returns.jsonl').open('w');controls=[]
def check(name,work):
    g.journal(j,dict(kind='BEFORE_NEW_temporal_algebra_control',name=name))
    result=work();g.journal(j,dict(kind='AFTER_NEW_temporal_algebra_control',name=name,record=result))
    controls.append(dict(name=name,passed=bool(result['passed']),record=result));assert result['passed']
def coefficients(weights,expected):
    p=h.phasor(weights)
    return dict(passed=p==expected,returned=[dict(exponent=e,real=a,imag=b)for e,(a,b)in p.items()])
check('single_half_angle_phasor',lambda:coefficients([1],{(0,):(1,0),(1,):(0,1)}))
check('difference_without_positive_1_plus_uv_assumption',lambda:coefficients([1,-1],{(0,0):(1,0),(1,0):(0,1),(0,1):(0,-1),(1,1):(1,0)}))
def complex_identity():
    maximum=0;records=[]
    for weights in[[1,-2,1],[-1,2,-1],[2,-1],[0,0]]:
        for point in[[F(1,3)]*len(weights),[F((-1)**k*(k+1),5)for k in range(len(weights))]]:
            p=h.phasor(weights);value=sum(complex(*z)*math.prod(float(v)**k for k,v in zip(e,point,strict=True))for e,z in p.items())
            direct=math.prod((1+1j*(1 if w>0 else -1)*float(v))**abs(w)for w,v in zip(weights,point,strict=True));error=abs(value-direct);maximum=max(maximum,error);records.append(dict(weights=weights,point=list(map(float,point)),closure=error))
    return dict(passed=maximum<2e-15,maximum=maximum,records=records)
check('second_added_repeated_conjugate_and_zero_factor_identities',complex_identity)
def trig_contract():
    records=[]
    for x in[F(-4),F(-1,3),F(0),F(1,10),F(4)]:
        a=h.sincos(x);b=h.sincos(x,40)
        records.append(dict(argument=h.rec(x),passed=all(a[k][0]<=b[k][0]<=b[k][1]<=a[k][1]for k in['sin','cos'])))
    return dict(passed=all(v['passed']for v in records),records=records)
check('rational_Taylor_enclosures_contain_independent_longer_series',trig_contract)
def gate_case(weights,box,L,U,angles,expected):
    ub=[[math.tan(a/2)-1e-14,math.tan(b/2)+1e-14]for a,b in box];r=h.build(weights,box,ub,F(L),F(U));p=[F(math.tan(q/2))for q in angles];decision=h.classify(r,p)
    return dict(passed=decision==expected,decision=decision,expected=expected,angles=angles,weights=weights,direct_combination=sum(k*q for k,q in zip(weights,angles,strict=True)),record=r)
check('adjacent_inside',lambda:gate_case([1,-1],[[-1,1],[-1,1]],-.05,.05,[.42,.4],'inside'))
check('adjacent_reject',lambda:gate_case([1,-1],[[-1,1],[-1,1]],-.05,.05,[.5,.4],'rejected'))
check('second_added_inside',lambda:gate_case([1,-2,1],[[-.6,.6]]*3,-.05,.05,[.2,.1,.02],'inside'))
check('second_added_reject',lambda:gate_case([1,-2,1],[[-.6,.6]]*3,-.05,.05,[.3,.1,.02],'rejected'))
check('negative_tangent_difference_denominator_still_rejects_correctly',lambda:gate_case([1,-1],[[2.5,2.7],[-2.7,-2.5]],-.05,.05,[2.6,-2.6],'rejected'))
check('nonuniform_actual_timestamp_velocity',lambda:gate_case([1,-1],[[-1,1]]*2,-12*.013,12*.013,[.21,.1],'inside'))
def unresolved():
    records=[h.build([1,-1],[[-2.99,2.99]]*2,[[-14,14]]*2,-.05,.05),h.build([1],[[-3.2,3.2]],[[-20,20]],-.1,.1)]
    return dict(passed=all(r['angle_branch'].startswith('UNRESOLVED')for r in records),records=records)
check('branch_cut_and_half_angle_pole_retained_as_unresolved',unresolved)
def strict_errors():
    errors=[]
    for call in[lambda:h.phasor([3]),lambda:h.build([1],[[1,0]],[[0,1]],0,1),lambda:h.build([1],[[0,1]],[[0,1]],1,0),lambda:h.sincos(5),lambda:h.build([1,1],[[0,1]],[[0,1]],0,1)]:
        try:call();errors.append(None)
        except ValueError as exc:errors.append(type(exc).__name__+':'+str(exc))
    return dict(passed=all(errors),errors=errors)
check('strict_input_contracts',strict_errors)
j.close();out=dict(schema='v2d.half-angle-temporal-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/half_angle_temporal_algebra_v1.py'),new_analytic_controls=controls,all_controls_passed=all(v['passed']for v in controls),
    no_old_source_native_projection_chart_graph_or_LP_job_repeated=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
