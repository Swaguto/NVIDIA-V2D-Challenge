"""Independent blossom and supporting-line certificates; no contractor rerun."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from math import comb,prod
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-source-time-clipping-independent-v1';sys.path.insert(0,str(S/'compute'))
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-independent-blossom-and-supporting-line-certificates.jsonl').open('w')
def dec(r):return F(int(r['numerator_hex'],16),int(r['denominator_hex'],16))
def rec(v):return dict(numerator_hex=hex(v.numerator),denominator_hex=hex(v.denominator),approx=float(v))
def grid(rows):return {tuple(r['index']):dec(r['value'])for r in rows}
def weights(n,k,a,b):
    return [sum(F(comb(n-k,p)*comb(k,i-p))*a**p*(1-a)**(n-k-p)*b**(i-p)*(1-b)**(k-(i-p))for p in range(n-k+1)if 0<=i-p<=k)for i in range(n+1)]
def blossom(original,degree,ranges):
    tables=[{k:weights(n,k,a,b)for k in range(n+1)}for n,(a,b)in zip(degree,ranges,strict=True)]
    return {k:sum(v*prod(tables[a][k[a]][e[a]]for a in range(len(degree)))for e,v in original.items())for k in product(*(range(n+1)for n in degree))}
def verify_floor(exact,saved):
    assert exact.keys()==saved.keys();unit=F(1,1<<64)
    assert all(v-unit<saved[k]<=v for k,v in exact.items())
def support_line(points,target,lower):
    # Independent scalar dual: for z<=0, x-lambda*z>=L (lambda<=0),
    # or x-lambda*z<=U (lambda>=0). No convex-hull clipping or LP is rerun.
    a=None;b=F(0)if lower else None
    if not lower:a=F(0)
    for x,z in points:
        if not z:
            assert (x>=target)if lower else(x<=target);continue
        value=(x-target)/z
        if (z>0)==lower:b=value if b is None else min(b,value)
        else:a=value if a is None else max(a,value)
    assert a is None or b is None or a<=b
    lam=b if lower else a
    assert lam is not None and (lam<=0 if lower else lam>=0)
    assert all((x-lam*z>=target)if lower else(x-lam*z<=target)for x,z in points)
    return lam
I=S/'e037-source-time-clipping-binding-v1/correlated-source-time-clipping-inputs.json';assert sha(I)==read(I.parent/'report.json')['all_outputs_sha256'][I.name];inputs=read(I);source_system=read(S/'e037-index-source-time-system-v1/coupled-source-time-polynomial-system.json');assert inputs['source_time_system_sha256']==sha(S/'e037-index-source-time-system-v1/coupled-source-time-polynomial-system.json')
initial={k:[dec(v)for v in r]for k,r in inputs['initial_domains'].items()};factors={f['id']:f for f in inputs['factors']};input_coefficient_checks=0;new_defect_checks=0
for f in inputs['factors']:
    degree=f['degree'];saved=grid(f['lower_grid']);bounds=[[dec(v)for v in r]for r in f['initial_bounds']];bind=f['binding']
    if f['id'].startswith('source_'):
        frame=bind['frame'];old=next(v['chart']for v in source_system['full_correlated_outer_region_COPIED']['source_charts']if v['frame']==frame);oldgrid={(i,k):F(v)for i,r in enumerate(old['source_range']['bernstein_coefficients'])for k,v in enumerate(r)};oldbounds=[[F(v)for v in r]for r in old['half_angle_bounds']];error=dec(bind['error_bound'])
        if frame!=109:
            basis=bind['basis_proof'];norm=F(0)
            for index,row in enumerate(basis):
                p=[dec(v)for v in row['power']];bb=grid(row['Bernstein']);assert all(sum(v*F(i,4)**k for k,v in enumerate(p))==int(i==index)for i in range(5))
                for k in range(5):assert bb[(k,)]==sum(p[r]*F(comb(k,r),comb(4,r))for r in range(k+1))
                assert dec(row['supremum_upper_bound'])==max(map(abs,bb.values()));norm+=dec(row['supremum_upper_bound'])
            maximum=F(0);power=next(v['source_record_COPIED_not_regenerated']['polynomial']['normalized_source_polynomial']for v in source_system['source_constraints']if v['frame']==frame)
            for sample in bind['tensor_interpolation_defects']:
                x,y=map(dec,sample['local']);u=oldbounds[0][0]+(oldbounds[0][1]-oldbounds[0][0])*x;v=oldbounds[1][0]+(oldbounds[1][1]-oldbounds[1][0])*y
                direct=sum(F(z)*u**i*v**k for i,row in enumerate(power)for k,z in enumerate(row));B=sum(z*comb(4,i)*x**i*(1-x)**(4-i)*comb(4,k)*y**k*(1-y)**(4-k)for(i,k),z in oldgrid.items());difference=direct-B;assert difference==dec(sample['exact_difference']);maximum=max(maximum,abs(difference));new_defect_checks+=1
            assert error==maximum*norm**2
        else:
            oldjournal=S/'e037-index-critical-bernstein-v1/all-new-critical-chart-controls-proposals-and-bounds.jsonl';assert sha(oldjournal)==bind['error_reused_from_saved_critical_audit_by_sha256'];audit=next(v for v in map(json.loads,oldjournal.open())if v['kind']=='AFTER_NEW_saved_coefficient_identity_audit'and v['section_id']==0);assert error==dec(audit['source_error_bound'])
        ranges=[((a-c)/(d-c),(b-c)/(d-c))for(a,b),(c,d)in zip(bounds,oldbounds,strict=True)];exact=blossom({k:v-error for k,v in oldgrid.items()},degree,ranges)
    else:
        power={tuple(v['index']):dec(v['value'])for v in bind['negated_midpoint_power']};error=dec(bind['coefficient_error']);exact={}
        for k in saved:
            value=F(0)
            for e,c in power.items():
                value+=c*prod(sum(F(comb(n,r))*a**(n-r)*(b-a)**r*F(comb(i,r),comb(d,r))for r in range(min(n,i)+1))for n,i,d,(a,b)in zip(e,k,degree,bounds,strict=True))
            exact[k]=value-error
    verify_floor(exact,saved);input_coefficient_checks+=len(saved);g.journal(j,dict(kind='AFTER_NEW_independent_input_lower_grid_certificate',factor=f['id'],coefficients=len(saved),passed=True))
P=S/'e037-correlated-source-time-clipping-v1/all-correlated-domain-proposals-coefficients-clips-and-outcomes.jsonl';assert sha(P)==read(P.parent/'report.json')['all_outputs_sha256'][P.name];domains={k:r.copy()for k,r in initial.items()};states={k:dict(grid=grid(f['lower_grid']),bounds=[[dec(v)for v in r]for r in f['initial_bounds']])for k,f in factors.items()};steps=blossom_coefficients=dual_count=0
for line in P.open():
    row=json.loads(line)
    if row['kind']=='BEFORE_NEW_correlated_factor_contraction':assert {k:list(map(dec,v))for k,v in row['domains'].items()}==domains
    if row['kind']!='AFTER_NEW_correlated_factor_contraction':continue
    factor=factors[row['factor']];names=factor['variables'];degree=factor['degree'];current=[domains[k]for k in names];state=states[row['factor']];saved=grid(row['lower_grid']);ranges=[((a-c)/(d-c),(b-c)/(d-c))for(a,b),(c,d)in zip(current,state['bounds'],strict=True)]
    exact=blossom(state['grid'],degree,ranges);verify_floor(exact,saved);blossom_coefficients+=len(saved)
    assert current==[[dec(v)for v in r]for r in row['grid_bounds']]
    for axis,clip in enumerate(row['axis_clips']):
        assert not clip['empty'];points=[(F(k[axis],degree[axis]),v)for k,v in saved.items()];L,U=dec(clip['lower']),dec(clip['upper']);lower=support_line(points,L,True);upper=support_line(points,U,False)
        g.journal(j,dict(kind='AFTER_NEW_independent_dual_axis_certificate',step=steps,factor=row['factor'],axis=axis,lower=rec(L),upper=rec(U),lower_slope=rec(lower),upper_slope=rec(upper),passed=True));dual_count+=2
        name=names[axis];lo,hi=current[axis];oldlo,oldhi=initial[name];raw=[lo+(hi-lo)*L,lo+(hi-lo)*U];saved_domain=list(map(dec,row['domains'][name]));assert oldlo<=saved_domain[0]<=raw[0]<=raw[1]<=saved_domain[1]<=oldhi
        assert saved_domain[0]>=domains[name][0]and saved_domain[1]<=domains[name][1];unit=(oldhi-oldlo)/2**32;assert raw[0]-saved_domain[0]<unit and saved_domain[1]-raw[1]<unit
    domains={k:list(map(dec,v))for k,v in row['domains'].items()};state.update(grid=saved,bounds=current);steps+=1
result=read(P.parent/'correlated-domain-contraction-result.json');assert domains=={k:list(map(dec,v))for k,v in result['final_domains'].items()}and result['classification']=='UNRESOLVED_declared_24pass_budget';assert steps==1176
j.close();out=dict(schema='v2d.correlated-source-time-independent-certificates.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),new_source_interpolation_defects_independently_checked=new_defect_checks,input_lower_coefficients_verified=input_coefficient_checks,independent_blossom_restriction_coefficients=blossom_coefficients,independent_supporting_line_duals=dual_count,all_contraction_steps_verified=steps,all_checks_passed=True,classification=result['classification'],no_contractor_or_old_source_tree_LP_graph_or_native_rerun=True,no_exclusion_or_candidate_qualified=True,current_components_and_all_original_authority_unchanged=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
