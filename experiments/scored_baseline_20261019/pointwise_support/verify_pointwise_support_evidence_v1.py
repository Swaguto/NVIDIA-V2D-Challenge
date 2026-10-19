"""Independent exact saved support upper bounds, maps and square witnesses.

No old polynomial generation, trig, matrix audit, solver, or physical candidate.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from collections import Counter
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-pointwise-support-independent-v1';P=S/'e037-pointwise-support-diagnosis-v1';sys.path.insert(0,str(S/'compute'))
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest();read=lambda q:json.loads(Path(q).read_text());j=(O/'all-independent-support-error-affine-map-and-square-certificates.jsonl').open('w');emit=lambda v:g.journal(j,v);start=time.monotonic()
report=read(P/'report.json')
for name,digest in report['all_outputs_sha256'].items():assert sha(P/name)==digest
systempath=S/'e037-index-source-time-system-v1/coupled-source-time-polynomial-system.json';system=read(systempath);assert sha(systempath)==report['binding']['source_time_system_sha256'];model=read(S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json');weights=list(map(F,read(S/'e037-direct-conditional-SDP-v1/single-direct-SDP-solver-RETURN.json')['x']));domains={k:list(map(c.dec,v))for k,v in read(S/'e037-common-moment-model-v1/common-degree-factor-representations.json')['domains'].items()};entries={}
for col,v in enumerate(model['columns']):entries.setdefault(tuple(v['scope']),{})[tuple(v['index'])]=col
expected_intervals={}
for temporal in system['temporal_constraints']:
    req=temporal['request'];poly=temporal['polynomial_constraint'];assert poly['angle_branch']=='certified_finite_angle_branch'and poly['squared_phasor_modulus_denominator_lower_bound']==1
    for boundary in poly['boundaries']:expected_intervals['time_'+str(req['id'])+'_'+boundary['name']]=boundary['coefficients']
for source in system['source_constraints']:
    poly=source['source_record_COPIED_not_regenerated']['polynomial'];expected_intervals['depth_'+str(source['frame'])]=[dict(exponent=[i,k],coefficient_lower=c.rec(F(v)),coefficient_upper=c.rec(F(v)))for i,row in enumerate(poly['normalized_depth_polynomial'])for k,v in enumerate(row)]
def add(a,b):
    out=dict(a)
    for k,v in b.items():out[k]=out.get(k,F(0))+v
    return {k:v for k,v in out.items()if v}
def mul(a,b):
    out={}
    for x,v in a.items():
        for y,w in b.items():
            z=tuple(i+k for i,k in zip(x,y));out[z]=out.get(z,F(0))+v*w
    return {k:v for k,v in out.items()if v}
def falling(k,r):
    if r>k:return F(0)
    a=b=1
    for t in range(r):a*=k-t;b*=4-t
    return F(a,b)
def coefficients(poly,table):
    out={}
    for index,col in table.items():
        value=F(0)
        for power,v in poly.items():
            t=v
            for k,r in zip(index,power):t*=falling(k,r)
            value+=t
        if value:out[col]=value
    return out
def decpoly(rows):return {tuple(v['power']):c.dec(v['value'])for v in rows}
coefficient_errors=0;shiftmaps=0;coefficient_values=0;matrix_entries=0;PSD=0;negative=0;negative_with_scalar=0;congruence_entries=0;tiny_kept=0;support_certificates=[]
for file in sorted(P.glob('factor-*-support-localizer-record.json')):
    record=read(file);case=record['case'];name=case['id'];scope=record['table_scope'];dim=len(scope);table=entries[tuple(scope)];bounds=[domains[v]for v in case['variables']];assert record['original_half_angle_bounds']==[[c.rec(v)for v in b]for b in bounds]and case['intervals']==expected_intervals[name]
    emit(dict(kind='BEFORE_NEW_independent_pointwise_support_factor',id=name,kind_of_support=case['kind']))
    upper=record['uniform_upper_binding'];nominal={};error=F(0);structural=[0]*dim
    for saved,co in zip(case['intervals'],upper['coefficient_records'],strict=True):
        power=tuple(saved['exponent']);lo=c.dec(saved['coefficient_lower']);hi=c.dec(saved['coefficient_upper']);z=c.dec(co['dyadic_center']);e=max(abs(z-lo),abs(z-hi));factor=F(1);assert lo<=hi and z*(1<<96)==int(z*(1<<96))
        for axis,(k,(a,b))in enumerate(zip(power,bounds)):
            factor*=max(abs(a),abs(b))**k
            if lo or hi:structural[axis]=max(structural[axis],k)
        assert e==c.dec(co['coefficient_error'])and factor==c.dec(co['domain_absolute_monomial_bound'])and (lo!=0 or hi!=0)==co['structural_nonzero'];error+=e*factor
        if z:nominal[power]=z
        if z==0 and(lo or hi):assert co['rounded_center_zero_is_NOT_structural_zero']and e>0;tiny_kept+=1
        coefficient_errors+=1
    guard=c.dec(upper['outward_uniform_guard']);assert error==c.dec(upper['exact_uniform_error'])and guard>=error and guard*(1<<96)==int(guard*(1<<96))and upper['structural_degrees']==structural
    zero=(0,)*dim;nominal[zero]=nominal.get(zero,F(0))+guard;assert nominal==decpoly(upper['upper_polynomial']);pulled={}
    # Independent repeated polynomial multiplication, not binomial expansion.
    for power,z in nominal.items():
        term={zero:z}
        for axis,(exponent,(a,b))in enumerate(zip(power,bounds)):
            unit=[0]*dim;unit[axis]=1;linear={zero:a,tuple(unit):b-a}
            for _ in range(exponent):term=mul(term,linear)
        pulled=add(pulled,term)
    ordered={tuple(power[case['variables'].index(v)]for v in scope):z for power,z in pulled.items()};assert ordered==decpoly(record['normalized_upper_polynomial']);degrees=[structural[case['variables'].index(v)]for v in scope];assert degrees==record['structural_degrees']
    basis=list(map(tuple,record['multiplier_basis']));expected_basis=sorted((a for a in product(range(2),repeat=dim)if all(r+2*k<=4 for r,k in zip(degrees,a))),key=lambda a:(sum(a),a));assert basis==expected_basis and len(basis)==(8 if dim==3 else 4)
    emit(dict(kind='AFTER_NEW_independent_support_upper_affine_and_capacity_RETURN',id=name,exact_error=c.rec(error),guard=c.rec(guard),structural_degrees=degrees,basis_size=len(basis),no_original_tolerance_change=True))
    values={};saved_maps={}
    for row in record['unique_shift_maps']:
        shift=tuple(row['shift']);shifted={tuple(x+k for x,k in zip(power,shift)):z for power,z in ordered.items()};assert all(all(k<=4 for k in power)for power in shifted);co=coefficients(shifted,table);stored={v['column']:c.dec(v['value'])for v in row['coefficients']};assert co==stored and hashlib.sha256(json.dumps(row['coefficients'],separators=(',',':')).encode()).hexdigest()==row['sha256'];value=sum(z*weights[col]for col,z in co.items());values[shift]=value;saved_maps[row['sha256']]=(shift,co);shiftmaps+=1;coefficient_values+=len(co);emit(dict(kind='AFTER_NEW_independent_support_shift_map_RETURN',id=name,shift=shift,nonzero_coefficients=len(co),exact_value=c.rec(value)))
    n=len(basis);M=[[c.dec(v)for v in row]for row in record['matrix']]
    for entry in record['matrix_entries']:
        i,k=entry['i'],entry['j'];shift=tuple(x+y for x,y in zip(basis[i],basis[k]));assert list(shift)==entry['shift']and saved_maps[entry['coefficient_map_sha256']][0]==shift and c.dec(entry['actual_entry'])==M[i][k]==values[shift];matrix_entries+=1
    cert=record['certificate']
    if cert['classification']=='exact_mixed_matrix_PSD':
        B=[[c.dec(v)for v in col]for col in cert['basis_columns']];D=list(map(c.dec,cert['diagonal']));assert len(B)==n and all(B[i][i]==1 and all(B[i][k]==0 for k in range(i+1,n))for i in range(n))and all(v>=0 for v in D)
        for i in range(n):
            for k in range(n):assert sum(B[i][a]*M[a][b]*B[k][b]for a in range(n)for b in range(n))==(D[i]if i==k else 0);congruence_entries+=1
        PSD+=1;emit(dict(kind='AFTER_NEW_independent_support_PSD_congruence_RETURN',id=name,rank=sum(v>0 for v in D),dimension=n))
    else:
        raw=list(map(c.dec,cert['vector']));rawvalue=sum(raw[i]*M[i][k]*raw[k]for i in range(n)for k in range(n));assert rawvalue==c.dec(cert['expectation'])<0;witness=record['negative_support_witness'];v=list(map(c.dec,witness['certificate_vector']));scale=max(map(abs,raw));assert v==[z/scale for z in raw];q={a:z for a,z in zip(basis,v)};q2=mul(q,q);weighted=mul(ordered,q2);assert q2==decpoly(witness['square_polynomial'])and weighted==decpoly(witness['support_times_square']);co=coefficients(weighted,table);stored={z['column']:c.dec(z['value'])for z in witness['expectation_coefficients']};expectation=sum(z*weights[col]for col,z in co.items());norm=max(map(abs,co.values()));distance=-expectation/norm
        assert co==stored and expectation==c.dec(witness['actual_exact_expectation'])==rawvalue/scale**2<0 and norm==c.dec(witness['coefficient_linf'])and distance==c.dec(witness['necessary_l1_weight_change_to_remove_THIS_witness'])>0;negative+=1;negative_with_scalar+=M[0][0]>=0
        audit=dict(id=name,kind=case['kind'],dimension=n,actual_scalar=c.rec(M[0][0]),exact_support_square=c.rec(expectation),necessary_l1_change=c.rec(distance),nonnegative_on_declared_support_by_exact_square_identity=True,no_source_time_family_exclusion=True);support_certificates.append(audit);emit(dict(kind='AFTER_NEW_independent_negative_support_square_RETURN',record=audit))
unavailable=read(P/'unavailable-nonconstant-source-localizers.json');assert len(unavailable)==5
for row in unavailable:
    co=row['source_polynomial_COPIED'];degrees=[max(i for i,r in enumerate(co)if any(z!=0 for z in r)),max(k for r in co for k,z in enumerate(r)if z!=0)];assert degrees==row['structural_degrees']==[4,4]and row['available_constant_basis']==[[0,0]]and len(row['rejected_nonconstant_basis'])==3
    for q in row['rejected_nonconstant_basis']:assert q['required_per_axis']==[r+2*k for r,k in zip(degrees,q['exponent'])]and max(q['required_per_axis'])>4
    emit(dict(kind='AFTER_NEW_independent_unavailable_source_support_RETURN',id=row['id'],structural_degrees=degrees,nonconstant_powers_unavailable=True,not_source_or_task_exclusion=True))
assert negative==report['negative_support_matrices']==25 and PSD==report['PSD_support_matrices']==24 and negative_with_scalar==25 and shiftmaps==657 and coefficient_values==26064 and matrix_entries==1360
j.close();out=dict(schema='v2d.pointwise-support-independent-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),all_independent_checks_passed=True,coefficient_enclosure_and_error_maps=coefficient_errors,tiny_nonzero_structural_terms_retained=tiny_kept,exact_affine_support_maps=49,exact_shift_maps=shiftmaps,exact_nonzero_map_coefficients=coefficient_values,matrix_entry_identities=matrix_entries,PSD_support_matrices=PSD,negative_support_matrices=negative,negative_by_type=dict(Counter(v['kind']for v in support_certificates)),negative_supports_with_nonnegative_scalar_expectation=negative_with_scalar,positive_congruence_entry_identities=congruence_entries,exact_negative_support_square_certificates=len(support_certificates),unavailable_source_nonconstant_supports=5,negative_support_certificates=support_certificates,original_table_joint_PSD_and_equality_residuals_NOT_reclassified=True,actual_weights_NOT_clipped_normalized_or_repaired=True,necessary_coefficient_guard_NOT_original_candidate_tolerance=True,conditional_saved_coefficient_model_NOT_upstream_camera_or_source_truth=True,no_old_polynomial_algebra_audit_or_solver_reexecuted=True,no_family_exclusion_or_candidate_qualification=True,classification='saved_table_rejected_by25exact_temporal_support_square_witnesses',new_solver_calls=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={q.name:sha(q)for q in O.iterdir()if q.is_file()and q.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in['all_outputs_sha256','negative_support_certificates']}))
