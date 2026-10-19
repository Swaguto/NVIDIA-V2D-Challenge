"""Independent mixed-map, congruence and cross-clique projection certificates.

No solver, source re-fit, root/reference change, old PSD or equality audit.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-mixed-joint-independent-v1';P=S/'e037-saved-mixed-joint-moments-v1';Q=S/'e037-mixed-overlap-certificates-v1';sys.path.insert(0,str(S/'compute'))
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-independent-mixed-maps-congruences-and-overlap-checks.jsonl').open('w');emit=lambda v:g.journal(j,v)
report=read(P/'report.json');overlapreport=read(Q/'report.json')
for folder,rep in [(P,report),(Q,overlapreport)]:
    for name,digest in rep['all_outputs_sha256'].items():assert sha(folder/name)==digest
model=read(S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json');weights=list(map(F,read(S/'e037-direct-conditional-SDP-v1/single-direct-SDP-solver-RETURN.json')['x']));eq=model['equalities'];eqco=[{v['column']:c.dec(v['value'])for v in row['coefficients']}for row in eq];maps=0;nonzero_coefficients=0;matrix_count=0;congruence_identities=0;square_checks=0;variables={};ranks={};records={}
def factor(k,r):
    if r>k:return F(0)
    a=1;b=1
    for v in range(r):a*=k-v;b*=4-v
    return F(a,b)
def add(a,b,scale=F(1)):
    out=dict(a)
    for k,v in b.items():out[k]=out.get(k,F(0))+scale*v
    return {k:v for k,v in out.items()if v}
for path in sorted(P.glob('scope-*-exact-joint-square-record.json')):
    record=read(path);dim=len(record['scope']);entries={tuple(v['index']):v['column']for v in record['columns']};assert set(entries)==set(product(range(5),repeat=dim));y={};emit(dict(kind='BEFORE_NEW_independent_mixed_scope',scope_id=record['scope_id'],scope=record['scope']))
    expected_powers={a for a in product(range(5),repeat=dim)if sum(a)<=4};assert {tuple(v['power'])for v in record['moment_maps']}==expected_powers
    for moment in record['moment_maps']:
        power=tuple(moment['power']);expected={}
        for index,col in entries.items():
            value=F(1)
            for k,r in zip(index,power):value*=factor(k,r)
            if value:expected[col]=value
        assert expected=={v['column']:c.dec(v['value'])for v in moment['coefficients']};actual=sum(v*weights[col]for col,v in expected.items());assert actual==c.dec(moment['actual_moment']);maps+=1;nonzero_coefficients+=len(expected);y[power]=actual;emit(dict(kind='AFTER_NEW_independent_mixed_monomial_identity',scope_id=record['scope_id'],power=power,coefficients=len(expected),actual_moment=c.rec(actual)))
    for matrix in record['matrices']:
        axis=matrix['axis'];basis=list(map(tuple,matrix['basis']));M=[[c.dec(v)for v in row]for row in matrix['matrix']];n=len(M);expected=[]
        for a in basis:
            row=[]
            for b in basis:
                power=[i+k for i,k in zip(a,b)]
                if axis is None:value=y[tuple(power)]
                else:
                    one=power.copy();two=power.copy();one[axis]+=1;two[axis]+=2;value=y[tuple(one)]-y[tuple(two)]
                row.append(value)
            expected.append(row)
        assert M==expected;cert=matrix['certificate']
        if cert['classification']=='exact_mixed_matrix_PSD':
            B=[[c.dec(v)for v in col]for col in cert['basis_columns']];D=list(map(c.dec,cert['diagonal']));assert len(B)==n and all(B[i][i]==1 and all(B[i][k]==0 for k in range(i+1,n))for i in range(n))and all(v>=0 for v in D)
            for i in range(n):
                for k in range(n):
                    assert sum(B[i][a]*M[a][b]*B[k][b]for a in range(n)for b in range(n))==(D[i]if i==k else 0);congruence_identities+=1
            assert sum(v>0 for v in D)==cert['rank'];ranks.setdefault('moment'if axis is None else'localizer',[]).append(cert['rank'])
        else:
            v=list(map(c.dec,cert['vector']));assert sum(v[i]*M[i][k]*v[k]for i in range(n)for k in range(n))==c.dec(cert['expectation'])<0;square_checks+=1
        matrix_count+=1;emit(dict(kind='AFTER_NEW_independent_mixed_matrix_certificate',scope_id=record['scope_id'],name=matrix['name'],classification=cert['classification'],rank=cert.get('rank'),dimension=n,unit_triangular_congruence_invertible=True))
    for axis,name in enumerate(record['scope']):
        if name not in variables:
            zero=(0,)*dim;one=tuple(int(i==axis)for i in range(dim));two=tuple(2*int(i==axis)for i in range(dim));numerator=y[zero]*y[two]-y[one]**2;variables[name]=dict(scope_id=record['scope_id'],mass=c.rec(y[zero]),unnormalized_variance_numerator=c.rec(numerator),positive=numerator>0)
    records[record['scope_id']]=record
assert matrix_count==63 and maps==405 and nonzero_coefficients==14818
overlap=read(Q/'all-mixed-overlap-linear-certificates.json');old=S/'e037-direct-SDP-independent-v1/all-independent-direct-SDP-packing-primal-and-dual-certificates.jsonl';assert sha(old)==overlap['source_equality_audit_sha256'];oldres={}
for line in old.open():
    v=json.loads(line)
    if v.get('kind')=='AFTER_NEW_exact_equality_binding_and_actual_residual':oldres[v['row']]=c.dec(v['residual'])
assert len(oldres)==549
projections=overlap['projections']
for item in projections:
    rowco={};bound=F(0);linear={v['row']:c.dec(v['coefficient'])for v in item['equality_linear_combination']};emit(dict(kind='BEFORE_NEW_independent_cross_clique_projection',id=item['id']))
    for row,v in linear.items():rowco=add(rowco,eqco[row],v);bound+=v*c.dec(eq[row]['bound'])
    current={v['column']:c.dec(v['value'])for v in item['original_moment_coefficients']};target={v['column']:c.dec(v['value'])for v in item['target_coefficients']};assert rowco==add(current,target,-1)and bound==c.dec(item['reference_constant']);actual=sum(v*weights[col]for col,v in current.items());reference=c.dec(item['reference_constant'])+sum(v*weights[col]for col,v in target.items());prediction=sum(v*oldres[row]for row,v in linear.items());assert actual==c.dec(item['actual_moment'])and reference==c.dec(item['actual_target_moment'])and actual-reference==prediction==c.dec(item['exact_projected_residual'])
    emit(dict(kind='AFTER_NEW_independent_cross_clique_projection_RETURN',id=item['id'],coefficient_identity=True,exact_projected_residual=c.rec(prediction)))
for item in overlap['overlaps']:
    left=projections[item['left']];right=projections[item['right']];linear={v['row']:c.dec(v['coefficient'])for v in item['linear_combination']};expected=add({v['row']:c.dec(v['coefficient'])for v in right['equality_linear_combination']},{v['row']:c.dec(v['coefficient'])for v in left['equality_linear_combination']},-1);delta=c.dec(right['actual_moment'])-c.dec(left['actual_moment']);bound=sum(abs(v)*abs(oldres[row])for row,v in linear.items());assert linear==expected and delta==sum(v*oldres[row]for row,v in linear.items())==c.dec(item['difference'])and abs(delta)<=bound==c.dec(item['exact_residual_bound']);emit(dict(kind='AFTER_NEW_independent_shared_mixed_moment_RETURN',left=item['left'],right=item['right'],exact_difference=c.rec(delta),bound=c.rec(bound),exactly_equal=delta==0))
emit(dict(kind='AFTER_NEW_unnormalized_variance_diagnostic',variables=variables,no_normalization_of_actual_table=True,nonzero_variance_rejects_THIS_table_as_a_single_atom_NOT_the_family=True))
j.close();out=dict(schema='v2d.mixed-joint-independent-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),all_independent_checks_passed=True,new_analytic_controls=12,mixed_scopes_checked=19,mixed_monomial_maps_checked=maps,exact_map_coefficients_checked=nonzero_coefficients,mixed_matrices_checked=matrix_count,exact_congruence_entry_identities=congruence_identities,negative_square_certificates_checked=square_checks,mixed_PSD_failures=report['negative_joint_matrices'],matrix_ranks=ranks,projection_identities_checked=len(projections),shared_moment_comparisons_checked=len(overlap['overlaps']),exact_shared_moment_mismatches=overlapreport['exact_shared_moment_mismatches'],maximum_exact_overlap_difference=overlapreport['maximum_exact_difference'],all_shared_differences_exactly_accounted_for_by_saved_equality_residuals=True,old_equality_univariate_PSD_or_solver_audit_NOT_reexecuted=True,positive_unnormalized_variances=sum(v['positive']for v in variables.values()),variables=variables,actual_table_NOT_a_deterministic_single_atom=True,no_exact_global_common_measure_or_source_path_qualified=True,classification='mixed_local_PSD_clear_but_not_exact_shared_or_deterministic_consistency',current_reference_middle_source_authority_parameters_gates_unchanged=True,new_solver_calls=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in['variables','all_outputs_sha256','matrix_ranks']}))
