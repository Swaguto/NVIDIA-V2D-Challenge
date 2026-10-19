"""New exact cross-clique moment projections from saved marginal equalities.

Reuse the prior exact equality residuals by hash; do not reexecute that audit.
Residual bounds describe the actual unaltered numerical table, not tolerances
or a repair that could qualify a common measure or robot trajectory.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-mixed-overlap-certificates-v1';sys.path.insert(0,str(S/'compute'))
import mixed_joint_moments_v1 as m
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-cross-clique-projections-residuals-and-overlap-certificates.jsonl').open('w');emit=lambda v:g.journal(j,v)
def add(a,b,scale=F(1)):
    out=dict(a)
    for k,v in b.items():out[k]=out.get(k,F(0))+scale*v
    return {k:v for k,v in out.items()if v}
def recmap(v):return [dict(column=k,value=c.rec(x))for k,x in sorted(v.items())]
def controls():
    emit(dict(kind='BEFORE_NEW_overlap_linear_identity_contracts'))
    big={a:k for k,a in enumerate(product(range(5),repeat=2))};small={(k,):25+k for k in range(5)};a=m.monomial_coefficients(big,(0,3));b=m.monomial_coefficients(small,(3,));difference=add(a,b,-1);composed={}
    for k in range(5):
        row={col:F(1)for (i,kk),col in big.items()if kk==k};row[25+k]=F(-1);co=m.monomial_coefficients({(k,):0},(3,)).get(0,F(0));composed=add(composed,row,co)
    emit(dict(kind='AFTER_NEW_overlap_projection_contract_RETURN',difference=recmap(difference),composed=recmap(composed),equal=difference==composed));assert difference==composed
    actual=[F((-1)**k*(k+1),k+7)for k in range(30)];left=sum(v*actual[k]for k,v in difference.items());right=F(0);bound=F(0)
    for k in range(5):
        residual=sum(actual[col]for (i,kk),col in big.items()if kk==k)-actual[25+k];co=F(k*(k-1)*(k-2),24)if k>=3 else F(0);right+=co*residual;bound+=abs(co)*abs(residual)
    emit(dict(kind='AFTER_NEW_signed_residual_contract_RETURN',direct=c.rec(left),composed=c.rec(right),absolute_bound=c.rec(bound)));assert left==right and abs(left)<=bound
    norm_big=sum(actual[:25])-1;norm_small=sum(actual[25:])-1;direct=sum(actual[:25])-sum(actual[25:]);emit(dict(kind='AFTER_NEW_mass_normalization_difference_contract_RETURN',direct=c.rec(direct),from_saved_normalization=c.rec(norm_big-norm_small)));assert direct==norm_big-norm_small
    return 3
new_controls=controls()
modelpath=S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json';model=read(modelpath);tables=m.tables(model['columns']);eq=model['equalities'];byname={v['name']:i for i,v in enumerate(eq)};eqco=[{v['column']:c.dec(v['value'])for v in row['coefficients']}for row in eq]
old=S/'e037-direct-SDP-independent-v1';oldreport=read(old/'report.json');saved=old/'all-independent-direct-SDP-packing-primal-and-dual-certificates.jsonl';assert sha(saved)==oldreport['all_outputs_sha256'][saved.name];residual={}
for line in saved.open():
    v=json.loads(line)
    if v.get('kind')=='AFTER_NEW_exact_equality_binding_and_actual_residual':residual[v['row']]=c.dec(v['residual'])
assert len(residual)==549
parent=S/'e037-saved-mixed-joint-moments-v1';parent_report=read(parent/'report.json');returnpath=S/'e037-direct-conditional-SDP-v1/single-direct-SDP-solver-RETURN.json';weights=list(map(F,read(returnpath)['x']));projections=[];groups={};scopefiles=sorted(parent.glob('scope-*-exact-joint-square-record.json'))
emit(dict(kind='BEFORE_NEW_cross_clique_moment_projection',scopes=len(scopefiles),saved_equality_audit_reused_sha256=sha(saved),saved_table_sha256=sha(returnpath),original_shared_equalities_unchanged=True,no_actual_weights_altered=True))
for path in scopefiles:
    assert sha(path)==parent_report['all_outputs_sha256'][path.name];record=read(path);scope=tuple(record['scope']);entries=tables[scope]
    for moment in record['moment_maps']:
        power=tuple(moment['power']);support=tuple(v for v,r in zip(scope,power)if r);globalkey=tuple(sorted((v,r)for v,r in zip(scope,power)if r));current={v['column']:c.dec(v['value'])for v in moment['coefficients']};value=c.dec(moment['actual_moment']);expr={};target=()
        emit(dict(kind='BEFORE_NEW_cross_clique_projection',scope=scope,power=power,global_monomial=globalkey))
        if not support:
            targetco={};reference_constant=F(1);expr={byname['normalize:'+','.join(scope)]:F(1)}
        else:
            available=[t for t in tables if set(support)<=set(t)<=set(scope)];target=min(available,key=lambda t:(len(t),t));targetpower=tuple(power[scope.index(v)]for v in target);targetco=m.monomial_coefficients(tables[target],targetpower);reference_constant=F(0)
            if scope!=target:
                for index in tables[target]:
                    coefficient=F(1)
                    from math import comb
                    for k,r in zip(index,targetpower):coefficient*=F(comb(k,r),comb(4,r))if k>=r else F(0)
                    if coefficient:expr[byname['marginal:'+','.join(scope)+'->'+','.join(target)+':'+str(index)]]=coefficient
        combined={};bound_constant=F(0)
        for row,co in expr.items():combined=add(combined,eqco[row],co);bound_constant+=co*c.dec(eq[row]['bound'])
        assert combined==add(current,targetco,-1)and bound_constant==reference_constant
        prediction=sum(co*residual[row]for row,co in expr.items());targetvalue=reference_constant+sum(co*weights[col]for col,co in targetco.items());assert value-targetvalue==prediction
        item=dict(id=len(projections),scope_id=record['scope_id'],scope=list(scope),power=list(power),global_monomial=[list(v)for v in globalkey],target_scope=list(target),target_coefficients=recmap(targetco),reference_constant=c.rec(reference_constant),original_moment_coefficients=recmap(current),equality_linear_combination=[dict(row=row,name=eq[row]['name'],coefficient=c.rec(co))for row,co in sorted(expr.items())],actual_moment=c.rec(value),actual_target_moment=c.rec(targetvalue),exact_projected_residual=c.rec(prediction))
        emit(dict(kind='AFTER_NEW_cross_clique_projection_RETURN',record=item));projections.append(item);groups.setdefault(globalkey,[]).append(item)
overlaps=[]
for key,items in groups.items():
    if len(items)<2:continue
    first=items[0];firstexpr={v['row']:c.dec(v['coefficient'])for v in first['equality_linear_combination']}
    for other in items[1:]:
        emit(dict(kind='BEFORE_NEW_shared_mixed_moment_comparison',global_monomial=key,left=first['id'],right=other['id']))
        expr=add({v['row']:c.dec(v['coefficient'])for v in other['equality_linear_combination']},firstexpr,-1);delta=c.dec(other['actual_moment'])-c.dec(first['actual_moment']);predicted=sum(co*residual[row]for row,co in expr.items());absolute=sum(abs(co)*abs(residual[row])for row,co in expr.items());assert delta==predicted and abs(delta)<=absolute and other['target_scope']==first['target_scope']and other['target_coefficients']==first['target_coefficients']
        comparison=dict(global_monomial=[list(v)for v in key],left=first['id'],right=other['id'],difference=c.rec(delta),linear_combination=[dict(row=row,coefficient=c.rec(co))for row,co in sorted(expr.items())],exact_residual_bound=c.rec(absolute),exactly_equal=delta==0)
        emit(dict(kind='AFTER_NEW_shared_mixed_moment_comparison_RETURN',record=comparison));overlaps.append(comparison)
j.close();data=dict(schema='v2d.mixed-overlap-certificates.v1',new_projection_contracts=new_controls,source_equality_audit_sha256=sha(saved),saved_solver_RETURN_sha256=sha(returnpath),model_sha256=sha(modelpath),projections=projections,overlaps=overlaps)
(O/'all-mixed-overlap-linear-certificates.json').write_text(json.dumps(g.plain(data),allow_nan=False,indent=2)+'\n');mismatch=[v for v in overlaps if not v['exactly_equal']]
out=dict(schema='v2d.mixed-overlap-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),new_analytic_projection_controls=3,projection_identities=len(projections),unique_global_monomials=len(groups),shared_monomial_comparisons=len(overlaps),exact_shared_moment_mismatches=len(mismatch),maximum_exact_difference=max([0.]+[abs(c.dec(v['difference']))for v in overlaps]),maximum_exact_difference_approx=max([0.]+[abs(v['difference']['approx'])for v in overlaps]),all_differences_bound_by_SAVED_original_marginal_equality_residuals=True,old_equality_audit_reused_NOT_reexecuted=True,no_candidate_tolerance_or_actual_table_changed=True,classification='not_exact_global_marginal_consistency_residuals_retained'if mismatch else'shared_moment_values_agree_NOT_global_measure_sufficiency',no_common_measure_deterministic_path_or_family_exclusion=True,independent_projection_verification_PENDING=True,new_solver_calls=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
out['maximum_exact_difference']=c.rec(out['maximum_exact_difference']);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))
