"""One NEW declared conic outer-approximation algorithm, six LP updates maximum.

Reuses the saved initial solver return without replaying that LP. Continuous
PSD separation supplies exact rational polynomial-square supporting planes.
Limits are fixed: 6updates,64unique planes/update,7cone audits,20s/subproblem.
"""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-bounded-moment-cone-v1';sys.path.insert(0,str(S/'compute'))
import conditional_moment_cones_v1 as c
import bernstein_common_moments_v1 as m
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-continuous-cone-matrices-vectors-cuts-and-returns.jsonl').open('w');emit=lambda v:g.journal(j,v)
contract=read(S/'e037-moment-cone-contract-v1/report.json');assert contract['all_controls_passed']and contract['helper_sha256']==sha(S/'compute/conditional_moment_cones_v1.py')
P=S/'e037-common-moment-square-repair-v1';base=P/'corrected-common-moment-LP-model.json';saved=P/'corrective-phase-one-solver-RETURN.json';parent=read(P/'report.json');assert sha(base)==parent['all_outputs_sha256'][base.name]and sha(saved)==parent['all_outputs_sha256'][saved.name]
model=read(base);result=read(saved);groups=c.groups(model['columns']);assert len(groups)==590 and len(model['inequalities'])==5606;updates=[];audits=[];seen=set()
for row in model['inequalities']:
    co=[(v['column'],hex(c.dec(v['value']).numerator),hex(c.dec(v['value']).denominator))for v in row['coefficients']];seen.add(hashlib.sha256(json.dumps(co,separators=(',',':')).encode()).hexdigest())
emit(dict(kind='BEFORE_NEW_bounded_continuous_cone_algorithm',base_model_sha256=sha(base),initial_RETURN_reused_sha256=sha(saved),initial_LP_NOT_reexecuted=True,maximum_LP_updates=6,maximum_new_planes_per_update=64,maximum_cone_audits=7,LP_time_limit_seconds=20,maximum_rows=5990,original_authority_and_candidate_gates_unchanged=True))
classification='UNRESOLVED_declared_conic_algorithm_budget';excluded=False;last_model=str(base);last_return=str(saved);last_dual=None
for iteration in range(7):
    if not result['success']or result['primal']is None:
        classification='UNRESOLVED_subproblem_RETURN_not_successful';break
    x=list(map(F,result['primal'][:-1]));bounds=[dict(column=i,value=c.rec(v),violation=c.rec(max(F(0),-v,v-1)))for i,v in enumerate(x)if not 0<=v<=1];allPSD=True;cuts=[];unresolved=[]
    emit(dict(kind='BEFORE_NEW_algorithm_cone_audit',iteration=iteration,solver_RETURN=last_return,actual_bound_failures=bounds,no_actual_weights_changed=True))
    for group in groups:
        value=c.separate(group,x,lambda v:emit(dict(iteration=iteration,**v)));allPSD&=value['all_exact_principal_minors_PSD'];cuts.extend(value['cuts']);unresolved.extend(value['unresolved_vectors'])
    audit=dict(iteration=iteration,groups=len(groups),all_exact_principal_minors_PSD=allPSD,exact_separating_planes=len(cuts),unresolved_vectors=unresolved,actual_bound_failures=bounds,selected_new_planes=0)
    candidates={}
    for cut in cuts:
        key=cut['cut_sha256']
        if key not in seen:candidates.setdefault(key,cut)
    ordered=sorted(candidates.values(),key=lambda v:(-c.dec(v['normalized_violation']),v['cut_sha256']));selected=ordered[:64]
    audit['available_unique_new_planes']=len(ordered);audit['selected_new_planes']=len(selected)if iteration<6 else 0;audits.append(audit)
    emit(dict(kind='AFTER_NEW_algorithm_cone_audit_RETURN',audit=audit,selection_scope='descending exact normalized mathematical violation then hash; no physical/source score selection',not_selected_due_to_fixed_cap=[v['cut_sha256']for v in ordered[64:]],all_unselected_witnesses_retained_in_journal=True))
    if allPSD:
        classification='local_continuous_PSD_clear_NOT_global_or_deterministic_qualification';break
    if iteration==6:break
    if not selected:
        classification='UNRESOLVED_no_new_exact_plane_or_unresolved_eigenvectors';break
    additions=[]
    for cut in selected:
        assert c.dec(cut['exact_square_expectation'])<0
        row=dict(name='continuous_cone:'+str(iteration)+':'+cut['cut_sha256'],coefficients=cut['coefficients'],bound=c.rec(0),meta=dict(kind='exact_continuous_cone_square_plane',iteration=iteration,cut_sha256=cut['cut_sha256'],group=cut['group'],cone=cut['cone']))
        model['inequalities'].append(row);seen.add(cut['cut_sha256']);additions.append(cut)
    assert len(model['inequalities'])<=5990
    cutfile=O/f'update-{iteration:02d}-selected-exact-planes.json';cutfile.write_text(json.dumps(g.plain(additions),allow_nan=False,indent=2)+'\n');modelpath=O/f'update-{iteration:02d}-necessary-LP-model.json';modelpath.write_text(json.dumps(g.plain(model),allow_nan=False,indent=2)+'\n')
    emit(dict(kind='BEFORE_NEW_cone_algorithm_LP_subproblem',iteration=iteration,model=str(modelpath),model_sha256=sha(modelpath),selected_planes_sha256=sha(cutfile),time_limit_seconds=20,no_solver_tolerance_or_source_gate_changes=True))
    result=m.solve_phase_one(model,time_limit=20);returnpath=O/f'update-{iteration:02d}-solver-RETURN.json';returnpath.write_text(json.dumps(g.plain(result),allow_nan=False,indent=2)+'\n');emit(dict(kind='AFTER_NEW_cone_LP_solver_RETURN',iteration=iteration,status=result['status'],success=result['success'],objective=result['objective'],return_file=str(returnpath),return_sha256=sha(returnpath)))
    dual=m.finite_box_dual(model,result);dualpath=O/f'update-{iteration:02d}-finite-box-dual.json';dualpath.write_text(json.dumps(g.plain(dual),allow_nan=False,indent=2)+'\n');emit(dict(kind='AFTER_NEW_cone_LP_exact_dual_RETURN',iteration=iteration,classification=dual['classification'],strict_gap=dual.get('strict_gap'),dual_sha256=sha(dualpath)))
    updates.append(dict(iteration=iteration,planes=len(additions),rows=len(model['inequalities']),model_file=str(modelpath),RETURN_file=str(returnpath),dual_file=str(dualpath),solver_status=result['status'],phase_one_objective=result['objective'],exact_dual_gap=dual.get('strict_gap')));last_model=str(modelpath);last_return=str(returnpath);last_dual=str(dualpath)
    if dual['classification']=='exact_saved_LP_model_INFEASIBILITY_certificate':excluded=True;classification='conditional_saved_source_time_cone_outer_model_EXCLUDED';break
j.close();data=dict(schema='v2d.bounded-continuous-moment-cone-outcome.v1',classification=classification,conditional_model_excluded=excluded,updates=updates,audits=audits,last_model_file=last_model,last_solver_RETURN_file=last_return,last_dual_file=last_dual,original_base_model_sha256=sha(base),original_saved_RETURN_sha256=sha(saved),limits=dict(LP_updates=6,planes_per_update=64,cone_audits=7,subproblem_seconds=20,maximum_rows=5990),
    no_point_pose_barycenter_eigenvector_or_reference_promoted=True,scope='Conditional inherited region2/fixed unqualified wrist/source coefficient model; no global source accuracy/task impossibility claim')
(O/'bounded-cone-outcome.json').write_text(json.dumps(g.plain(data),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.bounded-continuous-moment-cone-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/conditional_moment_cones_v1.py'),classification=classification,conditional_model_excluded=excluded,new_episode_conic_LP_subproblems=len(updates),cone_audits=len(audits),new_exact_continuous_cone_planes=sum(v['planes']for v in updates),last_phase_one_objective=result.get('objective'),last_model_file=last_model,last_solver_RETURN_file=last_return,last_dual_file=last_dual,
    fixed_declared_limits_respected=True,no_old_LP_source_clipping_graph_or_contract_job_repeated=True,current_verified_reference_and_middle_component_unchanged=True,local_moment_PSD_NOT_global_measure_pose_native_motor_or_grasp_success=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
