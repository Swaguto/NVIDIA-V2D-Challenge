"""One corrective LP with all247 independently proven square constraints.

This changes necessary measure semantics, not source/timing/physical gates.
No primary LP replay, restart, tuning, adaptive cut loop or barycenter pose.
"""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-common-moment-square-repair-v1';sys.path.insert(0,str(S/'compute'))
import bernstein_common_moments_v1 as m
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-proven-square-repair-inputs-and-solver-returns.jsonl').open('w')
P=S/'e037-common-moment-model-v1/common-moment-LP-model.json';C=S/'e037-common-moment-independent-v1/saved-table-negative-square-separators.json';evidence=read(C.parent/'report.json');assert evidence['all_model_and_transfer_checks_passed']and evidence['exact_negative_square_separators']==247;assert sha(P)==read(P.parent/'report.json')['all_outputs_sha256'][P.name]and sha(C)==evidence['all_outputs_sha256'][C.name]
model=read(P);cuts=read(C)['cuts'];g.journal(j,dict(kind='BEFORE_NEW_proven_square_correction',base_model_sha256=sha(P),independent_cut_sha256=sha(C),all_247_cuts_fixed_in_advance=True,one_corrective_LP=True,no_adaptive_iterations_or_tolerance_changes=True))
for k,cut in enumerate(cuts):
    co={v['column']:m.dec(v['value'])for v in cut['coefficients']};scale=max(map(abs,co.values()));row=dict(name='proven_off_grid_square:'+str(k),coefficients=[dict(column=c,value=m.rec(v/scale))for c,v in sorted(co.items())],bound=m.rec(F(0)),meta=dict(kind='independently_proven_conditioned_square',certificate_index=k,certificate_sha256=sha(C)))
    model['inequalities'].append(row);g.journal(j,dict(kind='AFTER_NEW_valid_necessary_square_row',row=row,certificate=cut))
model['square_repair_input_sha256']=sha(C);model['original_source_motion_physical_gates_unchanged']=True
(O/'corrected-common-moment-LP-model.json').write_text(json.dumps(g.plain(model),allow_nan=False,indent=2)+'\n');assert len(model['inequalities'])==5606 and len(model['columns'])==1125
g.journal(j,dict(kind='BEFORE_NEW_single_corrective_phase_one_LP',model_sha256=sha(O/'corrected-common-moment-LP-model.json'),time_limit_seconds=120))
result=m.solve_phase_one(model,time_limit=120);(O/'corrective-phase-one-solver-RETURN.json').write_text(json.dumps(g.plain(result),allow_nan=False,indent=2)+'\n');g.journal(j,dict(kind='AFTER_NEW_corrective_solver_RETURN',return_sha256=sha(O/'corrective-phase-one-solver-RETURN.json'),status=result['status'],objective=result['objective']))
certificate=m.finite_box_dual(model,result);(O/'corrective-finite-box-dual.json').write_text(json.dumps(g.plain(certificate),allow_nan=False,indent=2)+'\n');g.journal(j,dict(kind='AFTER_NEW_corrective_dual_RETURN',classification=certificate['classification'],strict_gap=certificate.get('strict_gap')))
classification='conditional_saved_model_EXCLUDED'if certificate['classification']=='exact_saved_LP_model_INFEASIBILITY_certificate'else('numerical_corrected_relaxation_feasible_NOT_a_pose'if result['success']and result['objective']<=1e-7 else'UNRESOLVED_corrective_relaxation')
j.close();out=dict(schema='v2d.common-moment-square-repair-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),corrective_episode_LPs=1,new_independently_proven_square_constraints=247,columns=1125,equality_rows=549,inequality_rows=5606,phase_one_status=result['status'],phase_one_objective=result['objective'],classification=classification,dual_classification=certificate['classification'],exact_dual_gap=certificate.get('strict_gap'),
    not_a_solver_restart_or_tolerance_repair=True,all_original_authority_parameters_and_reference_unchanged=True,no_adaptive_iterations_barycenter_or_pose_promotion=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
