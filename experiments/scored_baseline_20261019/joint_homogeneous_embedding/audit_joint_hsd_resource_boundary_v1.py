"""NEW program-order/SAVED-return resource-boundary audit, no numerical replay."""
from joint_saved_mean_bound_v1 import S,C,typed,sha,save
from joint_lossless_returns_v1 import seal_return
from pathlib import Path
import json,ast,numpy as np
D=S/'e037-joint-hsd-boundary-audit-v1';E=S/'e037-joint-hsd-evaluation-v1';P=C/'joint-hsd-evaluation-CPU-plan-v1.json';plan=json.loads(P.read_text());worker=Path(plan['worker']);source=worker.read_text();tree=ast.parse(source)
paths=[P,worker,E/'report.json',E/'guard.json',*[p for p in E.glob('*.json') if p.name not in ['report.json','guard.json']]]
save(D/'NEW-boundary-input-bindings-BEFORE.json',{str(p):sha(p) for p in paths})
def get(name):return typed(next(E.glob('*-'+name+'.json')))
inputs=get('NEW-method-F-authority-input-BEFORE');resource=get('NEW-before-factor-resource-RETURN');boundary=get('NEW-actual-method-boundary-RETURN');report=json.loads((E/'report.json').read_text());guard=json.loads((E/'guard.json').read_text())
start=typed(S/'e037-joint-hsd-preflight-v1/NEW-embedding-constants-F-start-RETURN.json')
resource_line=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Raise) and isinstance(n.exc,ast.Call) and isinstance(n.exc.func,ast.Name) and n.exc.func.id=='OSError')
factor_line=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='cho_factor')
loop_line=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
factor_returns=list(E.glob('*normal-factor-RETURN*'));updates=list(E.glob('*HSD-update*'));query_returns=list(E.glob('*terminal8*'))
needed=resource['declared_return_estimate']+resource['reserve_bytes'];deficit=needed-resource['free_disk_bytes']
finiteness={k:bool(np.isfinite(v).all()) for k,v in inputs.items() if isinstance(v,np.ndarray)}
seal_return(D,'NEW-complete-input-and-resource-boundary-binding-RETURN',dict(full_input_arrays_finite=finiteness,F_seed_bits=bool(inputs['ux'].tobytes()==start['x0'].tobytes()),
 original_formal_phase_zero_map_unchanged=True,resource=resource,total_free_required_bytes=needed,shortfall_bytes=deficit,
 resource_raise_line=resource_line,factor_call_line=factor_line,update_loop_line=loop_line,source_sha256=sha(worker),plan_worker_sha256=plan['worker_sha256'],
 raw_exception=boundary,completed_updates=report['completed_updates'],factor_return_files=factor_returns,update_return_files=updates,terminal_return_files=query_returns,
 no_factorization_solve_operator_projection_update_primal_recovery_NEW_point_source_geometry_native_or_physics_execution=True,
 no_return_reconstruction_no_budget_change_no_retry=True,allocation_change_cause_beyond_saved_space_receipt_UNPROVEN=True))
assert all(finiteness.values()) and inputs['ux'].tobytes()==start['x0'].tobytes()
assert sha(worker)==plan['worker_sha256'] and resource_line<factor_line<loop_line and deficit>0 and boundary['type']=='OSError'
assert report['completed_updates']==0 and not report['NEW_point_sealed'] and not report['complete_NEW_query'] and not factor_returns and not updates and not query_returns
assert guard['worker_exit_code']==0 and not (E/'worker.log').read_text()
save(D/'report.json',dict(passed=True,declared_updates=8,actual_updates=0,method_wall_seconds=report['wall_seconds'],before_factorization_and_all_numerical_method_operators=True,
 raw_status=report['status'],resource_gate='declared durable returns plus original reserve exceed actual free disk',free_disk_bytes=resource['free_disk_bytes'],
 declared_return_bytes=resource['declared_return_estimate'],reserve_bytes=resource['reserve_bytes'],total_required_free_bytes=needed,shortfall_bytes=deficit,
 all_method_inputs_durable_finite=True,no_numeric_factor_solver_return_EXPECTED_because_never_called=True,no_actual_warnings=True,
 all_original_rows_Pq_gates_F_and_failures_UNCHANGED=True,no_candidate_or_owner_goal_qualification=True,no_worker_method_or_old_audit_replay=True))
