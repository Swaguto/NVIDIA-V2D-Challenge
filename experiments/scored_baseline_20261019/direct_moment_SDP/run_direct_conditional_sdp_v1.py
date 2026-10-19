"""One declared direct SDP evaluation: 100 iterations, 60 seconds, no restart.

The saved 5990-row necessary model is reused; no old LP or source producer runs.
Direct local PSD constraints do not establish a global measure or robot path.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-direct-conditional-SDP-v1';sys.path[:0]=[str(S/'compute'),str(S/'e037-direct-SDP-runtime-v2/site-packages')]
import numpy as np
from scipy import sparse
import clarabel
import direct_conditional_moment_sdp_v1 as d
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-direct-SDP-packing-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(j,v)
def save(name,value):
    p=O/name
    with p.open('w')as f:f.write(json.dumps(g.plain(value),allow_nan=False,indent=2)+'\n');f.flush();__import__('os').fsync(f.fileno())
    return p
contract=read(S/'e037-direct-SDP-contract-v2/report.json');assert contract['all_controls_passed']and contract['helper_sha256']==sha(S/'compute/direct_conditional_moment_sdp_v1.py')
old=read(S/'e037-bounded-moment-cone-v1/report.json');base=S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json';assert sha(base)==old['all_outputs_sha256'][base.name]
model=read(base);assert len(model['columns'])==1125 and len(model['equalities'])==549 and len(model['inequalities'])==5990
emit(dict(kind='BEFORE_NEW_one_direct_SDP_evaluation',base_model=str(base),base_model_sha256=sha(base),maximum_episode_solver_calls=1,maximum_iterations=100,solver_time_limit_seconds=60,thread_limit=1,solver_tolerances_default_and_NOT_candidate_gates=True,all_original_model_box_authority_unchanged=True,old_planes_reused_NOT_regenerated=True))
A,b,packing=d.assemble(model,emit);packing.update(schema='v2d.direct-conditional-SDP-packing.v1',base_model=str(base),base_model_sha256=sha(base),helper_sha256=sha(S/'compute/direct_conditional_moment_sdp_v1.py'),packing_error_is_NECESSARY_outer_bound_NOT_candidate_tolerance=True)
save('complete-direct-SDP-packing-and-error-bounds.json',packing);sparse.save_npz(O/'canonical-A.npz',A);np.save(O/'canonical-b.npy',b)
settings=clarabel.DefaultSettings();settings.verbose=False;settings.time_limit=60.;settings.max_iter=100;settings.max_threads=1
cones=[clarabel.ZeroConeT(packing['equalities']),clarabel.NonnegativeConeT(packing['nonnegative_rows'])]+[clarabel.PSDTriangleConeT(v['dimension'])for v in packing['PSD_blocks']]
save('declared-solver-settings-and-cones.json',dict(version=clarabel.__version__,settings=repr(settings),cones=[repr(v)for v in cones],objective='zero, direct feasibility',P_zero_shape=[1125,1125],q_zero_length=1125,canonical_A_sha256=sha(O/'canonical-A.npz'),canonical_b_sha256=sha(O/'canonical-b.npy')))
emit(dict(kind='BEFORE_NEW_direct_solver_constructor',rows=A.shape[0],columns=A.shape[1],nnz=A.nnz,PSD_cones=len(packing['PSD_blocks']),settings=repr(settings)))
solver=clarabel.DefaultSolver(sparse.csc_matrix((1125,1125)),np.zeros(1125),A,b,cones,settings)
emit(dict(kind='AFTER_NEW_direct_solver_constructor_RETURN',constructed=True))
emit(dict(kind='BEFORE_NEW_SINGLE_direct_SDP_solve'))
solution=solver.solve();result=d.solver_return(solution);returned=save('single-direct-SDP-solver-RETURN.json',result)
emit(dict(kind='AFTER_NEW_SINGLE_direct_SDP_solver_RETURN',file=str(returned),sha256=sha(returned),status=result['status'],iterations=result.get('iterations'),solve_time=result.get('solve_time'),actual_primal_and_dual_retained_unchanged=True))
certificate=dict(classification='UNRESOLVED_nonfinite_or_missing_conic_return')
if result.get('z')is not None and len(result['z'])==A.shape[0]and np.isfinite(result['z']).all():
    emit(dict(kind='BEFORE_NEW_residual_corrected_rational_conic_certificate'))
    certificate=d.certificate(model,packing,result['z'],emit);save('residual-corrected-rational-conic-certificate.json',certificate)
    emit(dict(kind='AFTER_NEW_residual_corrected_conic_certificate_RETURN',classification=certificate['classification'],strict_gap=certificate.get('strict_gap')))
else:save('residual-corrected-rational-conic-certificate.json',certificate)
diagnostics=dict(solver_status=result['status'],NOT_original_candidate_tolerances=True)
if result.get('x')is not None and np.isfinite(result['x']).all():
    x=np.array(result['x']);diagnostics.update(max_unit_box_violation=float(max(0.,-min(x),max(x)-1)),canonical_equality_residual=float(np.max(abs(A[:549]@x-b[:549]))))
save('solver-diagnostics-NOT-trajectory-checks.json',diagnostics);j.close()
out=dict(schema='v2d.direct-conditional-SDP-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/direct_conditional_moment_sdp_v1.py'),base_model_sha256=sha(base),classification=certificate['classification'],solver_status=result['status'],iterations=result.get('iterations'),solve_seconds=result.get('solve_time'),columns=A.shape[1],rows=A.shape[0],nnz=A.nnz,conditional_groups=590,direct_PSD_cones=len(packing['PSD_blocks']),new_episode_SDP_solves=1,no_restart_or_budget_extension=True,strict_gap=certificate.get('strict_gap'),independent_certificate_verification_PENDING=True,necessary_moment_table_NOT_global_measure_or_pose_or_path=True,current_verified_reference_middle_source_and_gates_unchanged=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
save('report.json',out);print(json.dumps(g.plain({k:v for k,v in out.items()if k!='all_outputs_sha256'})))
