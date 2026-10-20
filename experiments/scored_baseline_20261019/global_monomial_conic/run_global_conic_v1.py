"""SOLE new global-monomial-aware evaluation;100iterations/60seconds/1thread."""
from pathlib import Path
import datetime,hashlib,json,math,os,struct,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-global-conic-evaluation-v1';P=S/'e037-global-conic-model-v1';sys.path.insert(0,str(S/'e037-direct-SDP-runtime-v2/site-packages'))
import numpy as np
from scipy import sparse
import clarabel
import global_conic24_v1 as q
import supported_conic_maps_v1 as h
import direct_conditional_moment_sdp_v1 as old
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();J=(O/'SOLE-global-solver-and-certificate-returns.jsonl').open('w');emit=lambda v:g.journal(J,v)
def safe(v):
 v=g.plain(v)
 if isinstance(v,float)and not math.isfinite(v):return dict(NONFINITE_raw_float=repr(v),float64_bytes_hex=struct.pack('>d',v).hex())
 if isinstance(v,dict):return {k:safe(x)for k,x in v.items()}
 if isinstance(v,list):return list(map(safe,v))
 return v
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(safe(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
ind=read(S/'e037-global-conic-binding-v1/report.json');binding=read(P/'global-model-binding.json');assert ind['all_independent_bindings_passed']and ind['model_binding_sha256']==sha(P/'global-model-binding.json')and sha(P/'global-canonical-A.npz')==ind['A_sha256']==binding['combined_A_sha256']and sha(P/'global-canonical-b.npy')==ind['b_sha256']==binding['combined_b_sha256'];assert clarabel.__version__=='0.11.1';sign=read(S/'e037-global24-dual-contract-v1/report.json');assert sign['all_NEW_independent_controls_passed'];A=sparse.load_npz(P/'global-canonical-A.npz');b=np.load(P/'global-canonical-b.npy');blocks,sb=q.saved_base_blocks(binding['prefix_image_folder']);globalblock=h.block_decode(read(P/'global24-original-rational-block.json'));globalblock.update(row_start=16570,row_end=16870);blocks.append(globalblock);model=read(P/'extended-original-exact-linear-model.json');assert len(blocks)==1298 and A.shape==(16870,1295)and len(model['columns'])==1295;settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100;settings.time_limit=60.;settings.max_threads=1;cones=[clarabel.ZeroConeT(549),clarabel.NonnegativeConeT(8240)]+[clarabel.PSDTriangleConeT(r['dimension'])for r in blocks[:-1]]+[clarabel.NonnegativeConeT(340),clarabel.PSDTriangleConeT(24)];declaration=dict(schema='v2d.ONE-global-conic-declaration.v1',model_binding_sha256=sha(P/'global-model-binding.json'),independent_binding_sha256=sha(S/'e037-global-conic-binding-v1/report.json'),solver_version=clarabel.__version__,settings=repr(settings),maximum_episode_solver_calls=1,rows=16870,columns=1295,original_prefix_rows=16230,original_columns=1125,auxiliary_FREE_UNOBSERVED_columns=170,total_PSDcones=1298,cone_order=[repr(v)for v in cones],objective='zero feasibility ONLY',no_warmstart_restart_tolerance_repair_continuation_or_sweep=True,all_actual_x_z_s_will_remain_unaltered=True,all_guards_NECESSARY_not_original_candidate_tolerances=True);save('declared-ONE-global-conic-evaluation.json',declaration);emit(dict(kind='BEFORE_NEW_GLOBAL_solver_constructor',declaration=declaration))
try:solver=clarabel.DefaultSolver(sparse.csc_matrix((1295,1295)),np.zeros(1295),A,b,cones,settings);constructor=dict(constructed=True,version=clarabel.__version__,rows=A.shape[0],columns=A.shape[1])
except BaseException as ex:
 constructor=dict(constructed=False,error=repr(ex));save('global-solver-constructor-RETURN.json',constructor);emit(dict(kind='AFTER_NEW_GLOBAL_solver_constructor_RETURN',RETURN=constructor));raise
save('global-solver-constructor-RETURN.json',constructor);emit(dict(kind='AFTER_NEW_GLOBAL_solver_constructor_RETURN',RETURN=constructor));emit(dict(kind='BEFORE_NEW_SOLE_global_solve'))
try:result=old.solver_return(solver.solve())
except BaseException as ex:save('global-solver-EXCEPTION-RETURN.json',dict(error=repr(ex),one_call_attempted_NO_retry=True));emit(dict(kind='AFTER_NEW_SOLE_global_solve_EXCEPTION',error=repr(ex)));raise
save('single-global-conic-solver-RETURN.json',result);emit(dict(kind='AFTER_NEW_SOLE_global_solver_RETURN',file='single-global-conic-solver-RETURN.json',sha256=sha(O/'single-global-conic-solver-RETURN.json'),status=result['status'],iterations=result.get('iterations'),seconds=result.get('solve_time'),all_actual_primal_dual_slack_saved_before_further_work=True));certificate=dict(classification='UNRESOLVED_nonfinite_or_missing_raw_dual')
if result.get('z')is not None and len(result['z'])==16870 and np.isfinite(result['z']).all():
 emit(dict(kind='BEFORE_NEW_global_exact_Gram_finitebox_certificate',all1295original_AND_aux_variables_in_finite_box_bound=True,unitbox_duals_not_needed_for_exact_residual_minimum=True));certificate=h.dual_certificate(model,blocks,result['z'],emit);save('global-exact-Gram-finite-box-certificate.json',certificate);emit(dict(kind='AFTER_NEW_global_exact_certificate_RETURN',classification=certificate['classification'],strict_gap=certificate.get('strict_gap')))
else:save('global-exact-Gram-finite-box-certificate.json',certificate)
J.close();out=dict(schema='v2d.global-conic-evaluation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_binding_sha256=sha(P/'global-model-binding.json'),new_episode_solver_calls=1,solver_status=result['status'],solver_iterations=result.get('iterations'),solver_seconds=result.get('solve_time'),total_rows=16870,total_columns=1295,total_PSDcones=1298,certificate_classification=certificate['classification'],strict_gap_approx=certificate.get('strict_gap',{}).get('approx'),new_RETURN_independent_exact_audit_PENDING=True,no_restart_warmstart_continuation_or_tolerance_sweep=True,actual_primal_dual_slack_NOT_clipped_normalized_repaired_or_promoted=True,global_necessary_feasibility_NOT_common_measure_or_trajectory=True,all_original_source_root_finger_native_physical_gates_UNCHANGED=True,persistent_trajectory_qualification_goal_active=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))
