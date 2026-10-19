"""ONE new support-aware conic evaluation,100iterations/60seconds/one thread.

Reuses the independently bound saved canonical base and complete additional
PSD maps. No source producer, old solver, anchor/cut loop or restart runs.
"""
from pathlib import Path
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-supported-conic-evaluation-v1';P=S/'e037-supported-conic-model-v1';sys.path[:0]=[str(S/'compute'),str(S/'e037-direct-SDP-runtime-v2/site-packages')]
import numpy as np
from scipy import sparse
import clarabel
import supported_conic_maps_v1 as h
import direct_conditional_moment_sdp_v1 as old
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());j=(O/'all-new-supported-solver-and-certificate-returns.jsonl').open('w');emit=lambda v:g.journal(j,v);start=time.monotonic()
def save(name,value):
    with (O/name).open('w')as f:f.write(json.dumps(g.plain(value),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
binding=read(P/'supported-model-binding.json');ind=read(S/'e037-supported-model-independent-v1/report.json');assert ind['all_independent_bindings_passed']and ind['model_binding_sha256']==sha(P/'supported-model-binding.json');assert sha(P/'supported-canonical-A.npz')==ind['combined_A_sha256']==binding['combined_A_sha256']and sha(P/'supported-canonical-b.npy')==ind['combined_b_sha256']==binding['combined_b_sha256']
contract=read(S/'e037-supported-conic-contract-v2/report.json');assert contract['all_controls_passed']and contract['helper_sha256']==sha(S/'compute/supported_conic_maps_v1.py');A=sparse.load_npz(P/'supported-canonical-A.npz');b=np.load(P/'supported-canonical-b.npy');basepack=read(binding['base_packing']);packed=read(P/'additional-weighted-PSD-packing.json');assert sha(binding['base_packing'])==binding['base_packing_sha256'];blocks=h.base_blocks(basepack)
for file,packing in zip(binding['additional_block_files'],packed,strict=True):
    path=P/file['file'];assert sha(path)==file['sha256'];block=h.block_decode(read(path));block.update(row_start=packing['row_start'],row_end=packing['row_end']);blocks.append(block)
assert len(blocks)==1292 and A.shape==(15840,1125);model=read(binding['base_model']);assert sha(binding['base_model'])==binding['base_model_sha256'];settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100;settings.time_limit=60.;settings.max_threads=1
cones=[clarabel.ZeroConeT(549),clarabel.NonnegativeConeT(8240)]+[clarabel.PSDTriangleConeT(v['dimension'])for v in blocks]
save('declared-ONE-supported-evaluation.json',dict(schema='v2d.single-supported-conic-declaration.v1',model_binding_sha256=sha(P/'supported-model-binding.json'),independent_binding_sha256=sha(S/'e037-supported-model-independent-v1/report.json'),settings=repr(settings),solver_version=clarabel.__version__,cones=[repr(v)for v in cones],maximum_episode_calls=1,no_warm_start_pose_or_old_table_seed=True,objective='zero, feasibility only',P_zero_shape=[1125,1125],q_zero_length=1125,no_restart_tolerance_repair_or_continuation=True,solver_tolerances_NOT_original_candidate_gates=True))
emit(dict(kind='BEFORE_NEW_SUPPORTED_solver_constructor',rows=15840,columns=1125,PSD_cones=1292,settings=repr(settings)))
solver=clarabel.DefaultSolver(sparse.csc_matrix((1125,1125)),np.zeros(1125),A,b,cones,settings)
emit(dict(kind='AFTER_NEW_SUPPORTED_solver_constructor_RETURN',constructed=True,version=clarabel.__version__))
emit(dict(kind='BEFORE_NEW_SOLE_supported_conic_solve'))
result=old.solver_return(solver.solve());save('single-supported-solver-RETURN.json',result);emit(dict(kind='AFTER_NEW_SOLE_supported_conic_solver_RETURN',file='single-supported-solver-RETURN.json',sha256=sha(O/'single-supported-solver-RETURN.json'),status=result['status'],iterations=result.get('iterations'),solve_seconds=result.get('solve_time'),all_actual_x_z_s_retained=True))
certificate=dict(classification='UNRESOLVED_nonfinite_or_missing_dual_RETURN')
if result.get('z')is not None and len(result['z'])==15840 and np.isfinite(result['z']).all():
    emit(dict(kind='BEFORE_NEW_supported_exact_Gram_finite_box_certificate'))
    certificate=h.dual_certificate(model,blocks,result['z'],emit);save('supported-exact-Gram-finite-box-certificate.json',certificate);emit(dict(kind='AFTER_NEW_supported_exact_certificate_RETURN',classification=certificate['classification'],strict_gap=certificate.get('strict_gap')))
else:save('supported-exact-Gram-finite-box-certificate.json',certificate)
j.close();out=dict(schema='v2d.supported-conic-evaluation-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/supported_conic_maps_v1.py'),model_binding_sha256=sha(P/'supported-model-binding.json'),new_episode_solver_calls=1,solver_status=result['status'],solver_iterations=result.get('iterations'),solver_seconds=result.get('solve_time'),certificate_classification=certificate['classification'],strict_gap_approx=certificate.get('strict_gap',{}).get('approx'),independent_exact_certificate_check_PENDING=True,no_restart_continuation_or_parameter_sweep=True,all_original_canonical_rows_boxes_and_authority_unchanged=True,source_nonconstant_support_factors_still_UNAVAILABLE=5,local_supported_moments_NOT_common_measure_deterministic_robot_path_or_source_truth=True,all_actual_primal_dual_weights_NOT_clipped_normalized_or_promoted=True,current_F_middle_source_missing_heldout_parameters_and_gates_unchanged=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))
