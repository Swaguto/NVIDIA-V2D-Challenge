"""SOLE new image-support formulation,100iterations/60seconds/one thread."""
from pathlib import Path
import datetime,hashlib,json,math,os,struct,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-image-conic-evaluation-v1';P=S/'e037-image-conic-model-v1';sys.path[:0]=[str(S/'compute'),str(S/'e037-direct-SDP-runtime-v2/site-packages')]
import numpy as np
from scipy import sparse
import clarabel
import supported_conic_maps_v1 as h
import direct_conditional_moment_sdp_v1 as old
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());j=(O/'all-NEW-image-conic-solver-and-certificate-returns.jsonl').open('w');emit=lambda v:g.journal(j,v);start=time.monotonic()
def safe(v):
 v=g.plain(v)
 if isinstance(v,float)and not math.isfinite(v):return dict(NONFINITE_raw_float=repr(v),raw_float64_bytes_hex=struct.pack('>d',v).hex())
 if isinstance(v,dict):return {k:safe(z)for k,z in v.items()}
 if isinstance(v,list):return list(map(safe,v))
 return v
def save(name,v):
 with (O/name).open('w')as f:f.write(json.dumps(safe(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
binding=read(P/'image-model-binding.json');ind=read(S/'e037-image-conic-binding-v1/report.json');assert ind['all_independent_bindings_passed']and ind['model_binding_sha256']==sha(P/'image-model-binding.json');assert sha(P/'image-canonical-A.npz')==ind['A_sha256']==binding['combined_A_sha256']and sha(P/'image-canonical-b.npy')==ind['b_sha256']==binding['combined_b_sha256'];assert clarabel.__version__=='0.11.1'
contract=read(S/'e037-image12-contract-v1/report.json');assert contract['all_controls_passed']and contract['helper_sha256']==sha(S/'compute/image_conic_packing_v1.py');sb=read(binding['supported_base_binding']);SB=Path(binding['supported_base_binding']).parent;assert sha(binding['supported_base_binding'])==binding['supported_base_binding_sha256'];basepack=read(sb['base_packing']);assert sha(sb['base_packing'])==sb['base_packing_sha256'];blocks=h.base_blocks(basepack)
for folder,files,packedfile in [(SB,sb['additional_block_files'],SB/'additional-weighted-PSD-packing.json'),(P,binding['image_block_files'],P/'image-five-PSD-packing.json')]:
 for f,packing in zip(files,read(packedfile),strict=True):
  path=folder/f['file'];assert sha(path)==f['sha256'];block=h.block_decode(read(path));block.update(row_start=packing['row_start'],row_end=packing['row_end']);blocks.append(block)
A=sparse.load_npz(P/'image-canonical-A.npz');b=np.load(P/'image-canonical-b.npy');assert len(blocks)==1297 and A.shape==(16230,1125);model=read(sb['base_model']);assert sha(sb['base_model'])==sb['base_model_sha256'];settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100;settings.time_limit=60.;settings.max_threads=1;cones=[clarabel.ZeroConeT(549),clarabel.NonnegativeConeT(8240)]+[clarabel.PSDTriangleConeT(v['dimension'])for v in blocks]
save('declared-ONE-image-conic-evaluation.json',dict(schema='v2d.single-image-conic-declaration.v1',model_binding_sha256=sha(P/'image-model-binding.json'),independent_binding_sha256=sha(S/'e037-image-conic-binding-v1/report.json'),settings=repr(settings),solver_version=clarabel.__version__,cones=[repr(v)for v in cones],maximum_episode_calls=1,no_warmstart_or_diagnostic_table_seed=True,objective='zero feasibility only',P_zero_shape=[1125,1125],q_zero_length=1125,no_restart_tolerance_repair_or_continuation=True,all_model_guards_NECESSARY_NOT_candidate_tolerances=True));emit(dict(kind='BEFORE_NEW_image_solver_constructor',rows=16230,columns=1125,PSD_cones=1297,settings=repr(settings)));solver=clarabel.DefaultSolver(sparse.csc_matrix((1125,1125)),np.zeros(1125),A,b,cones,settings);emit(dict(kind='AFTER_NEW_image_solver_constructor_RETURN',constructed=True,version=clarabel.__version__));emit(dict(kind='BEFORE_NEW_SOLE_image_conic_solve'));result=old.solver_return(solver.solve());save('single-image-conic-solver-RETURN.json',result);emit(dict(kind='AFTER_NEW_SOLE_image_solver_RETURN',file='single-image-conic-solver-RETURN.json',sha256=sha(O/'single-image-conic-solver-RETURN.json'),status=result['status'],iterations=result.get('iterations'),solve_seconds=result.get('solve_time'),all_actual_x_z_s_retained=True))
certificate=dict(classification='UNRESOLVED_nonfinite_or_missing_dual_RETURN')
if result.get('z')is not None and len(result['z'])==16230 and np.isfinite(result['z']).all():
 emit(dict(kind='BEFORE_NEW_image_exact_Gram_finitebox_certificate'));certificate=h.dual_certificate(model,blocks,result['z'],emit);save('image-exact-Gram-finite-box-certificate.json',certificate);emit(dict(kind='AFTER_NEW_image_exact_certificate_RETURN',classification=certificate['classification'],strict_gap=certificate.get('strict_gap')))
else:save('image-exact-Gram-finite-box-certificate.json',certificate)
j.close();out=dict(schema='v2d.image-conic-evaluation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),model_binding_sha256=sha(P/'image-model-binding.json'),new_episode_solver_calls=1,solver_status=result['status'],solver_iterations=result.get('iterations'),solver_seconds=result.get('solve_time'),certificate_classification=certificate['classification'],strict_gap_approx=certificate.get('strict_gap',{}).get('approx'),independent_exact_return_certificate_check_PENDING=True,no_restart_continuation_warmstart_or_sweep=True,all1297original_rational_maps_and_original_rows_unchanged=True,scalar_nonconstant_source_supports_still_UNAVAILABLE=5,local_support_positivity_NOT_common_measure_deterministic_path_or_source_truth=True,all_actual_returns_NOT_clipped_normalized_or_promoted=True,current_F_middle_source_missing_heldout_parameters_and_gates_unchanged=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))
