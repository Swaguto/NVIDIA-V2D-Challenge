"""Build one NEW support-aware model by appending saved complete PSD maps.

No original canonical row is regenerated or altered; no solver call occurs.
"""
from pathlib import Path
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-supported-conic-model-v1';sys.path.insert(0,str(S/'compute'))
import numpy as np
from scipy import sparse
import supported_conic_maps_v1 as h
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-new-supported-model-bindings-and-packing-returns.jsonl').open('w');emit=lambda v:g.journal(j,v)
def save(name,value):
    with (O/name).open('w')as f:f.write(json.dumps(g.plain(value),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
contractpath=S/'e037-supported-conic-contract-v2/report.json';contract=read(contractpath);assert contract['all_controls_passed']and contract['helper_sha256']==sha(S/'compute/supported_conic_maps_v1.py')and contract['new_analytic_controls']==10
B=S/'e037-direct-conditional-SDP-v1';parent=read(B/'report.json');basefiles=[B/'canonical-A.npz',B/'canonical-b.npy',B/'complete-direct-SDP-packing-and-error-bounds.json']
for p in basefiles:assert sha(p)==parent['all_outputs_sha256'][p.name]
A=sparse.load_npz(basefiles[0]);b=np.load(basefiles[1]);basepacking=read(basefiles[2]);assert A.shape==(14099,1125)and len(b)==14099 and len(basepacking['PSD_blocks'])==1180
emit(dict(kind='BEFORE_NEW_supported_model_assembly',base_canonical_A_reused_sha256=sha(basefiles[0]),base_canonical_b_reused_sha256=sha(basefiles[1]),base_rows_NOT_reexecuted_or_regenerated=True,old_solver_NOT_called=True,additional_complete_joint_cones=63,additional_complete_support_cones=49))
blocks=[];sources=[]
for folder,pattern,kind in [(S/'e037-saved-mixed-joint-moments-v1','scope-*-exact-joint-square-record.json','joint'),(S/'e037-pointwise-support-diagnosis-v1','factor-*-support-localizer-record.json','support')]:
    report=read(folder/'report.json')
    for p in sorted(folder.glob(pattern)):
        assert sha(p)==report['all_outputs_sha256'][p.name];origin=dict(path=str(p),sha256=sha(p));record=read(p);emit(dict(kind='BEFORE_NEW_saved_matrix_map_binding',origin=origin,kind_of_scope=kind));value=h.joint_blocks(record,origin)if kind=='joint'else[h.support_block(record,origin)];emit(dict(kind='AFTER_NEW_saved_matrix_map_binding_RETURN',origin=origin,blocks=[dict(id=v['id'],dimension=v['dimension'])for v in value]));blocks.extend(value);sources.append(origin)
assert len(blocks)==112 and sum(v['kind'].startswith('joint:')for v in blocks)==63 and sum(v['kind'].startswith('support:')for v in blocks)==49
blockfiles=[]
for i,block in enumerate(blocks):
    emit(dict(kind='BEFORE_NEW_exact_block_serialization',id=block['id']));record=h.block_record(block);name=f'additional-block-{i:03d}.json';save(name,record);blockfiles.append(dict(id=block['id'],file=name,sha256=sha(O/name)));emit(dict(kind='AFTER_NEW_exact_block_serialization_RETURN',id=block['id'],file=name,sha256=sha(O/name)))
combined,rhs,packed=h.append_blocks(A,b,blocks,emit);assert combined.shape==(15840,1125)and len(packed)==112;assert (combined[:14099]-A).nnz==0 and np.array_equal(rhs[:14099],b)
sparse.save_npz(O/'supported-canonical-A.npz',combined);np.save(O/'supported-canonical-b.npy',rhs);save('additional-weighted-PSD-packing.json',packed)
unavailable=S/'e037-pointwise-support-diagnosis-v1/unavailable-nonconstant-source-localizers.json';binding=dict(schema='v2d.supported-conic-model-binding.v1',base_model=str(S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json'),base_model_sha256=parent['base_model_sha256'],base_canonical_A=str(basefiles[0]),base_canonical_A_sha256=sha(basefiles[0]),base_canonical_b=str(basefiles[1]),base_canonical_b_sha256=sha(basefiles[1]),base_packing=str(basefiles[2]),base_packing_sha256=sha(basefiles[2]),new_contract_report=str(contractpath),new_contract_report_sha256=sha(contractpath),source_map_files=sources,additional_block_files=blockfiles,additional_packing_sha256=sha(O/'additional-weighted-PSD-packing.json'),combined_A_sha256=sha(O/'supported-canonical-A.npz'),combined_b_sha256=sha(O/'supported-canonical-b.npy'),columns=1125,base_rows=14099,new_rows=1741,rows=15840,original_equalities=549,original_inequalities=5990,explicit_unit_box_rows=2250,original_conditional_PSD_cones=1180,new_joint_PSD_cones=63,new_support_PSD_cones=49,total_PSD_cones=1292,source_nonconstant_supports_UNAVAILABLE=5,source_unavailable_record=str(unavailable),source_unavailable_record_sha256=sha(unavailable),unknown109bands_region2_and_fixed_unqualified_wrist_preserved=True,all_original_boxes_caps_authority_and_acceptance_tolerances_unchanged=True,declared_single_NEW_evaluation=dict(maximum_episode_solver_calls=1,maximum_iterations=100,time_limit_seconds=60,threads=1,solver_tolerances='defaults',wall_guard_seconds=240,RSS_guard_bytes=2*1024**3,no_restart_or_continuation=True),new_variable_columns_or_moment_degree_increase=0,local_support_PSD_NOT_common_measure_or_robot_path=True)
save('supported-model-binding.json',binding);j.close();report=dict(schema='v2d.supported-conic-model-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/supported_conic_maps_v1.py'),canonical_base_reused_by_hash_prefix_unchanged=True,new_joint_blocks=63,new_support_blocks=49,new_PSD_rows=1741,total_rows=15840,total_columns=1125,total_PSD_cones=1292,independent_pre_solve_binding_verification_PENDING=True,new_solver_calls=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
save('report.json',report);print(json.dumps({k:v for k,v in report.items()if k!='all_outputs_sha256'}))
