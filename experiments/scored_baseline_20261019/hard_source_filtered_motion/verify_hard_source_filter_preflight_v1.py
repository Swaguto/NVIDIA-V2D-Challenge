"""Independent NEW initial model binding; no builder, source or solver replay."""
from pathlib import Path
import datetime,hashlib,json,os
import numpy as np
from scipy import sparse
from hard_source_filter_direction_v1 import CONFIG
from hard_source_filter_binding_v1 import independently_bind

S=Path(__file__).resolve().parents[1];O=S/'e037-hard-source-filter-preflight-binding-v1';B=S/'e037-hard-source-filter-preflight-v1';D=S/'e037-object-dynamic-runtime-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {str(k):plain(x)for k,x in v.items()}
 return v
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):log.write(json.dumps(plain(v),allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
 scope=dict(NEW_complete_first_model_binding_ONLY=True,no_builder_source_NN_raster_solver_pose_native_or_GPU=True,
  original_authority_inputs_and_completed_controls_reused_BY_HASH=True)
 save('ONE-independent-preflight-scope-BEFORE.json',scope);emit(dict(kind='BEFORE_NEW_binding',scope=scope))
 provenance=read(B/'NEW-input-provenance-BY-HASH.json')
 for p,h in provenance.items():assert sha(p)==h
 with np.load(B/'NEW-initial-state-RETURN.npz')as z:state={k:z[k]for k in z.files}
 with np.load(B/'NEW-linear-data-RETURN.npz')as z:data={k:z[k]for k in z.files}
 data['motion_M']=sparse.load_npz(B/'NEW-motion-M-BY-HASH.npz')
 meta=read(B/'NEW-canonical-metadata-RETURN.json');problem=dict(meta)
 for k in ['A','P','raw_A']:problem[k]=sparse.load_npz(B/f'NEW-canonical-{k}-RETURN.npz')
 with np.load(B/'NEW-canonical-vectors-RETURN.npz')as z:problem.update({k:z[k]for k in z.files})
 bound=independently_bind(problem,state,data,CONFIG)
 p=O/'NEW-independent-full-matrix-residuals-RETURN.npz';np.savez_compressed(p,**bound['residuals']);p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 summary={k:v for k,v in bound.items()if k!='residuals'};summary['residual_RETURN']=dict(path=str(p),sha256=sha(p))
 save('NEW-independent-full-matrix-binding-RETURN.json',summary);emit(dict(kind='AFTER_NEW_complete_binding_RETURN',actual_RETURN=summary));assert bound['passed']
 # New copied-input bindings, not an old native/source/marginal audit.
 stats=read(S/'e037-training-source-constraints-v1/all151-NEW-training-source-statistics.json')
 assert np.array_equal(state['world_delta'],np.zeros((151,3)))
 assert np.array_equal(data['baseline_median'],[r['training_mesh_vertex_median_m']for r in stats])
 assert np.array_equal(state['median'],data['baseline_median'])
 assert np.array_equal(state['median_gradient'],[r['local_median_world_gradient']for r in stats])
 caps=read(D/'runtime-config-and-original-authority.json')['original_object_caps']
 assert np.array_equal(data['motion_caps'],np.r_[np.full(150,caps['maximum_linear_speed_m_s']),np.full(149,caps['maximum_linear_acceleration_m_s2'])])
 with np.load(D/'dynamic-source-data.npz')as z:
  assert np.array_equal(data['Rcw'],z['camera_from_world_linear'])
  assert np.array_equal(data['minimum_C0_z'],z['C0'][:,:,2].min(axis=1))
 with np.load(S/'e037-object-dynamic-evaluation-v1/evaluation-000-NEW-union-support-RETURN.npz')as z:
  assert np.array_equal(state['gap_proxy'],z['gap_proxy_m'])and np.array_equal(state['active_axis_indices'],z['active_axis_indices'])
 axes=np.load(D/'ALL-retained-axis-world-coefficients.npy',mmap_mode='r')
 assert np.array_equal(state['active_normals'],axes[state['active_axis_indices']])
 assert (data['motion_M']-sparse.load_npz(D/'dynamic-motion-linear-map.npz')).nnz==0
 assert np.array_equal(data['motion_base'],np.load(D/'dynamic-motion-constants.npy'))
 assert meta['rows']==19926 and meta['columns']==454 and meta['nonnegative_rows']==18730 and meta['SOC4']==299
 report=dict(schema='v2d.NEW-hard-source-filter-preflight-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  rows=meta['rows'],columns=meta['columns'],nonnegative_rows=meta['nonnegative_rows'],SOC4=meta['SOC4'],
  NEW_complete_matrix_binding=summary,NEW_median_gradient_and_stat_copies=151,
  NEW_initial_axis_branch_rows=17520,NEW_source_depth_rows=151,NEW_motion_cones=299,
  old_data_controls_source_authority_reused_BY_HASH_NOT_replayed=True,
  no_episode_solver_source_NN_raster_pose_native_physics_GPU=True,
  original_gates_parameters_F_and_all_failed_returns_UNCHANGED=True)
 save('report.json',report);emit(dict(kind='AFTER_NEW_preflight_binding_RETURN',actual_RETURN=report));print(json.dumps(plain(report)))
