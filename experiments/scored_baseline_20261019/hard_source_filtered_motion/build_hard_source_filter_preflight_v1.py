"""ONE NEW full-clock initial step model; no episode solver or source query."""
from pathlib import Path
import datetime,hashlib,json,math,os
import numpy as np
from scipy import sparse
from hard_source_filter_direction_v1 import CONFIG,build_direction
from hard_source_filter_inputs_v1 import load_inputs

S=Path(__file__).resolve().parents[1];O=S/'e037-hard-source-filter-preflight-v1'
def plain(v):
 if isinstance(v,np.ndarray):return plain(v.tolist())
 if isinstance(v,np.generic):return plain(v.item())
 if isinstance(v,dict):return {str(k):plain(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x)for x in v]
 return v
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def array(n,**v):
 p=O/n;np.savez_compressed(p,**v);p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):log.write(json.dumps(plain(v),allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
 save('ONE-NEW-fullclock-preflight-declaration-BEFORE.json',dict(config=CONFIG,
  source='SAVED F zero state/source medians+chosen gradients/initial union axes BY HASH',
  source_gradients_predictor_ONLY_no_IoU_derivative=True,
  no_new_pose_solver_source_NN_raster_or_physical_call=True,
  complete_actual_source_motion_filter_required_before_accepting_any_future_trial=True))
 inputs=load_inputs();state=inputs['state'];data=inputs['data']
 save('NEW-input-provenance-BY-HASH.json',inputs['provenance'])
 state_ref=array('NEW-initial-state-RETURN.npz',**state)
 data_ref=array('NEW-linear-data-RETURN.npz',**{k:v for k,v in data.items()if k!='motion_M'})
 sparse.save_npz(O/'NEW-motion-M-BY-HASH.npz',data['motion_M'])
 emit(dict(kind='BEFORE_NEW_canonical_constructor',state=state_ref,data=data_ref))
 model=build_direction(state,data)
 # Entire raw and conditioned constructor returns precede any binding/assertion.
 for k in ['A','P','raw_A']:sparse.save_npz(O/f'NEW-canonical-{k}-RETURN.npz',model[k])
 vec=array('NEW-canonical-vectors-RETURN.npz',b=model['b'],q=model['q'],raw_b=model['raw_b'],scales=model['scales'],upper_step=model['upper_step'],lower_step=model['lower_step'])
 meta={k:model[k]for k in ['rows','columns','nonnegative_rows','SOC4','blocks']}
 save('NEW-canonical-metadata-RETURN.json',meta);emit(dict(kind='AFTER_NEW_COMPLETE_constructor_RETURN',actual_RETURN=meta,vectors=vec))
 assert model['columns']==454 and model['rows']==19926 and model['SOC4']==299
 rho=float(np.maximum(-.001-state['gap_proxy'],0).max(initial=0))
 record=dict(schema='v2d.NEW-hard-source-filter-preflight.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  rows=model['rows'],columns=model['columns'],nonnegative_rows=model['nonnegative_rows'],SOC4=model['SOC4'],
  median_SEARCH_predictor_rows=151,strict_depth_SEARCH_predictor_rows=151,geometry_SEARCH_branch_rows=17520,
  initial_proxy_phase_m=rho,raw_A_nonzeros=model['raw_A'].nnz,
  config=CONFIG,initial_F_source_geometry_BY_HASH_NOT_requeried=True,
  no_episode_source_NN_raster_solver_pose_native_physics_GPU=True,
  initial_model_NOT_original_source_or_physical_feasibility_certificate=True,
  original_caps_tolerances_source_F_parameters_gates_UNCHANGED=True)
 save('report.json',record);emit(dict(kind='AFTER_NEW_preflight_RETURN',actual_RETURN=record));print(json.dumps(record))
