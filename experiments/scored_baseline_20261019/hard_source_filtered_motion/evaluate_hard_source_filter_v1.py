"""SOLE declared bounded hard-training-source-filtered episode construction.

The direction model is heuristic. Only actual training/source-motion returns
can accept a method trial; none establishes original full-source/physical
qualification. No prior rejected vector or fullpixel/heldout value is an input.
"""
from pathlib import Path
import datetime,hashlib,json,math,os,sys,time
import numpy as np
from scipy import sparse
S=Path(__file__).resolve().parents[1];C=S/'compute';O=S/'e037-hard-source-filter-evaluation-v1';B=S/'e037-hard-source-filter-preflight-v1'
sys.path.insert(0,str(S/'e037-direct-SDP-runtime-v2/site-packages'))
sys.path.insert(0,str(Path('/Users/rohithiruku/v2d_hand_continuity_fix/scripts/track3')))
import clarabel
from temporal_pose_path import motion_summary
from hard_source_filter_direction_v1 import CONFIG,build_direction,actual_trial_decision
from hard_source_filter_binding_v1 import independently_bind
from hard_source_filter_inputs_v1 import load_inputs
from hard_training_conic_returns_v1 import snapshot
from object_training_constraint_oracle_v1 import TrainingSourceOracle

def plain(v):
 if isinstance(v,np.ndarray):return plain(v.tolist())
 if isinstance(v,np.generic):return plain(v.item())
 if isinstance(v,dict):return {str(k):plain(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x)for x in v]
 if isinstance(v,float)and not math.isfinite(v):return dict(nonfinite_float=repr(v))
 return v
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def array(n,**v):
 p=O/n;np.savez_compressed(p,**v);p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def save_problem(prefix,model):
 for k in ['A','P','raw_A']:
  p=O/(prefix+'-'+k+'-RETURN.npz');sparse.save_npz(p,model[k]);p.chmod(0o600)
  with p.open('rb')as f:os.fsync(f.fileno())
 vectors=array(prefix+'-vectors-RETURN.npz',b=model['b'],q=model['q'],raw_b=model['raw_b'],scales=model['scales'],upper_step=model['upper_step'],lower_step=model['lower_step'])
 meta={k:model[k]for k in ['rows','columns','nonnegative_rows','SOC4','blocks']};save(prefix+'-metadata-RETURN.json',meta)
 return dict(metadata=meta,vectors=vectors)

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):log.write(json.dumps(plain(v),allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
 declaration=read(S/'e037-hard-source-filter-contracts-v1/ONE-NEW-method-declaration-BEFORE.json')
 save('ONE-episode-declaration-BEFORE.json',declaration);emit(dict(kind='BEFORE_SOLE_NEW_episode',declaration=declaration))
 # Completed controls and first matrix evidence are not replayed.
 assert read(S/'e037-hard-source-filter-contracts-v1/report.json')['all_pass']
 assert read(S/'e037-hard-source-filter-preflight-binding-v1/report.json')['NEW_complete_matrix_binding']['passed']
 inputs=load_inputs();data=inputs['data'];state=inputs['state'];source=inputs['source'];geometry=inputs['geometry']
 save('NEW-input-provenance-BY-HASH.json',inputs['provenance'])
 with np.load(B/'NEW-initial-state-RETURN.npz')as z:
  for k in z.files:assert np.array_equal(z[k],state[k])
 current_phase=float(np.maximum(-.001-state['gap_proxy'],0).max(initial=0))
 source_returns=[];cache={};start=time.monotonic();major_records=[];trial_records=[];trials=0;accepted_steps=0;QP_calls=0;status='UNRESOLVED_major_cap'
 def training_data(f):
  if f not in cache:
   r=source['stats'][f]
   assert sha(r['partition']['path'])==r['partition']['sha256']and sha(r['point_data']['path'])==r['point_data']['sha256']
   with np.load(r['partition']['path'])as z:a=dict(training_mask=z['training_mask'],domain=z['domain'])
   with np.load(r['point_data']['path'])as z:a['points_C_m']=z['points_C_m']
   cache[f]=a
  return cache[f]
 oracle=TrainingSourceOracle(source['V'],source['faces'],source['K'],source['C0'],source['T'],source['Rcw'],source['stats'],training_data,emit,lambda name,vals:array(name+'.npz',**vals))
 array('NEW-initial-cached-state-RETURN.npz',**state)
 for major in range(CONFIG['maximum_major_steps']):
  if current_phase<=0:status='TERMINAL_training_motion_proxy_pass_REQUIRES_original_qualification';break
  if trials>=CONFIG['maximum_trials']:status='UNRESOLVED_trial_cap';break
  if time.monotonic()-start>=CONFIG['method_seconds']:status='UNRESOLVED_method_time_cap';break
  prefix=f'major-{major:02d}'
  emit(dict(kind='BEFORE_NEW_step_model',major=major,current_proxy_phase=current_phase,state_world_delta=state['world_delta']))
  array(prefix+'-state-RETURN.npz',**state)
  if major==0:
   model=read(B/'NEW-canonical-metadata-RETURN.json')
   for k in ['A','P','raw_A']:model[k]=sparse.load_npz(B/f'NEW-canonical-{k}-RETURN.npz')
   with np.load(B/'NEW-canonical-vectors-RETURN.npz')as z:model.update({k:z[k]for k in z.files})
   model['delta']=state['world_delta'];emit(dict(kind='AFTER_initial_matrix_REUSED_BY_HASH',major=major,preflight_binding_sha256=sha(S/'e037-hard-source-filter-preflight-binding-v1/report.json')))
  else:
   model=build_direction(state,data);ret=save_problem(prefix,model);emit(dict(kind='AFTER_NEW_step_model_constructor_RETURN',major=major,actual_RETURN=ret))
  if major==0:
   binding=read(S/'e037-hard-source-filter-preflight-binding-v1/NEW-independent-full-matrix-binding-RETURN.json')
  else:
   binding=independently_bind(model,state,data,CONFIG)
   residual=array(prefix+'-independent-binding-residuals-RETURN.npz',**binding.pop('residuals'))
   binding['residual_RETURN']=residual
  save(prefix+'-binding-RETURN.json',binding);emit(dict(kind='AFTER_NEW_pre_solver_binding_RETURN',major=major,actual_RETURN=binding));assert binding['passed']
  if time.monotonic()-start>=CONFIG['method_seconds']:status='UNRESOLVED_method_time_cap';break
  settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=CONFIG['QP_max_iterations'];settings.time_limit=CONFIG['QP_seconds']
  save(prefix+'-settings-RETURN.json',snapshot(settings))
  cones=[clarabel.NonnegativeConeT(model['nonnegative_rows'])]+[clarabel.SecondOrderConeT(4)for _ in range(model['SOC4'])]
  save(prefix+'-cones-RETURN.json',[snapshot(x)for x in cones])
  emit(dict(kind='BEFORE_NEW_QP_constructor',major=major))
  solver=clarabel.DefaultSolver(model['P'],model['q'],model['A'],model['b'],cones,settings)
  save(prefix+'-constructor-RETURN.json',snapshot(solver));emit(dict(kind='AFTER_NEW_QP_constructor_RETURN',major=major))
  callback_count=[0]
  def cb(info):
   k=callback_count[0];callback_count[0]+=1;p=save(prefix+f'-callback-{k:03d}-RETURN.json',snapshot(info));emit(dict(kind='AFTER_NEW_actual_info_callback_RETURN',major=major,callback=k,actual_RETURN=p));return False
  solver.set_termination_callback(cb);QP_calls+=1;sol=solver.solve()
  rawx=np.asarray(sol.x,float);rawz=np.asarray(sol.z,float);raws=np.asarray(sol.s,float)
  raw=array(prefix+'-raw-x-z-s-RETURN.npz',x=rawx,z=rawz,s=raws)
  summary=dict(major=major,status=str(sol.status),actual_solution=snapshot(sol,exclude=('x','z','s')),info=snapshot(solver.get_info()),raw_RETURN=raw,callbacks=callback_count[0])
  save(prefix+'-solver-RETURN.json',summary);emit(dict(kind='AFTER_NEW_actual_solver_RETURN',actual_RETURN=summary));major_records.append(summary)
  if str(sol.status)not in ['Solved','AlmostSolved']or rawx.shape!=(454,)or not np.isfinite(rawx).all():status='UNRESOLVED_step_solver_return';break
  predicted=current_phase-float(rawx[-1])
  save(prefix+'-prediction-RETURN.json',dict(current_phase=current_phase,raw_QP_rho=float(rawx[-1]),predicted_reduction=predicted,not_original_gate=True))
  if predicted<=0:status='UNRESOLVED_no_predicted_descent';break
  accepted=False
  for alpha in CONFIG['trial_fractions']:
   if trials>=CONFIG['maximum_trials']or time.monotonic()-start>=CONFIG['method_seconds']:break
   trial_index=trials;trials+=1;trial_delta=state['world_delta']+alpha*rawx[:-1].reshape(151,3)
   # Separate, predeclared algorithm trial; raw QP return stays immutable.
   trial_hash=hashlib.sha256(np.ascontiguousarray(trial_delta).tobytes()).hexdigest();pid=f'trial-{trial_index:02d}-{trial_hash[:16]}'
   sealed=array(pid+'-SEALED-proposal-BEFORE.npz',world_delta=trial_delta,raw_QP_step=rawx,alpha=np.array(alpha),original_world_poses=inputs['world_poses'])
   emit(dict(kind='BEFORE_NEW_actual_trial',trial=trial_index,major=major,alpha=alpha,proposal_id=pid,sealed=sealed,delta_sha256=trial_hash))
   finite=bool(np.isfinite(trial_delta).all());box=finite and bool(np.max(np.abs(trial_delta))<=CONFIG['world_box_halfwidth_m'])
   if not box:
    record=dict(trial=trial_index,major=major,alpha=alpha,proposal_id=pid,delta_sha256=trial_hash,status='REJECTED_algorithmic_box_or_nonfinite',accepted=False)
    save(pid+'-decision-RETURN.json',record);emit(dict(kind='AFTER_NEW_trial_decision_RETURN',actual_RETURN=record));trial_records.append(record);continue
   # All retained directions for this NEW point; saved initial F is never replayed.
   axis_values=np.empty(len(geometry['gaps0']),float);active=np.empty(len(geometry['pair_frames']),np.int64);gap=np.empty(len(active))
   for pair,f in enumerate(geometry['pair_frames']):
    a,b=map(int,geometry['ptr'][pair:pair+2]);v=np.asarray(geometry['axes'][a:b])@trial_delta[int(f)]+geometry['gaps0'][a:b]
    axis_values[a:b]=v;j=int(np.argmax(v));active[pair]=a+j;gap[pair]=v[j]
   geo=array(pid+'-NEW-all-axis-union-RETURN.npz',all_axis_values=axis_values,active_axis_indices=active,gap_proxy=gap,world_delta=trial_delta)
   emit(dict(kind='AFTER_NEW_actual_union_proxy_RETURN',proposal_id=pid,actual_RETURN=geo))
   trial_phase=float(np.maximum(-.001-gap,0).max(initial=0))
   world=inputs['world_poses'].copy();world[:,:3,3]+=trial_delta
   pose_ref=array(pid+'-NEW-raw-world-poses-RETURN.npz',world_from_object=world)
   emit(dict(kind='AFTER_NEW_derived_pose_constructor_RETURN',proposal_id=pid,actual_RETURN=pose_ref))
   motion=motion_summary(world,inputs['timestamps']);save(pid+'-NEW-original-motion-RETURN.json',motion)
   motion_checks={k:bool(motion[k]<=v+1e-9)for k,v in inputs['caps'].items()};save(pid+'-NEW-original-motion-decisions-RETURN.json',motion_checks)
   emit(dict(kind='AFTER_NEW_actual_original_motion_RETURN',proposal_id=pid,actual_RETURN=motion,decisions=motion_checks))
   returned=[]
   for f in range(151):
    if time.monotonic()-start>=CONFIG['method_seconds']:
     returned.extend(dict(frame=k,proposal_id=pid,status='UNRESOLVED_NOT_QUERIED_method_deadline',passes_TRAINING_construction=False)for k in range(f,151));break
    returned.append(oracle.query(f,trial_delta[f],pid))
   source_ref=save(pid+'-ALL151-actual-training-RETURNS.json',returned)
   emit(dict(kind='AFTER_NEW_actual_ALL151_training_RETURNS',proposal_id=pid,actual_RETURN=source_ref))
   nonzero={f for f in range(151)if not np.array_equal(trial_delta[f],np.zeros(3))}
   decision=actual_trial_decision(current_phase,trial_phase,returned,motion_checks,box,list(range(151)),pid,nonzero)
   record=dict(trial=trial_index,major=major,alpha=alpha,proposal_id=pid,delta_sha256=trial_hash,sealed_proposal=sealed,
    proxy_phase_m=trial_phase,current_phase_m=current_phase,source_RETURN=source_ref,geometry_RETURN=geo,pose_RETURN=pose_ref,
    decision=decision,accepted=decision['accepted'],actual_source_failed_frames=[r['frame']for r in returned if not r.get('passes_TRAINING_construction')],
    unknown_source_frames=[r['frame']for r in returned if r.get('status','').startswith('UNRESOLVED')])
   save(pid+'-decision-RETURN.json',record);emit(dict(kind='AFTER_NEW_trial_decision_RETURN',actual_RETURN=record));trial_records.append(record)
   if decision['accepted']:
    source_returns=returned;accepted=True;accepted_steps+=1;current_phase=trial_phase
    med=np.array([r['training_median_m']for r in returned]);grad=np.array([r.get('local_median_world_gradient',source['stats'][f]['local_median_world_gradient'])for f,r in enumerate(returned)])
    state=dict(world_delta=trial_delta,active_normals=np.asarray(geometry['axes'][active]).copy(),gap_proxy=gap,median=med,median_gradient=grad,active_axis_indices=active)
    break
  if not accepted:
   status='UNRESOLVED_method_time_cap'if time.monotonic()-start>=CONFIG['method_seconds']else('UNRESOLVED_trial_cap'if trials>=CONFIG['maximum_trials']else'UNRESOLVED_no_hard_source_motion_feasible_descent')
   break
 if current_phase<=0:status='TERMINAL_training_motion_proxy_pass_REQUIRES_original_qualification'
 terminal=array('ONE-algorithm-terminal-RETURN.npz',**state)
 world=inputs['world_poses'].copy();world[:,:3,3]+=state['world_delta']
 array('UNQUALIFIED-terminal-derived-object-poses.npz',world_from_object=world,world_translation=state['world_delta'])
 report=dict(schema='v2d.NEW-hard-source-filter-evaluation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  status=status,method_seconds=time.monotonic()-start,QP_calls=QP_calls,actual_trial_vectors=trials,accepted_method_steps=accepted_steps,
  terminal_proxy_phase_m=current_phase,terminal_RETURN=terminal,major_RETURNS=major_records,trial_RETURNS=trial_records,
  final_return_is_algorithm_terminal_NOT_best_intermediate=True,
  terminal_is_derived_object_motion_NOT_source_observation_or_reference=True,
  original_F_robot_commands_object_rotations_source_geometry_parameters_gates_UNCHANGED=True,
  no_original_fullpixel_heldout_source_rescue_or_fitting=True,
  original_fullpixel_whole_robot_source_native_self_swept_motor_grasp_STILL_REQUIRED=True,
  no_source_motion_component_fullreference_task_policy_AUC_qualification=True,
  no_native_position_physics_GPU_PPO=True,all_raw_QP_trial_and_source_returns_retained_UNALTERED=True)
 save('report.json',report);emit(dict(kind='AFTER_SOLE_NEW_episode_RETURN',actual_RETURN=report));print(json.dumps(plain({k:v for k,v in report.items()if k not in ['major_RETURNS','trial_RETURNS']})))
