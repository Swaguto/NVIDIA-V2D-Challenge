"""SOLE NEW bounded nonlinear all-clock object construction; no failed point promotion."""
from pathlib import Path
import datetime,hashlib,json,os,sys,time,traceback,warnings
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/track3'))
import cv2,numpy as np,trimesh
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from object_dynamic_runtime_v1 import DynamicRuntime,MethodBudgetExceeded
from object_dynamic_visibility_v1 import CONFIG
from select_pose_hypothesis import observed_fit
from foundationpose_track import pose_rejection
from camera_prior import transport
from temporal_pose_path import motion_summary
from thumb_conic_geometry_v1 import plain,journal
O=S/'e037-object-dynamic-evaluation-v1';D=S/'e037-object-dynamic-runtime-v1';F=S/'e037-thumb-ray-independent-validation-v1/frozen';start=time.monotonic();cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
def snpz(name,**kw):
 p=O/name;np.savez_compressed(p,**kw);p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 bind=read(S/'e037-object-dynamic-runtime-binding-v2/report.json');assert bind['all_actual_runtime_bindings_pass'];bindings=read(S/'e037-object-dynamic-runtime-binding-v2/actual-runtime-data-bindings.json')
 for n,h in bindings['data_sha256'].items():assert sha(D/n)==h
 declaration=dict(schema='v2d.ONE-NEW-dynamic-source-union-support-object-method.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),config=CONFIG,actual_immutable_runtime_bindings_sha256=sha(S/'e037-object-dynamic-runtime-binding-v2/report.json'),starting_reference_F=True,initial_normalized_vector='453zeros ONLY, no failed/intermediate warmstart',ONE_method_call_NO_restart_retry_tolerance_weight_profile_limit_sweep=True,all131missing_source_and_source_truth_limits_preserved=True,source_uses_original_FIXED_even24samples_only=True,actual_original_fullpixel_and_motion_gates_POSTconstruction_only=True,native_positions_physics_GPU_PPO_NOT_executed=True);save('ONE-declared-dynamic-method.json',declaration);emit(dict(kind='BEFORE_ONE_NEW_dynamic_object_method',proposal=declaration));runtime=DynamicRuntime(D,emit,snpz);result=None;exception=None
 emit(dict(kind='BEFORE_ONE_NEW_actual_sparse_nonlinear_optimizer_call',method='trf',tr_solver='lsmr',max_nfev=40,default_ftol_xtol_gtol=1e-8,bounds=[-1,1],normalized_variables=453,original_algorithm_box_m=.02,method_seconds=170,threads=1,not_original_candidate_acceptance_tolerances=True))
 try:
  raw=least_squares(runtime.fun,np.zeros(453),jac=runtime.jac,bounds=(-np.ones(453),np.ones(453)),method='trf',tr_solver='lsmr',max_nfev=40,callback=runtime.callback);result={k:v for k,v in raw.items()if k!='jac'}
  if sparse.issparse(raw.jac):sparse.save_npz(O/'ONE-actual-optimizer-Jacobian-RETURN.npz',raw.jac)
  else:np.save(O/'ONE-actual-optimizer-Jacobian-RETURN.npy',raw.jac)
  save('ONE-actual-optimizer-RETURN.json',result);emit(dict(kind='AFTER_ONE_NEW_actual_optimizer_RETURN',actual_RETURN=result,raw_values_UNALTERED=True))
 except (MethodBudgetExceeded,ValueError,FloatingPointError,StopIteration)as exc:
  exception=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),actual_optimizer_RETURN_NOT_produced=True,all_recorded_states_REJECTED_as_candidates=True);save('ONE-actual-method-exception.json',exception);emit(dict(kind='AFTER_ONE_NEW_actual_method_exception_RETURN',actual_RETURN=exception))
 out=dict(schema='v2d.ONE-NEW-dynamic-source-union-support-evaluation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_optimizer_calls=1,actual_function_evaluations=runtime.calls,actual_callback_returns=runtime.callback_count,actual_optimizer_RETURN_produced=result is not None,method_exception=exception,optimizer_success=None if result is None else bool(result['success']),optimizer_status=None if result is None else int(result['status']),optimizer_message=None if result is None else str(result['message']),actual_returned_nfev=None if result is None else int(result['nfev']),actual_last_model_stats=None if runtime.last is None else runtime.last['stats'],all_original_F_source_geometry_parameters_gates_UNCHANGED=True,no_old_solver_control_query_audit_replayed=True,original_source_evaluations=0,wholeclock_motion_evaluations=0,new_source_motion_object_components=0,new_full_reference_or_trajectory_candidates=0,native_position_physics_GPU_PPO_calls=0,no_task_policy_or_score_qualified=True,source_failed_frames=[])
 if result is None or not result['success']:out['construction_status']='REJECTED_failed_method_or_optimizer_RETURN_NO_pose_proposal';out['all_intermediate_and_terminal_vectors_REJECTED']=True
 else:
  x=np.asarray(result['x'],float)
  if x.shape!=(453,)or not np.isfinite(x).all()or abs(x).max()>.1/.1:out['construction_status']='REJECTED_actual_normalized_vector_outside_declared_bounds_NO_pose_proposal'
  else:
   delta=CONFIG['construction_halfwidth_m']*x.reshape(151,3);emit(dict(kind='BEFORE_NEW_raw_dynamic_object_pose_proposal',raw_translation_world_m=delta,not_reference_or_observation=True))
   with np.load(D/'dynamic-source-data.npz')as z:world=z['original_world_from_object'].copy();camera=z['original_camera_from_object'].copy();times=z['timestamps_s'];axis=z['frame_indices'];Rcw=z['camera_from_world_linear'];K=z['K']
   with np.load(F/'targets.npz')as z:native=z['object_poses'].copy();robot=z['robot_qpos'].copy()
   with np.load(S/'plane-aligned-pose-temporal-path-v4/inferred-object-trajectory.npz')as z:cameras=z['candidate_camera_c2w'];flags=z['original_object_observed_flags']
   stage=np.asarray(read(S/'plane-aligned-articulated-native-validation-v2/frozen/command-proposal-report.json')['stage_rotation_from_inferred_support_normal']);world[:,:3,3]+=delta;camera[:,:3,3]+=np.einsum('tij,tj->ti',Rcw,delta,optimize=False);native[:,0,:3]+=np.einsum('ij,tj->ti',stage,delta,optimize=False);proposal=snpz('RAW-dynamic-object-pose-proposal-before-gates.npz',raw_world_translation=delta,world_from_object=world,camera_from_object=camera,native_object_poses=native,unchanged_F_robot_qpos=robot,timestamps_s=times,frame_indices=axis,original_object_observed_flags=flags);emit(dict(kind='AFTER_NEW_raw_dynamic_object_pose_proposal_RETURN',proposal=proposal,rotations_and_robot_commands_UNCHANGED=True))
   emit(dict(kind='BEFORE_NEW_actual_fullclock_original_object_motion_gate',proposal=proposal));motion=motion_summary(world,times);caps=runtime.caps;over=[k for k in caps if motion[k]>caps[k]+1e-9];motion_ret=dict(actual_motion=motion,original_caps=caps,original_object_comparison_tolerance=1e-9,failed_caps=over);save('actual-original-wholeclock-motion-RETURN.json',motion_ret);emit(dict(kind='AFTER_NEW_actual_fullclock_original_object_motion_RETURN',actual_RETURN=motion_ret));out['wholeclock_motion_evaluations']=1;out['object_motion_failed_caps']=over
   manifest=read(F/'manifest.json');mesh=trimesh.load(F/manifest['objects'][0]['mesh'],force='mesh',process=False);vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces);tree=cKDTree(vertices);bounds=np.stack([vertices.min(0),vertices.max(0)]);source=[]
   for f in range(151):
    if time.monotonic()-start>225:out['postgate_status']='UNRESOLVED_postconstruction_wall_cap';break
    a=read(S/f'e037-object-source-authority-v1/frame-{f:03d}-source-authority.json');dp=Path(a['depth_path']);mp=Path(a['mask_path']);assert sha(dp)==a['depth_sha256']and sha(mp)==a['mask_sha256']
    with np.load(dp)as z:depth=z['depth_m'];valid=z['left_right_consistent']
    mask=cv2.imread(str(mp),0)>127;emit(dict(kind='BEFORE_NEW_original_fullpixel_source_gate',frame=f,new_camera_object_pose=camera[f],original_thresholds_COPIED=a['current_accepted_source_thresholds_COPIED'],fullpixel_and_heldout_values_NOT_used_by_optimizer=True))
    with warnings.catch_warnings(record=True)as ws:
     warnings.simplefilter('always');actual=observed_fit(vertices,faces,tree,camera[f],depth,valid,mask,K);prior=None if f==0 else transport(camera[f-1],np.linalg.inv(cameras[f-1]),np.linalg.inv(cameras[f]));reject,support=pose_rejection(camera[f],depth,valid,(mask*255).astype(np.uint8),K,bounds,True,prior)
    ret=dict(frame=f,actual_source_RETURN=actual,tracker_rejection_RETURN=reject,tracker_support_RETURN=support,original_object_observed_flag=bool(flags[f]),warnings=[str(w.message)for w in ws],warning_cause_NOT_proven=True);save(f'frame-{f:03d}-actual-source-RETURN-before-decision.json',ret);emit(dict(kind='AFTER_NEW_original_fullpixel_source_gate_RETURN',actual_RETURN=ret));threshold=a['current_accepted_source_thresholds_COPIED'];before=a['old_strict_source_comparison_COPIED'];ok=bool(flags[f]and reject is None and actual.get('quality',0)>=threshold['quality']-1e-10 and actual.get('silhouette_iou',0)>=threshold['silhouette_iou']-1e-10 and actual.get('median_measured_vertex_distance_m',float('inf'))<=threshold['median_measured_vertex_distance_m']+1e-10 and actual.get('quality',0)>before['quality']and actual.get('silhouette_iou',0)>=before['silhouette_iou']and actual.get('median_measured_vertex_distance_m',float('inf'))<before['median_measured_vertex_distance_m']);source.append(dict(**ret,original_source_nonregression_passed=ok,original_source_comparison_tolerance=1e-10));out['original_source_evaluations']+=1;emit(dict(kind='AFTER_NEW_original_fullpixel_source_decision',frame=f,passed=ok))
   save('all-actual-source-gate-RETURNS.json',source);out['source_failed_frames']=[r['frame']for r in source if not r['original_source_nonregression_passed']]
   if len(source)==151 and not over and not out['source_failed_frames']:
    import shutil
    shutil.copyfile(proposal['path'],O/'UNQUALIFIED-dynamic-object-source-motion-component.npz');out['new_source_motion_object_components']=1;out['construction_status']='PROPOSED_object_source_motion_component_ONLY_whole_native_self_swept_motor_grasp_source_PENDING'
   else:out['construction_status']='REJECTED_actual_original_source_or_motion_gates';out['raw_and_intermediate_vectors_REJECTED_as_reference']=True
 out['wall_seconds']=time.monotonic()-start;save('report.json',out);emit(dict(kind='AFTER_ONE_NEW_dynamic_object_method_RETURN',actual_RETURN=out));print(json.dumps(out))
