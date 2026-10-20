"""SOLE NEW bounded median-conic construction, followed by actual hard filters.

No restart, warmstart or rejected-point rescue. Actual source/native authority
never follows from a solver status or a numerically guarded proxy cone.
"""
from pathlib import Path
import datetime,hashlib,json,os,sys,time,warnings
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'e037-direct-SDP-runtime-v2/site-packages'),str(R/'scripts/track3')]
import clarabel,cv2,numpy as np,trimesh
from scipy import sparse
from hard_training_conic_returns_v1 import snapshot
from object_training_constraint_oracle_v1 import TrainingSourceOracle
from training_orderstat_support_v1 import source_filter_decision
from temporal_pose_path import motion_summary
from thumb_conic_geometry_v1 import plain,journal

O=S/'e037-hard-training-source-evaluation-v2';M=S/'e037-hard-training-source-model-v1';D=S/'e037-object-dynamic-runtime-v1';T=S/'e037-training-source-constraints-v1';F=S/'e037-thumb-ray-independent-validation-v1/frozen';start=time.monotonic()
cv2.setNumThreads(1);read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def arrays(n,v):
 p=O/(n if n.endswith('.npz')else n+'.npz');np.savez_compressed(p,**v);p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def deadline():
 if time.monotonic()-start>225:raise TimeoutError('ONE225second overall internal evaluation cap; no point promotion')

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 bind=read(S/'e037-hard-training-source-binding-v1/report.json');meta=read(M/'canonical-metadata.json');assert bind['all_bindings_pass']
 declaration=dict(schema='v2d.ONE-hard-training-source-conic-evaluation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  model_hashes=bind['canonical_hashes'],independent_binding_sha256=sha(S/'e037-hard-training-source-binding-v1/report.json'),
  NEW5arithmetic_and_NEWcallback_contracts_reused_BY_HASH=True,
  formulation='454columns,286907SOC4,18428nonnegative,1166056rows; allsavedmedianproxies + initialFselectedgeometry + original299motion',
  solver_iterations=100,solver_seconds=60.,threads=1,default_tolerances=True,overall_internal_seconds=225,
  ONE_solve_NO_restart_retry_warmstart_continuation_or_tolerance_weight_degree_profile_limit_sweep=True,
  source_training_raster_median_quality_ALL151_hard_nominal_POSTconstruction=True,
  original_all6object_motion_gates_actual_p95_original1e_9_POSTconstruction=True,
  fullpixel_heldout_original_source_POSTconstruction_IF_training_and_motion_pass_ONLY=True,
  original_source_comparison_tolerance=1e-10,raw_x_z_s_UNALTERED=True,
  no_native_position_physics_GPU_PPO_or_trajectory_qualification=True)
 save('ONE-declared-evaluation-BEFORE-solver.json',declaration);emit(dict(kind='BEFORE_ONE_NEW_episode_evaluation',declaration=declaration))
 for name,h in bind['canonical_hashes'].items():assert sha(M/name)==h
 A=sparse.load_npz(M/'canonical-A.npz');P=sparse.load_npz(M/'canonical-P.npz')
 with np.load(M/'canonical-vectors.npz')as z:b=z['b'];q=z['q']
 settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100;settings.time_limit=60.;settings.max_threads=1
 cones=[clarabel.NonnegativeConeT(meta['nonnegative_rows'])]+[clarabel.SecondOrderConeT(4)for _ in range(meta['SOC4'])]
 cone_return=dict(actual_count=len(cones),actual_types=sorted(set(type(c).__name__ for c in cones)),actual_dimensions_counted={str(k):sum(c.dim==k for c in cones)for k in {c.dim for c in cones}},actual_total_rows=sum(c.dim for c in cones))
 save('actual-cone-constructors-RETURN.json',cone_return);emit(dict(kind='AFTER_NEW_cone_constructor_RETURN',actual_RETURN=cone_return));assert cone_return['actual_total_rows']==A.shape[0]
 emit(dict(kind='BEFORE_ONE_NEW_solver_constructor',actual_settings=dict(max_iter=settings.max_iter,time_limit=settings.time_limit,max_threads=settings.max_threads,tol_gap_abs=settings.tol_gap_abs,tol_gap_rel=settings.tol_gap_rel,tol_feas=settings.tol_feas)))
 solver=clarabel.DefaultSolver(P,q,A,b,cones,settings);constructor=dict(type=type(solver).__name__,opaque_repr=repr(solver),actual_initial_info=snapshot(solver.get_info()));save('actual-solver-constructor-RETURN.json',constructor);emit(dict(kind='AFTER_ONE_NEW_solver_constructor_RETURN',actual_RETURN=constructor))
 callbacks=[]
 def callback(info):
  ret=snapshot(info);save(f'callback-{len(callbacks):03d}-actual-info-RETURN.json',ret);emit(dict(kind='AFTER_NEW_actual_solver_callback_RETURN',callback=len(callbacks),actual_RETURN=ret));callbacks.append(ret);return False
 registration=solver.set_termination_callback(callback);save('actual-callback-registration-RETURN.json',dict(actual_RETURN=registration));deadline();emit(dict(kind='BEFORE_ONE_NEW_episode_solve'))
 sol=solver.solve();raw=arrays('ONE-actual-primal-dual-slack-RETURN',dict(x=np.asarray(sol.x,float),z=np.asarray(sol.z,float),s=np.asarray(sol.s,float)));returned=snapshot(sol,exclude=('x','z','s'));save('ONE-actual-solver-RETURN.json',dict(**returned,raw_arrays=raw));emit(dict(kind='AFTER_ONE_NEW_episode_solver_RETURN',actual_RETURN=returned,raw_arrays=raw,raw_values_UNALTERED=True));deadline()
 status=str(sol.status);x=np.asarray(sol.x,float);summary=dict(schema='v2d.ONE-hard-training-source-evaluation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),episode_solver_calls=1,actual_solver_status=status,actual_iterations=int(sol.iterations),actual_solve_seconds=float(sol.solve_time),actual_callbacks=len(callbacks),raw_x_z_s_UNALTERED=True,actual_training_queries=0,original_source_queries=0,wholeclock_motion_queries=0,new_source_motion_components=0,new_whole_reference_candidates=0,new_native_position_physics_GPU_PPO=0,whole_trajectory_qualified=False,original_F_source_masks_parameters_tolerances_gates_UNCHANGED=True)
 if status not in ['Solved','AlmostSolved']or x.shape!=(454,)or not np.isfinite(x).all():summary['construction_status']='UNRESOLVED_solver_RETURN_NO_pose_proposal'
 elif np.any(np.abs(x[:453])>.02)or x[453]<0 or x[453]>.05:
  summary['construction_status']='REJECTED_actual_raw_vector_outside_declared_finite_box_NO_repair';summary['box_failures']=int(np.count_nonzero(np.abs(x[:453])>.02)+(x[453]<0)+(x[453]>.05))
 else:
  delta=x[:453].reshape(151,3);summary['actual_raw_phase_slack_m']=float(x[453]);emit(dict(kind='BEFORE_NEW_raw_object_pose_constructor',raw_translation=delta,phase_slack_m=float(x[453]),not_a_reference_or_observation=True))
  with np.load(D/'dynamic-source-data.npz')as z:world=z['original_world_from_object'].copy();camera=z['original_camera_from_object'].copy();RC=z['camera_from_world_linear'];C0=z['C0'];K=z['K'];times=z['timestamps_s'];frame_axis=z['frame_indices']
  original_world_rotation=world[:,:3,:3].copy();original_camera_rotation=camera[:,:3,:3].copy();world[:,:3,3]+=delta;camera[:,:3,3]+=np.einsum('tij,tj->ti',RC,delta,optimize=False)
  with np.load(F/'targets.npz')as z:robot=z['robot_qpos'].copy();native=z['object_poses'].copy()
  stage=np.asarray(read(S/'plane-aligned-articulated-native-validation-v2/frozen/command-proposal-report.json')['stage_rotation_from_inferred_support_normal']);native[:,0,:3]+=np.einsum('ij,tj->ti',stage,delta,optimize=False)
  rawpose=arrays('RAW-new-object-proposal-before-gates',dict(world_from_object=world,camera_from_object=camera,native_object_poses=native,raw_world_translation=delta,timestamps_s=times,frame_indices=frame_axis,unchanged_robot_qpos=robot));emit(dict(kind='AFTER_NEW_raw_pose_constructor_RETURN',actual_RETURN=rawpose,original_world_rotation_bytes_unchanged=world[:,:3,:3].tobytes()==original_world_rotation.tobytes(),original_camera_rotation_bytes_unchanged=camera[:,:3,:3].tobytes()==original_camera_rotation.tobytes()))
  manifest=read(F/'manifest.json');mesh=trimesh.load(F/manifest['objects'][0]['mesh'],force='mesh',process=False);V=np.asarray(mesh.vertices,float);faces=np.asarray(mesh.faces,int);base=read(T/'all151-NEW-training-source-statistics.json')
  def load_training(frame):
   with np.load(T/f'frame-{frame:03d}-NEW-training-partition.npz')as z:training_mask=z['training_mask'];domain=z['domain']
   with np.load(T/f'frame-{frame:03d}-NEW-training-point-RETURN.npz')as z:points=z['points_C_m']
   return dict(training_mask=training_mask,domain=domain,points_C_m=points)
  with np.load(D/'dynamic-source-data.npz')as z:oldcam=z['original_camera_from_object']
  oracle=TrainingSourceOracle(V,faces,K,C0,oldcam,RC,base,load_training,emit,arrays);training=[]
  for f in range(151):
   deadline();ret=oracle.query(f,delta[f],'ONE-raw-conic-point');save(f'frame-{f:03d}-actual-training-gate-RETURN.json',ret);training.append(ret);summary['actual_training_queries']+=1
  train_decision=source_filter_decision(training,list(range(151)));save('all151-actual-hard-training-RETURN.json',training);save('actual-hard-training-decision.json',train_decision);emit(dict(kind='AFTER_NEW_all151_training_decision_RETURN',actual_RETURN=train_decision));summary['training_failed_frames']=train_decision.get('failed_frames',[]);summary['actual_training_passed']=train_decision['accepted']
  deadline();emit(dict(kind='BEFORE_NEW_raw_wholeclock_object_motion_check'));motion=motion_summary(world,times);caps=meta['original_caps'];over=[k for k in caps if motion[k]>caps[k]+1e-9];motret=dict(actual=motion,original_caps=caps,original_comparison_tolerance=1e-9,over_caps=over);save('actual-wholeclock-object-motion-RETURN.json',motret);emit(dict(kind='AFTER_NEW_actual_object_motion_RETURN',actual_RETURN=motret));summary['wholeclock_motion_queries']=1;summary['motion_failed_caps']=over
  if not train_decision['accepted']or over:summary['construction_status']='REJECTED_actual_hard_training_or_original_motion_NO_fullpixel_rescue';summary['all_raw_points_REJECTED_as_reference']=True
  else:
   # Original fullpixel/heldout values enter ONLY after this completed construction.
   from scipy.spatial import cKDTree
   from select_pose_hypothesis import observed_fit
   from foundationpose_track import pose_rejection
   from camera_prior import transport
   tree=cKDTree(V);vbounds=np.stack([V.min(0),V.max(0)])
   with np.load(S/'plane-aligned-pose-temporal-path-v4/inferred-object-trajectory.npz')as z:cameras=z['candidate_camera_c2w'];flags=z['original_object_observed_flags']
   inv=np.linalg.inv(cameras);results=[]
   for f in range(151):
    deadline();authority=read(S/f'e037-object-source-authority-v1/frame-{f:03d}-source-authority.json');dp=Path(authority['depth_path']);mp=Path(authority['mask_path']);assert sha(dp)==authority['depth_sha256']and sha(mp)==authority['mask_sha256']
    with np.load(dp)as z:depth=z['depth_m'];valid=z['left_right_consistent']
    mask=cv2.imread(str(mp),0)>127;emit(dict(kind='BEFORE_NEW_original_fullpixel_POSTgate',frame=f,pose=camera[f],thresholds_NOT_optimizer_inputs=True))
    with warnings.catch_warnings(record=True)as ws:
     warnings.simplefilter('always');actual=observed_fit(V,faces,tree,camera[f],depth,valid,mask,K);prior=None if f==0 else transport(camera[f-1],inv[f-1],inv[f]);rejection,support=pose_rejection(camera[f],depth,valid,(mask*255).astype(np.uint8),K,vbounds,True,prior)
    rawret=dict(frame=f,actual_source_RETURN=actual,tracker_rejection_RETURN=rejection,tracker_support_RETURN=support,warnings=[str(w.message)for w in ws]);save(f'frame-{f:03d}-actual-original-source-RETURN.json',rawret);emit(dict(kind='AFTER_NEW_original_fullpixel_POSTgate_RETURN',actual_RETURN=rawret))
    cap=authority['current_accepted_source_thresholds_COPIED'];old=authority['old_strict_source_comparison_COPIED'];passed=bool(flags[f]and rejection is None and actual.get('quality',0)>=cap['quality']-1e-10 and actual.get('silhouette_iou',0)>=cap['silhouette_iou']-1e-10 and actual.get('median_measured_vertex_distance_m',float('inf'))<=cap['median_measured_vertex_distance_m']+1e-10 and actual.get('quality',0)>old['quality']and actual.get('silhouette_iou',0)>=old['silhouette_iou']and actual.get('median_measured_vertex_distance_m',float('inf'))<old['median_measured_vertex_distance_m']);results.append(dict(**rawret,original_source_passed=passed));summary['original_source_queries']+=1
   save('all151-actual-original-source-gates.json',results);failed=[r['frame']for r in results if not r['original_source_passed']];summary['original_source_failed_frames']=failed
   if failed:summary['construction_status']='REJECTED_original_fullpixel_source_POSTgate';summary['all_raw_points_REJECTED_as_reference']=True
   else:
    import shutil
    shutil.copyfile(rawpose['path'],O/'UNQUALIFIED-object-source-motion-component.npz');summary['new_source_motion_components']=1;summary['construction_status']='UNQUALIFIED_source_motion_component_all_native_self_swept_motor_grasp_whole_source_PENDING'
 summary['wall_seconds']=time.monotonic()-start;save('report.json',summary);emit(dict(kind='AFTER_ONE_NEW_evaluation_report_RETURN',actual_RETURN=summary));print(json.dumps(summary))
