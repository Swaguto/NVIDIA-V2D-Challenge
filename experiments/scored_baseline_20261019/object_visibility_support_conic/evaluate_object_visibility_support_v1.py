"""SOLE bounded NEW object construction. Actual gates, never solver status, rule."""
from pathlib import Path
import datetime,hashlib,json,os,sys,time,warnings
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path[:0]=[str(S/'e037-direct-SDP-runtime-v2/site-packages'),str(R/'scripts/track3')]
import clarabel,cv2,numpy as np,trimesh
from scipy import sparse
from scipy.spatial import cKDTree
from select_pose_hypothesis import observed_fit
from foundationpose_track import pose_rejection
from camera_prior import transport
from temporal_pose_path import motion_summary
from object_visibility_support_v1 import CONFIG,returned
from thumb_conic_geometry_v1 import plain,journal
O=S/'e037-object-visibility-support-evaluation-v1';M=S/'e037-object-visibility-support-model-v3';F=S/'e037-thumb-ray-independent-validation-v1/frozen';start=time.monotonic();cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 binding=read(S/'e037-object-visibility-support-binding-v3/report.json');assert binding['all_columns_shapes_origin_bindings_pass']and binding['all66591collision_pair_incidence_complete'];meta=read(M/'canonical-shapes-and-config.json');assert meta['config']==CONFIG
 declaration=dict(schema='v2d.ONE-declared-training-visibility-support-object-evaluation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_report_sha256=sha(M/'report.json'),independent_binding_sha256=sha(S/'e037-object-visibility-support-binding-v3/report.json'),canonical_hashes={n:sha(M/n)for n in ['canonical-A.npz','canonical-P.npz','canonical-vectors.npz']},config=CONFIG,ONE_solve_NO_retry_warmstart_restart_tolerance_limit_profile_or_anchor_sweep=True,actual_fullpixel_source_and_all_six_original_motion_gates_POSTsolve_only=True,raw_primal_dual_slack_UNALTERED=True,no_native_position_physics_GPU_PPO_queued=True);save('ONE-declared-evaluation.json',declaration);emit(dict(kind='BEFORE_ONE_NEW_object_SOCP_evaluation',proposal=declaration))
 A=sparse.load_npz(M/'canonical-A.npz');P=sparse.load_npz(M/'canonical-P.npz')
 with np.load(M/'canonical-vectors.npz')as z:q=z['q'];b=z['b']
 settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100;settings.time_limit=60.;settings.max_threads=1;cones=[clarabel.NonnegativeConeT(meta['nonnegative_rows'])]+[clarabel.SecondOrderConeT(4)for _ in range(meta['SOC4'])];emit(dict(kind='BEFORE_ONE_NEW_episode_constructor',columns=453,rows=27772,nonnegative_rows=26576,SOC4=299,actual_settings=dict(max_iter=settings.max_iter,time_limit=settings.time_limit,max_threads=settings.max_threads,tol_gap_abs=settings.tol_gap_abs,tol_gap_rel=settings.tol_gap_rel,tol_feas=settings.tol_feas)))
 solver=clarabel.DefaultSolver(P,q,A,b,cones,settings);emit(dict(kind='AFTER_ONE_NEW_episode_constructor_RETURN',actual_type=type(solver).__name__,opaque_repr=repr(solver)));emit(dict(kind='BEFORE_ONE_NEW_episode_solver_call'));sol=solver.solve();ret=returned(sol);save('ONE-actual-SOCP-RETURN.json',ret);emit(dict(kind='AFTER_ONE_NEW_episode_solver_RETURN',actual_RETURN=ret,no_values_clipped_normalized_or_repaired=True))
 summary=dict(schema='v2d.ONE-NEW-object-visibility-support-evaluation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_solver_status=ret['status'],actual_iterations=ret['iterations'],actual_solve_seconds=ret['solve_time'],episode_solver_calls=1,raw_primal_dual_slack_unaltered=True,original_source_evaluations=0,wholeclock_motion_evaluations=0,source_failed_frames=[],new_object_source_motion_components=0,new_full_candidates=0,native_position_physics_GPU_PPO_calls=0,whole_trajectory_qualified=False,old_F_components_masks_source_parameters_gates_UNCHANGED=True,no_family_or_task_exclusion_claimed=True)
 x=np.asarray(ret['x'],float)
 if ret['status']not in ['Solved','AlmostSolved']or x.shape!=(453,)or not np.isfinite(x).all():summary['construction_status']='UNRESOLVED_raw_solver_status_NO_pose_proposal'
 elif float(abs(x).max())>CONFIG['world_translation_construction_halfwidth_m']:summary['construction_status']='REJECTED_raw_translation_outside_DECLARED_constructor_box_NO_pose_proposal';summary['maximum_raw_translation_component_m']=float(abs(x).max())
 else:
  delta=x.reshape(151,3);emit(dict(kind='BEFORE_NEW_actual_raw_object_pose_proposal',raw_translation=delta,not_a_reference_or_observation=True))
  with np.load(S/'e037-joint-pose-translation-repair-v1/unqualified-combined-object-proposal.npz')as z:world=z['world_from_object'].copy();camera=z['camera_from_object'].copy();times=z['timestamps_s'];frame_axis=z['frame_indices']
  with np.load(S/'plane-aligned-pose-temporal-path-v4/inferred-object-trajectory.npz')as z:cameras=z['candidate_camera_c2w'];source_flags=z['original_object_observed_flags']
  with np.load(F/'targets.npz')as z:original_native=z['object_poses'].copy();robot=z['robot_qpos'].copy()
  stage=np.asarray(read(S/'plane-aligned-articulated-native-validation-v2/frozen/command-proposal-report.json')['stage_rotation_from_inferred_support_normal']);inv=np.linalg.inv(cameras);world[:,:3,3]+=delta;camera[:,:3,3]+=np.einsum('tij,tj->ti',inv[:,:3,:3],delta,optimize=False);native=original_native.copy();native[:,0,:3]+=np.einsum('ij,tj->ti',stage,delta,optimize=False)
  proposal=O/'RAW-object-proposal-before-actual-gates.npz';np.savez_compressed(proposal,world_from_object=world,camera_from_object=camera,native_object_poses=native,raw_world_translation=delta,timestamps_s=times,frame_indices=frame_axis,unchanged_original_robot_qpos=robot,original_object_observed_flags=source_flags);emit(dict(kind='AFTER_NEW_actual_raw_object_pose_proposal_RETURN',proposal_sha256=sha(proposal),maximum_component_m=float(abs(delta).max()),rotations_and_robot_commands_byte_unchanged=True,not_promoted=True))
  emit(dict(kind='BEFORE_NEW_actual_wholeclock_original_object_motion_check',proposal_sha256=sha(proposal)));motion=motion_summary(world,times);authority=read(S/'e037-object-source-authority-v1/report.json');caps=authority['original_object_motion_caps_COPIED'];over=[k for k in caps if motion[k]>caps[k]+1e-9];save('actual-wholeclock-object-motion-RETURN.json',dict(actual=motion,ORIGINAL_caps=caps,original_comparison_tolerance=1e-9,over_caps=over));emit(dict(kind='AFTER_NEW_actual_wholeclock_original_object_motion_RETURN',actual_motion=motion,ORIGINAL_caps=caps,original_comparison_tolerance=1e-9,over_caps=over));summary['wholeclock_motion_evaluations']=1;summary['object_motion_failed_caps']=over
  with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz')as z:K=z['intrinsics']
  mf=read(F/'manifest.json');mesh=trimesh.load(F/mf['objects'][0]['mesh'],force='mesh',process=False);vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces);tree=cKDTree(vertices);bounds=np.stack([vertices.min(0),vertices.max(0)]);source=[];warning_records=[]
  for f in range(151):
   assert time.monotonic()-start<220,'ONEevaluation overall internal cap'
   ar=read(S/f'e037-object-source-authority-v1/frame-{f:03d}-source-authority.json');dp=Path(ar['depth_path']);mp=Path(ar['mask_path']);assert sha(dp)==ar['depth_sha256']and sha(mp)==ar['mask_sha256']
   with np.load(dp)as z:depth=z['depth_m'];valid=z['left_right_consistent']
   mask=cv2.imread(str(mp),0)>127;emit(dict(kind='BEFORE_NEW_actual_original_source_gate',frame=f,new_pose=camera[f],original_thresholds_COPIED=ar['current_accepted_source_thresholds_COPIED'],not_used_in_solver_or_selection=True))
   with warnings.catch_warnings(record=True)as ws:
    warnings.simplefilter('always');actual=observed_fit(vertices,faces,tree,camera[f],depth,valid,mask,K);prior=None if f==0 else transport(camera[f-1],np.linalg.inv(cameras[f-1]),np.linalg.inv(cameras[f]));rejection,support=pose_rejection(camera[f],depth,valid,(mask*255).astype(np.uint8),K,bounds,True,prior)
   output=dict(frame=f,actual_source_RETURN=actual,tracker_rejection_RETURN=rejection,tracker_support_RETURN=support,original_object_observed_flag=bool(source_flags[f]),warnings=[str(w.message)for w in ws],warning_cause_NOT_proven=True);save(f'frame-{f:03d}-actual-source-RETURN-before-acceptance.json',output);emit(dict(kind='AFTER_NEW_actual_original_source_gate_RETURN',actual_RETURN=output));threshold=ar['current_accepted_source_thresholds_COPIED'];before=ar['old_strict_source_comparison_COPIED'];passed=bool(source_flags[f]and rejection is None and actual.get('quality',0)>=threshold['quality']-1e-10 and actual.get('silhouette_iou',0)>=threshold['silhouette_iou']-1e-10 and actual.get('median_measured_vertex_distance_m',float('inf'))<=threshold['median_measured_vertex_distance_m']+1e-10 and actual.get('quality',0)>before['quality']and actual.get('silhouette_iou',0)>=before['silhouette_iou']and actual.get('median_measured_vertex_distance_m',float('inf'))<before['median_measured_vertex_distance_m']);source.append(dict(**output,actual_original_source_nonregression_passed=passed,original_comparison_tolerance=1e-10));warning_records.extend(output['warnings']);summary['original_source_evaluations']+=1;emit(dict(kind='AFTER_NEW_actual_original_source_decision',frame=f,passed=passed,no_acceptance_tolerance_changed=True))
  save('all151-actual-original-source-gates.json',source);failed=[r['frame']for r in source if not r['actual_original_source_nonregression_passed']];summary['source_failed_frames']=failed;summary['runtime_warning_count']=len(warning_records);summary['runtime_warnings_retained_cause_unknown']=warning_records
  if not failed and not over:
   import shutil
   shutil.copyfile(proposal,O/'UNQUALIFIED-object-source-motion-component.npz');summary['new_object_source_motion_components']=1;summary['construction_status']='PROPOSED_object_source_motion_component_ONLY_all_native_self_swept_motor_grasp_whole_source_PENDING'
  else:summary['construction_status']='REJECTED_actual_original_source_or_motion_failure';summary['all_raw_and_intermediate_object_poses_REJECTED_as_reference']=True
 summary['wall_seconds']=time.monotonic()-start;save('report.json',summary);emit(dict(kind='AFTER_ONE_NEW_object_construction_evaluation_RETURN',actual_RETURN=summary));print(json.dumps(summary))
