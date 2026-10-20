"""NEW provenance/authority inventory from saved receipts; no source/native replay."""
from pathlib import Path
import ast,datetime,hashlib,json,os,time
import numpy as np
from scipy.spatial.transform import Rotation
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-object-source-authority-v1';start=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb') as f:os.fsync(f.fileno())
 return p
T=S/'plane-aligned-pose-temporal-path-v4';A=S/'e037-joint-pose-temporal-compatibility-v1';B=S/'e037-joint-pose-translation-repair-v1';F=S/'e037-thumb-ray-independent-validation-v1/frozen'
paths={'track':T/'report.json','smooth':A/'report.json','repair':B/'report.json','depth':S/'fast-stereo-cpu-full-e037-v1/depth-report.json','frame50':S/'e037-joint-object-pose-v1/report.json'}
d={k:read(v) for k,v in paths.items()};old=read(S/'compute/private-integrity-joint-object-command-v1.json')
with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('w') as log:
 def emit(v):journal(log,v)
 emit(dict(kind='BEFORE_NEW_saved_object_authority_inventory',private_parent_receipt_sha256=sha(S/'compute/private-integrity-joint-object-command-v1.json'),no_source_native_fit_or_motion_producer_replay=True))
 verified=[]
 for rel,rec in old['files'].items():
  assert sha(S/rel)==rec['sha256'],rel;verified.append(rel)
 emit(dict(kind='AFTER_parent_integrity_return',verified_files=len(verified),numerical_audits_NOT_repeated=True))
 producer_bindings={}
 for key,file in [('track','select_plane_aligned_pose_temporal_path_v4.py'),('smooth','probe_joint_pose_temporal_compatibility_v1.py'),('repair','repair_joint_pose_translations_v1.py')]:
  p=S/'compute'/file;assert sha(p)==d[key]['producer_sha256'];producer_bindings[key]={'path':str(p),'sha256':sha(p)}
 cap=d['track']['motion']['original'];assert cap==d['smooth']['original_motion_caps']==d['repair']['original_motion_caps']
 emit(dict(kind='BEFORE_original_object_gate_source_binding',producers=producer_bindings))
 functions={}
 for rel,name in [('scripts/track3/select_pose_hypothesis.py','observed_fit'),('scripts/track3/foundationpose_track.py','pose_rejection'),('scripts/track3/reconstruct_stereo_object.py','fusion_depth'),('scripts/track3/temporal_pose_path.py','motion_summary')]:
  p=R/rel;tree=ast.parse(p.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);functions[name]={'path':str(p),'source_sha256':sha(p),'first_line':node.lineno,'exact_function_source':ast.get_source_segment(p.read_text(),node),'AST_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()}
 gates=dict(source_nonregression_against_current_accepted_component=dict(quality_minimum='saved per-frame quality',silhouette_IoU_minimum='saved per-frame silhouette_iou',mesh_vertex_median_maximum='saved per-frame median_measured_vertex_distance_m',inherited_comparison_tolerance=1e-10,original_v4_strict_before_thresholds_retained=True,not_optimized_on_heldout=True),original_object_motion_caps=cap,inherited_object_motion_comparison_tolerance=1e-9,robot_position_motion_image_root_tolerances_unchanged=[1e-8,1e-7,1e-6,1e-9],tracker=dict(center_camera_z_strictly_greater_m=.05,camera_vertices_strictly_greater_m=.01,interframe_prior_translation_norm_max_m=.15,interframe_prior_rotation_max_degrees=40,fusion_minimum_count=200,fusion_minimum_fraction=.5,fusion_body_margin='max(.02m,.1*max_body_extent)',fusion_original_eligibility='valid & mask>127 & finite & depth>.1m & depth<2.5m'),optimization_bounds_are_NOT_original_authority=dict(frame50_component_rotation_rad=.35,frame50_component_translation_m=.05,two_repair_component_translation_m=.005,old_cosine_time_profile_NOT_new_pose_authority=True),object_pose_component_box_NOT_found_in_original_gates=True,robot_8mm_08rad_boxes_NOT_object_pose_boxes=True,geometry_scale_and_physical_parameters_fixed=True)
 save('original-gate-source-bindings.json',dict(functions=functions,producer_bindings=producer_bindings,gates=gates));emit(dict(kind='AFTER_original_object_gate_source_binding',returned=gates))
 with np.load(T/'inferred-object-trajectory.npz') as z: cameras=z['candidate_camera_c2w'];flags=z['original_object_observed_flags'];plane=z['source_plane_qualified'];times=z['timestamps_s'];axis=z['frame_indices']
 with np.load(B/'unqualified-combined-object-proposal.npz') as z: worlds=z['world_from_object'];camera_objects=z['camera_from_object'];assert np.array_equal(times,z['timestamps_s']) and np.array_equal(axis,z['frame_indices'])
 with np.load(F/'targets.npz') as z: native=z['object_poses'][:,0];assert np.array_equal(times,z['timestamps_s']) and np.array_equal(axis,z['frame_indices'])
 stage=np.asarray(read(S/'plane-aligned-articulated-native-validation-v2/frozen/command-proposal-report.json')['stage_rotation_from_inferred_support_normal']);stage4=np.eye(4);stage4[:3,:3]=stage
 native_matrices=np.tile(np.eye(4),(len(axis),1,1));native_matrices[:,:3,3]=native[:,:3];native_matrices[:,:3,:3]=Rotation.from_quat(native[:,3:]).as_matrix()
 source_world=np.einsum('tij,tjk->tik',cameras,camera_objects,optimize=False);stage_world=np.einsum('ij,tjk->tik',stage4,worlds,optimize=False)
 closure=dict(source_camera_world_max_component_error=float(abs(source_world-worlds).max()),source_world_native_max_component_error=float(abs(stage_world-native_matrices).max()),stage_rotation_orthogonal_max_error=float(abs(np.einsum('ji,jk->ik',stage,stage,optimize=False)-np.eye(3)).max()))
 save('NEW-source-world-native-frame-return.json',dict(closure=closure,stage_rotation=stage,source_cameras_sha256=sha(T/'inferred-object-trajectory.npz'),object_source_proposal_sha256=sha(B/'unqualified-combined-object-proposal.npz'),F_targets_sha256=sha(F/'targets.npz'),timestamps_s=times,frame_indices=axis,no_new_pose_proposed=True));assert max(closure.values())<1e-12
 smooth={r['frame']:r for r in d['smooth']['rows']};repair={r['frame']:r for r in d['repair']['rows']};rows=[];input_records={};depth_rows={r.get('frame',i):r for i,r in enumerate(d['depth']['complete'])}
 for f in range(len(axis)):
  emit(dict(kind='BEFORE_NEW_frame_authority_inventory',frame=f,original_object_observed_flag=bool(flags[f]),source_plane_qualified_flag=bool(plane[f])))
  tr=d['track']['rows'][f];assert tr['frame']==f and tr['original_object_observed_flag']==bool(flags[f])
  current=tr['source_fit_after'];origin=paths['track'];path=['rows',f,'source_fit_after']
  if f in smooth:current=smooth[f]['source_fit_proposal'];origin=paths['smooth'];path=['rows',list(smooth).index(f),'source_fit_proposal']
  if f in repair:current=repair[f]['all_source_comparison_proposal'];origin=paths['repair'];path=['rows',list(repair).index(f),'all_source_comparison_proposal']
  dp=S/f'fast-stereo-cpu-full-e037-v1/{f:06d}-depth.npz';mp=S/f'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/{f:06d}.png'
  actual_dp=sha(dp);actual_mp=sha(mp);assert actual_dp==d['depth']['complete'][f]['sha256'];assert actual_mp==d['track']['source_files_sha256'][str(mp.relative_to(S))]
  assert flags[f] and tr['source_fit_qualified'] and current['quality']>=tr['source_fit_after']['quality']-1e-10 and current['silhouette_iou']>=tr['source_fit_after']['silhouette_iou']-1e-10 and current['median_measured_vertex_distance_m']<=tr['source_fit_after']['median_measured_vertex_distance_m']+1e-10
  source_split=dict(training_parity='even',heldout_parity='odd',split_kind='deterministic checkerboard; not independent episode',optimization_uses_training_only=True,original_fullpixel_gate_postconstruction_only=True,old_fullpixel_score_mixes_parities_NOT_a_training_objective=True)
  row=dict(frame=f,timestamp_s=float(times[f]),original_object_observed_flag=bool(flags[f]),source_plane_qualified_flag=bool(plane[f]),flags_are_inferred_source_provenance_NOT_measured_pose_truth=True,current_accepted_source_thresholds_COPIED=current,current_source_receipt=str(origin),current_source_receipt_sha256=sha(origin),current_source_JSON_path=path,old_strict_source_comparison_COPIED=tr['source_fit_before'],depth_path=str(dp),depth_sha256=actual_dp,mask_path=str(mp),mask_sha256=actual_mp,partition_authority=source_split,first_surface_ray_contract='fixed original eligible pixels; retain every hit and every missing infinite residual; no boundary/component removal',no_source_metric_recomputed=True)
  rows.append(row);save(f'frame-{f:03d}-source-authority.json',row);emit(dict(kind='AFTER_NEW_frame_authority_inventory',returned=row))
  input_records[str(dp)]=actual_dp;input_records[str(mp)]=actual_mp
 save('all151-source-authority-inventory.json',rows);save('cached-source-hashes.json',input_records)
 summary=dict(schema='v2d.NEW-object-source-motion-authority-inventory.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_private_files_verified=len(verified),frames=len(rows),frame_source_provenance_counts={'track':132,'smooth':17,'repair':2},original_object_motion_caps_COPIED=cap,source_world_native_frame_closure=closure,four_immutable_robot_native_failure_frames=[36,37,47,55],object_source_flags_at_four=[bool(flags[f]) for f in [36,37,47,55]],all_cached_depth_and_mask_hashes_verified=len(input_records),current_source_caps_not_recomputed=True,no_object_component_pose_box_in_original_authority=True,old_50mm_5mm_construction_bounds_NOT_original_gates=True,heldout_and_original_fullpixel_gates_NOT_new_optimizer_objectives=True,source_accuracy_and_heldout_skin_failure_UNCHANGED=True,need_NEW_training_only_joint_object_source_native_method_and_wholeclock_original_motion_checks=True,no_new_fit_source_native_or_physics_calls=True,new_candidates=0,new_GPU_jobs=0,new_PPO_steps=0,F_components_commands_masks_arrays_parameters_and_original_gates_UNCHANGED=True,wall_seconds=time.monotonic()-start)
 emit(dict(kind='AFTER_NEW_authority_inventory',returned=summary));summary['all_outputs_sha256']={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log','report.json']};save('report.json',summary);print(json.dumps(summary,default=plain))
