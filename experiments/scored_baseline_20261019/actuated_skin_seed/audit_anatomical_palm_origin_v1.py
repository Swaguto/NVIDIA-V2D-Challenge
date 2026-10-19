"""NEW object-independent palm-origin hypothesis from saved source MCP anchors.

No source fit, orientation fit, contact/force optimization, native pass or
command mutation. Two inferred human MCP anchors are not measured source truth.
The hypothesis does not authorize replacing the existing root bounds.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'))
import numpy as np
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
O=S/'e037-anatomical-palm-origin-audit-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();plan_path=S/'compute/anatomical-palm-origin-CPU-plan-v1.json';plan=json.loads(plan_path.read_text())
P=S/'e037-motion-constrained-wrist-native-v1/frozen';urdf=P/'robot/urdfs/g1/dex3_right.urdf';kin=URDFKinematics(urdf);manifest=json.loads((P/'manifest.json').read_text());names=manifest['joint_names']['right'];rest=kin.poses([0,0,0,0,0,0,1],dict.fromkeys(kin.names,0.));robot_anchors=np.stack([rest[f'right_hand_{d}_0_link'][:3,3] for d in ['index','middle']]);robot_center=robot_anchors.mean(0);robot_thumb=rest['right_hand_thumb_0_link'][:3,3]
with np.load(P/'targets.npz') as z:old=z['robot_qpos'][50,1].copy()
with np.load(P/'aligned-source-reference.npz') as z:camera=z['camera_c2w'][50].copy();flags={k:bool(z[k][50,1]) for k in ['hand_valid','hand_directly_observed','hand_metric_valid']}
assert not any(flags.values())
stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);origin=np.einsum('ij,j->i',stage,camera[:3,3],optimize=False)
orientation_path=S/'e037-source-wrist-convention-audit-v1/saved-palm-orientation-diagnostics.npz'
with np.load(orientation_path) as z:NC=z['native_from_C'].copy();source_R=z['source_robot_palm_rotations_native'].copy()
bundle=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(bundle) as z:oldcam=z['camera_c2w'][50].copy();K=z['intrinsics'].copy()
paths=[S/'e037-native-RGB-hand-source-audit-v1/unfitted-source-geometry.npz',S/'e037-native-RGB-multiview-source-fit-v1/source-hand-candidate.npz',S/'e037-native-RGB-hard-skin-coverage-fit-v2/source-hand-candidate.npz'];labels=['raw_RGB','multiview_fit_unqualified','hard_skin_fit_failed'];centers=[];rows=[];canonical_offsets=[]
# Only index/middle MCP anchors and the saved orientation enter this mapping.
# Object sites and image errors are read AFTER all three centers are frozen.
for i,(path,label) in enumerate(zip(paths,labels,strict=True)):
 with np.load(path) as z:C=z['joints_camera_C'].copy() if i==0 else np.einsum('ni,ij->nj',z['joints_world'][0]-oldcam[:3,3],oldcam[:3,:3],optimize=False)
 human=np.einsum('ni,ji->nj',C,NC,optimize=False)+origin;anchor_center=human[[5,9]].mean(0);new_origin=anchor_center-np.einsum('ij,j->i',source_R[i],robot_center,optimize=False);delta=new_origin-human[0];local=np.einsum('ji,j->i',source_R[i],delta,optimize=False);canonical_offsets.append(local);centers.append(np.r_[new_origin,Rotation.from_matrix(source_R[i]).as_quat()]);robot_world=np.einsum('ni,ji->nj',robot_anchors,source_R[i],optimize=False)+new_origin;source_wrist_robot_anchors=np.einsum('ni,ji->nj',robot_anchors,source_R[i],optimize=False)+human[0]
 rows.append(dict(source_label=label,source_flags=flags,source_wrist_native_world_m=human[0].tolist(),proposed_robot_palm_origin_native_world_m=new_origin.tolist(),derived_translation_from_source_wrist_m=delta.tolist(),derived_translation_norm_m=float(np.linalg.norm(delta)),derived_translation_robot_local_m=local.tolist(),orientation_unchanged=True,before_MCP_anchor_errors_m=np.linalg.norm(source_wrist_robot_anchors-human[[5,9]],axis=1).tolist(),after_MCP_anchor_errors_m=np.linalg.norm(robot_world-human[[5,9]],axis=1).tolist(),source_MCP_separation_m=float(np.linalg.norm(human[5]-human[9])),robot_MCP_separation_m=float(np.linalg.norm(robot_anchors[0]-robot_anchors[1])),unused_thumb_base_anchor_error_m=float(np.linalg.norm(new_origin+np.einsum('ij,j->i',source_R[i],robot_thumb,optimize=False)-human[1])),existing_component_translation_box_contains_derived_origin=bool(np.max(abs(delta))<=.008),inferred_anatomical_mapping_not_a_source_or_contact_certificate=True))
centers=np.asarray(centers);offset_spread=float(np.max(np.linalg.norm(np.asarray(canonical_offsets)-canonical_offsets[0],axis=1)))
# Read held-out/descriptive checks only after the mapping is frozen. They never
# choose or tune the center, orientation, anchor ids, scale or robot morphology.
image_path=S/'e037-watcloud-mask-refinement-handoff-v1/image-observations/right/000050.json';image=json.loads(image_path.read_text());kp=np.asarray(image['keypoints']);ids=np.array([4,8,12]);selected=kp[ids,2]>0;control_path=S/'e037-source-patch-wrench-sites-v1/report.json';controls=json.loads(control_path.read_text())
for i,row in enumerate(rows):
 original_root=np.r_[row['source_wrist_native_world_m'],centers[i,3:]];distances=[];pixels=[]
 for root in [original_root,centers[i]]:
  poses=kin.poses(root,dict(zip(names,old[7:],strict=True)));points=np.stack([poses[f'right_{d}_tip'][:3,3] for d in ['thumb','index','middle']]);sites=np.stack([controls['rows'][i]['proposed_site_seed']['sites'][d]['point'] for d in ['thumb','index','middle']]);distances.append(np.linalg.norm(points-sites,axis=1).tolist());C=np.einsum('ni,ij->nj',points-origin,NC,optimize=False);assert (C[:,2]>.01).all();uv=np.c_[K[0,0]*C[:,0]/C[:,2]+K[0,2],K[1,1]*C[:,1]/C[:,2]+K[1,2]];pixels.append(np.linalg.norm(uv[selected]-kp[ids[selected],:2],axis=1).tolist())
 row.update(fixed_finger_tip_to_saved_potential_site_distance_before_after_m=distances,inferred_tip_image_error_before_after_px=pixels,image_nonregression=bool(np.all(np.asarray(pixels[1])<=np.asarray(pixels[0])+1e-6)),no_fit_to_object_or_image_checks=True)
np.savez_compressed(O/'anatomical-root-hypotheses.npz',source_indices=np.arange(3),root_xyzw=centers,robot_local_translation_offsets=np.asarray(canonical_offsets))
report=dict(schema='v2d.object-independent-anatomical-palm-origin-hypothesis.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),inputs_sha256={str(p):sha(p) for p in [urdf,P/'targets.npz',P/'aligned-source-reference.npz',orientation_path,bundle,image_path,control_path]+paths},robot_MCP_anchor_local_m=robot_anchors.tolist(),human_anchor_ids=[5,9],canonical_translation_offset_spread_across_retained_source_models_m=offset_spread,rows=rows,no_object_or_image_fit_or_center_selection=True,no_source_orientation_fit_or_180_degree_flip=True,source_pose_winner_not_selected=True,source_and_heldout_failures_unchanged=True,existing_boxes_not_enlarged_or_overridden=True,default_mapping_or_commands_not_changed=True,new_source_fit_native_or_old_pose_solver_not_run=True,hypothesis_requires_independent_anatomical_mapping_source_and_motion_validation=True,required_fresh_full_native_reference_contact_gate_m=.001,new_GPU_work=False,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
