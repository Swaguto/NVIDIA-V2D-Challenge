"""Apply the explicit inferred hold to the actual unknown right-hand interval."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path[:0]=[str(S/'cpu-native-runtime-v1/src'),str(R/'scripts/track3'),str(R/'scripts/dex3_training')]
import numpy as np
from scipy.spatial.transform import Rotation
from object_relative_commands import infer_object_relative_gap
from temporal_pose_path import motion_summary
from v2d_dex3_flashchord.kinematics import URDFKinematics
O=S/'e037-object-relative-command-proposal-v2';O.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=S/'e037-joint-object-native-validation-v1/frozen';m=json.loads((P/'manifest.json').read_text())
for name,h in m['files'].items():assert sha(P/name)==h
with np.load(P/'targets.npz') as z:targets={k:z[k].copy() for k in z.files}
with np.load(P/'aligned-source-reference.npz') as z:source={k:z[k].copy() for k in z.files}
obj=targets['object_poses'][:,0];transforms=np.tile(np.eye(4),(len(obj),1,1));transforms[:,:3,:3]=Rotation.from_quat(obj[:,3:]).as_matrix();transforms[:,:3,3]=obj[:,:3]
result=infer_object_relative_gap(targets['robot_qpos'][:,1],transforms,targets['timestamps_s'],36,70,source_valid=source['hand_valid'][:,1],contact_assumption='inferred_object_relative_hold')
commands=targets['robot_qpos'].copy();commands[:,1]=result.commands
assert np.array_equal(commands[:,0],targets['robot_qpos'][:,0]);assert np.array_equal(commands[:,:,7:],targets['robot_qpos'][:,:,7:]);assert np.array_equal(commands[~result.inferred_command_mask],targets['robot_qpos'][~result.inferred_command_mask]);assert np.array_equal(result.original_source_valid,source['hand_valid'][:,1])
def root_transform(q):
 t=np.tile(np.eye(4),(len(q),1,1));t[:,:3,:3]=Rotation.from_quat(q[:,3:7]).as_matrix();t[:,:3,3]=q[:,:3];return t
motion={}
for label,q in [('original_source',source['robot_qpos'][:,1]),('previous_repaired',targets['robot_qpos'][:,1]),('object_relative_hypothesis',commands[:,1])]:motion[label]=motion_summary(root_transform(q),targets['timestamps_s'])
motion_over=[k for k,v in motion['original_source'].items() if motion['object_relative_hypothesis'][k]>v+1e-9]
stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);kin=URDFKinematics(P/'robot/urdfs/g1/dex3_right.urdf')
def project(points,frame):
 world_y=np.einsum('ni,ij->nj',points,stage,optimize=False);cam=source['camera_c2w'][frame];c=np.einsum('ni,ij->nj',world_y-cam[:3,3],cam[:3,:3],optimize=False);assert (c[:,2]>.01).all();return np.c_[K[0,0]*c[:,0]/c[:,2]+K[0,2],K[1,1]*c[:,1]/c[:,2]+K[1,2]]
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:K=z['intrinsics'].copy()
image_rows=[];hashes={}
for frame in [40,45,50,55,67]:
 f=S/f'e037-watcloud-mask-refinement-handoff-v1/image-observations/right/{frame:06d}.json';d=json.loads(f.read_text());points=np.asarray(d['keypoints']);ids=np.array([4,8,12]);selected=points[ids,2]>0;ids=ids[selected];xy=points[ids,:2];assert len(ids)>0 and not d['metric_valid'] and not d['directly_observed'];errors=[]
 for q in [targets['robot_qpos'][frame,1],commands[frame,1]]:
  poses=kin.poses(q[:7],dict(zip(m['joint_names']['right'],q[7:],strict=True)));tips=np.stack([poses['right_'+digit+'_tip'][:3,3] for digit in ['thumb','index','middle']])[selected];errors.append(np.linalg.norm(project(tips,frame)-xy,axis=1))
 image_rows.append(dict(frame=frame,inferred_landmark_ids=ids.tolist(),source_metric_valid=False,source_directly_observed=False,previous_tip_errors_px=errors[0].tolist(),proposal_tip_errors_px=errors[1].tolist(),previous_median_px=float(np.median(errors[0])),proposal_median_px=float(np.median(errors[1])),all_inferred_points_nonregressing=bool((errors[1]<=errors[0]+1e-6).all())))
 hashes[str(f.relative_to(S))]=sha(f)
np.savez_compressed(O/'inferred-robot-commands.npz',robot_qpos=commands,inferred_command_mask=result.inferred_command_mask,original_source_valid=result.original_source_valid,frame_indices=targets['frame_indices'],timestamps_s=targets['timestamps_s'])
report=dict(schema='v2d.object-relative-inferred-command-proposal.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),helper_sha256=sha(R/'scripts/track3/object_relative_commands.py'),input_manifest_sha256=sha(P/'manifest.json'),source_bundle_sha256=sha(B),inferred_source_image_hashes=hashes,gap_half_open=[36,70],anchor_frames=[35,70],assumption='inferred_object_relative_hold; source anchors valid but not directly or metrically observed',original_direct_and_metric_flags_retained=True,source_valid_flags_unchanged=True,only_unknown_command_roots_changed=True,fingers_left_hand_and_all_outside_gap_commands_exact=True,maximum_root_reconstruction_difference_m=float(np.linalg.norm(commands[:,1,:3]-targets['robot_qpos'][:,1,:3],axis=1).max()),motion=motion,over_original_root_motion_caps=motion_over,root_motion_stage_passed=not motion_over,inferred_image_checks=image_rows,all_inferred_image_points_nonregressing=all(r['all_inferred_points_nonregressing'] for r in image_rows),old_14mm_contact_correction_bound_not_used_to_hide_a_new_source_reconstruction=True,existing_source_human_mesh_or_observation_flags_changed=False,contact_labels_fabricated=False,source_human_metric_recovery_certified=False,previous_native_certificates_invalid_for_new_robot_commands=True,required_native_reference_contact_gate_m=.001,full_source_reference_qualified=False,physical_task_accepted=False,new_GPU_work=False,new_PPO_steps=0)
for name,h in m['files'].items():assert sha(P/name)==h
for rel,h in hashes.items():assert sha(S/rel)==h
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
