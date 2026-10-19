"""Test one declared smooth body-frame bias proposal; no fitting or source relabeling."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'scripts/track3')]
import cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from select_pose_hypothesis import observed_fit
from temporal_pose_path import motion_summary
from foundationpose_track import pose_rejection
from camera_prior import transport
cv2.setNumThreads(1);O=S/'e037-joint-pose-temporal-compatibility-v1';O.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
track_path=S/'plane-aligned-pose-temporal-path-v4/inferred-object-trajectory.npz';track_report_path=S/'plane-aligned-pose-temporal-path-v4/report.json';track_report=json.loads(track_report_path.read_text())
with np.load(track_path) as z:
 before_world=z['candidate_world_from_object'].copy();cameras=z['candidate_camera_c2w'].copy();before_camera=z['candidate_camera_from_object'].copy();times=z['timestamps_s'].copy();original_object=z['original_camera_from_object'].copy();original_flags=z['original_object_observed_flags'].copy()
candidate_path=S/'e037-joint-object-pose-v1/unqualified-joint-object-pose.npz'
with np.load(candidate_path) as z:center_pose=z['camera_from_object'].copy()
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:K=z['intrinsics'].copy();original_camera=z['camera_c2w'].copy()
P=S/'plane-aligned-native-reference-v1/frozen';manifest=json.loads((P/'manifest.json').read_text());mesh_path=P/manifest['objects'][0]['mesh'];assert sha(mesh_path)==manifest['files'][manifest['objects'][0]['mesh']]
mesh=trimesh.load(mesh_path,force='mesh',process=False);v=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces);tree=cKDTree(v)
center_world=np.einsum('ij,jk->ik',cameras[50],center_pose,optimize=False);delta=np.einsum('ij,jk->ik',np.linalg.inv(before_world[50]),center_world,optimize=False);rotvec=Rotation.from_matrix(delta[:3,:3]).as_rotvec()
# One physically bounded proposal, declared before inspecting the adjacent frames.
# Its endpoints and source confidence remain untouched. It is not an observation.
center_t=times[50];half_window_s=.5;weights=np.zeros(len(times));within=abs(times-center_t)<half_window_s;weights[within]=.5*(1+np.cos(np.pi*(times[within]-center_t)/half_window_s))
world=before_world.copy()
for frame,a in enumerate(weights):
 if a<=1e-12:continue
 correction=np.eye(4);correction[:3,:3]=Rotation.from_rotvec(rotvec*a).as_matrix();correction[:3,3]=delta[:3,3]*a;world[frame]=np.einsum('ij,jk->ik',before_world[frame],correction,optimize=False)
camera=np.einsum('tij,tjk->tik',np.linalg.inv(cameras),world,optimize=False);assert np.max(abs(camera[50]-center_pose))<1e-12
original_motion=motion_summary(np.einsum('tij,tjk->tik',original_camera,original_object,optimize=False),times);before_motion=motion_summary(before_world,times);after_motion=motion_summary(world,times)
over=[k for k in original_motion if after_motion[k]>original_motion[k]+1e-9]
depth_report_path=S/'fast-stereo-cpu-full-e037-v1/depth-report.json';depth_report=json.loads(depth_report_path.read_text());rows=[];hashes={}
for frame in np.flatnonzero(weights>1e-12):
 frame=int(frame);dp=S/f'fast-stereo-cpu-full-e037-v1/{frame:06d}-depth.npz';assert sha(dp)==depth_report['complete'][frame]['sha256']
 with np.load(dp) as z:depth=z['depth_m'].copy();valid=z['left_right_consistent'].copy()
 mp=S/f'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/{frame:06d}.png';assert sha(mp)==track_report['source_files_sha256'][str(mp.relative_to(S))]
 mask=cv2.imread(str(mp),0)>0;after=observed_fit(v,faces,tree,camera[frame],depth,valid,mask,K);before=track_report['rows'][frame]['source_fit_after']
 prior=transport(camera[frame-1],np.linalg.inv(cameras[frame-1]),np.linalg.inv(cameras[frame]))
 reject,diagnostics=pose_rejection(camera[frame],depth,valid,(mask*255).astype(np.uint8),K,np.stack([v.min(0),v.max(0)]),True,prior)
 nonregression=bool(original_flags[frame] and reject is None and after['quality']>=before['quality']-1e-10 and after['silhouette_iou']>=before['silhouette_iou']-1e-10 and after['median_measured_vertex_distance_m']<=before['median_measured_vertex_distance_m']+1e-10)
 rows.append(dict(frame=frame,proposal_weight=float(weights[frame]),source_fit_before=before,source_fit_proposal=after,tracker_rejection=reject,source_nonregression_against_frozen_selected=nonregression,original_observed_flag=bool(original_flags[frame])))
 hashes[str(dp.relative_to(S))]=sha(dp);hashes[str(mp.relative_to(S))]=sha(mp)
 (O/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
failed=[r['frame'] for r in rows if not r['source_nonregression_against_frozen_selected']]
np.savez_compressed(O/'unqualified-smooth-object-proposal.npz',frame_indices=np.arange(len(times)),timestamps_s=times,camera_from_object=camera,world_from_object=world,proposal_weights=weights,original_object_observed_flags=original_flags)
report=dict(schema='v2d.joint-object-pose-temporal-compatibility.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_hashes={str(p.relative_to(S)):sha(p) for p in [track_path,track_report_path,candidate_path,B,P/'manifest.json',mesh_path,depth_report_path]},changed_frame_input_hashes=hashes,center_frame=50,declared_half_window_seconds=half_window_s,profile='Cosine-weighted object-body rotation-vector/translation bias; no fit or observed data fill',changed_frames=[r['frame'] for r in rows],center_pose_closure_m=float(abs(camera[50]-center_pose).max()),original_motion_caps=original_motion,frozen_selected_motion=before_motion,proposed_motion=after_motion,over_original_motion_caps=over,all_original_motion_caps_passed=not over,rows=rows,source_nonregression_failed_frames=failed,all_source_nonregression_passed=not failed,mesh_scale_or_geometry_changed=False,source_flags_or_confidence_changed=False,original_tracks_modified=False,all_earlier_failed_controls_retained=True,new_full_reference_promoted=False,full_source_reference_qualified=False,physical_task_accepted=False,required_native_reference_contact_gate_m=.001,native_contact_or_grasp_certified=False,new_GPU_work=False,new_PPO_steps=0)
for rel,h in report['source_hashes'].items():assert sha(S/rel)==h,rel
for rel,h in hashes.items():assert sha(S/rel)==h,rel
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
