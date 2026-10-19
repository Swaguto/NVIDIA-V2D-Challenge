"""Trace the saved surface-plus-offset heuristic without relabeling or dropping failed observations."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[2];sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(S/'compute')]
import cv2,numpy as np
from numpy_mano_source_geometry_v2 import NumpyMano
from audit_source_hand_surface_depth_v1 import ray_depth
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cv2.setNumThreads(1);O=S/'e037-source-depth-authority-audit-v1';O.mkdir(exist_ok=True)
request_path=S/'e037-fresh-source-hand-request-v2/request.json';request=json.loads(request_path.read_text());row=next(x for x in request['frames'] if x['frame_idx']==50);points=np.asarray(row['keypoints']);ids=np.flatnonzero(points[:,2]>0)
source_report_path=S/'e037-source-hand-motion-fit-v1/report.json';historical=json.loads(source_report_path.read_text());producer=S/'compute/fit_source_supported_hand_motion_v1.py';assert sha(producer)==historical['producer_sha256'];assert historical['skin_to_joint_depth_prior_m']==.005
oldfit=S/'e037-source-hand-motion-fit-v1/source-hand-candidate.npz'
with np.load(oldfit) as z:old_depth=z['estimated_depth_m'][50].copy();assert np.array_equal(z['landmarks_px'][50,ids],points[ids,:2])
dp=S/'fast-stereo-cpu-full-e037-v1/000050-depth.npz'
with np.load(dp) as z:depth=z['depth_m'].copy();valid=z['left_right_consistent'].copy()&np.isfinite(depth)&(depth>=.15)&(depth<=1.4)
op=S/'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/000050.png';obj=cv2.imread(str(op),0);assert obj is not None;valid&=~(cv2.dilate((obj>0).astype(np.uint8),np.ones((9,9),np.uint8))>0)
with np.load(request['canonical_bundle']) as z:K=z['intrinsics'].copy();source_flags={k:z[k][50,1].item() for k in ['hand_valid','hand_directly_observed','hand_metric_valid']}
record_path=S/'e037-native-RGB-hand-source-audit-v1/canonical-C-unqualified-record.json';record=json.loads(record_path.read_text());fit=S/'e037-native-RGB-landmark-adapter-fit-v1/source-hand-candidate.npz'
with np.load(fit) as z:pose=z['mano_pose'][0].copy();t=z['camera_translation'][0].copy()
m=NumpyMano(S/'models/mano/models/MANO_RIGHT.pkl');vertices,joints,_=m.record(record,pose=pose,translation=t)
model_uv=np.c_[K[0,0]*joints[:,0]/joints[:,2]+K[0,2],K[1,1]*joints[:,1]/joints[:,2]+K[1,2]]
model_front=ray_depth(vertices,m.faces,model_uv,K);source_front=ray_depth(vertices,m.faces,points[ids,:2],K);rows=[]
for n,k in enumerate(ids):
 x,y=np.rint(points[k,:2]).astype(int);patch=depth[y-4:y+5,x-4:x+5][valid[y-4:y+5,x-4:x+5]];median=float(np.median(patch)) if len(patch) else None;mad=float(np.median(abs(patch-median))) if len(patch) else None;passing=len(patch)>=5 and mad<=.005
 if passing:assert np.isfinite(old_depth[k]) and abs(median+.005-old_depth[k])<1e-7
 else:assert not np.isfinite(old_depth[k])
 model_gap=float(joints[k,2]-model_front[k]) if np.isfinite(model_front[k]) else None
 rows.append(dict(joint_id=int(k),source_pixel=points[k,:2].tolist(),source_kind=row['evidence_kind'],one_sided_identity_or_flow=True,original_image_confidence=float(points[k,2]),patch_valid_pixels=len(patch),patch_surface_median_m=median,patch_surface_MAD_m=mad,patch_heuristic_accepted=bool(passing),historical_plus5mm_target_m=float(old_depth[k]) if passing else None,historical_depth_target_type='visible_surface_estimate_plus_unverified_offset',historical_depth_is_direct_joint_observation=False,estimated_model_joint_depth_m=float(joints[k,2]),estimated_model_front_at_exact_joint_pixel_m=float(model_front[k]) if np.isfinite(model_front[k]) else None,estimated_model_joint_behind_front_m=model_gap,tip_is_native_mesh_surface_vertex=bool(k in [4,8,12,16,20]),model_visibility_is_not_video_visibility=True,source_image_projection_error_px=float(np.linalg.norm(model_uv[k]-points[k,:2])),estimated_model_front_at_source_pixel_m=float(source_front[n]) if np.isfinite(source_front[n]) else None))
# Exact synthetic occlusion: an image-aligned surface vertex can be hidden by a nearer hand surface.
test_K=np.array([[100.,0,20.],[0,100.,20.],[0,0,1.]])
v=np.array([[-.02,-.02,.3],[.02,-.02,.3],[0,.02,.3],[0,0,.35],[-.01,-.01,.35],[.01,-.01,.35]])
faces=np.array([[0,1,2],[3,4,5]]);pixel=np.array([[20.,20.]]);front=float(ray_depth(v,faces,pixel,test_K)[0]);assert abs(front-.3)<1e-12
legacy_target=front+.005;true_occluded_vertex_z=.35;false_error=abs(legacy_target-true_occluded_vertex_z);assert false_error>.01
report=dict(schema='v2d.source-depth-authority-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_hashes={str(p.relative_to(S)):sha(p) for p in [request_path,source_report_path,producer,oldfit,dp,op,record_path,fit]},original_frame50_source_flags=source_flags,all_available_historical_targets_reproduced_without_repair=True,historical_skin_to_joint_offset_m=.005,observations_or_heldout_failure_labels_changed=False,depth_or_image_weights_changed=False,original_failed_gates_overridden=False,model_visibility_does_not_upgrade_source_observations=True,synthetic_occlusion_control=dict(visible_front_surface_depth_m=front,exact_occluded_vertex_depth_m=true_occluded_vertex_z,original_heuristic_target_depth_m=legacy_target,joint_depth_error_m=false_error,image_error_px=0.,front_surface_error_m=0.,unchanged_legacy_depth_gate_m=.01,legacy_joint_screen_falsely_rejects_exact_occluded_geometry=True),rows=rows,required_native_reference_contact_gate_m=.001,no_new_CPU_fit_or_GPU_inference=True,new_PPO_steps=0,physical_task_accepted=False)
(O/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2,allow_nan=False))
