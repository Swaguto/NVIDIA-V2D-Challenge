"""Consume one privately verified RGB prediction; preserve estimated authority and camera closure."""
from pathlib import Path
import copy,datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[2];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'scripts/track3'),str(S/'compute')]
import cv2,numpy as np
from source_camera_geometry import transform_mano_camera
from recover_source_hand import make_record
from numpy_mano_source_geometry_v2 import NumpyMano
from audit_source_hand_surface_depth_v1 import ray_depth
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cv2.setNumThreads(1)
O=S/'e037-native-RGB-hand-source-audit-v1';O.mkdir(exist_ok=True)
ip=S/'e037-native-RGB-hand-input-v1/input.json';inp=json.loads(ip.read_text());out=S/'e037-watcloud-native-RGB-v1/outputs';rp=out/'native-RGB/receipt.json';receipt=json.loads(rp.read_text());prediction_path=out/'native-RGB/prediction.npz';assert sha(prediction_path)==receipt['prediction_sha256'] and sha(ip)==receipt['input_sha256']
backup=json.loads((S/'compute/watcloud/native-RGB-private-backup-verified-v1.json').read_text());assert backup['worker_complete']
request_path=S/'e037-fresh-source-hand-request-v2/request.json';request=json.loads(request_path.read_text());old_row=next(x for x in request['frames'] if x['frame_idx']==50)
row=copy.deepcopy(old_row);row['crop']=inp['crop'];row['evidence_kind']='calibrated_RGB_model_prior';assert row['is_right']
assert row['fixed_betas']==inp['fixed_hand_betas'] and row['fixed_hand_scale']==inp['fixed_hand_scale']
with np.load(prediction_path) as z:prediction={k:z[k].copy() for k in z.files}
native=make_record(row,prediction,inp['camera'],sha(ip));native['source_camera']='ego_cam_a_undistorted';native['fixed_source_record_sha256']=inp['fixed_source_record_sha256']
m=NumpyMano(S/'models/mano/models/MANO_RIGHT.pkl');pose=np.r_[native['mano']['global_orient'],native['mano']['hand_pose']];J0=m.shaped(native['mano']['betas'])[1][0]
T=np.asarray(inp['Crect_from_A']);new_pose,new_t=transform_mano_camera(pose,native['cam_t'],J0,T,is_right=True)
canonical=copy.deepcopy(native);canonical['mano']['global_orient']=new_pose[:3].tolist();canonical['mano']['hand_pose']=new_pose[3:].tolist();canonical['cam_t']=new_t.tolist();canonical['native_inference_camera']=canonical.pop('camera');canonical['native_inference_crop']=canonical['crop'];canonical['crop']=old_row['crop'];canonical['image_size']=[request['camera']['width'],request['camera']['height']];canonical['camera']=dict(pred_cam_t_full=new_t.tolist(),scaled_focal_length=request['camera']['fx']);canonical['source_camera']='ego_cam_c_rectified';canonical['camera_frame_transform']=T.tolist();canonical['camera_conversion_code_sha256']=sha(R/'scripts/track3/source_camera_geometry.py');canonical['canonical_parent_request_sha256']=sha(request_path)
vA,jA,_=m.record(native);vC,jC,_=m.record(canonical);ev=np.einsum('ni,ji->nj',vA,T[:3,:3])+T[:3,3];ej=np.einsum('ni,ji->nj',jA,T[:3,:3])+T[:3,3];closure=max(float(abs(ev-vC).max()),float(abs(ej-jC).max()));assert closure<1e-6
assert canonical['mano']['betas']==old_row['fixed_betas'] and canonical['hand_scale']==old_row['fixed_hand_scale']
assert canonical['estimated'] and not any(canonical[k] for k in ['directly_observed','metric_valid','visual_valid'])
with np.load(request['canonical_bundle']) as z:K=z['intrinsics'].copy()
pc=np.c_[K[0,0]*jC[:,0]/jC[:,2]+K[0,2],K[1,1]*jC[:,1]/jC[:,2]+K[1,2]]
points=np.asarray(old_row['keypoints']);known=points[:,2]>0;hold=np.isin(np.arange(21),[1,4,7,10,13,16,19]);test=known&hold;errors=np.linalg.norm(pc-points[:,:2],axis=1)
with np.load(S/'e037-source-hand-motion-fit-v1/source-hand-candidate.npz') as z:depth=z['estimated_depth_m'][50].copy()
hd=test&np.isfinite(depth);dz=float(np.median(abs(jC[hd,2]-depth[hd])));median=float(np.median(errors[test]));p95=float(np.percentile(errors[test],95))
ca=inp['camera'];uvA=np.c_[ca['fx']*jA[:,0]/jA[:,2]+ca['cx'],ca['fy']*jA[:,1]/jA[:,2]+ca['cy']];rgb_delta=(uvA-np.asarray(inp['inferred_keypoints']))/np.array([ca['fx']/K[0,0],ca['fy']/K[1,1]]);er=np.linalg.norm(rgb_delta,axis=1)
parent=S/'e037-watcloud-dense-skin-source-fit-v2/source-hand-candidate.npz'
with np.load(parent) as z:px=z['heldout_pixels'].copy();source_depth=z['heldout_estimated_source_depth_m'].copy()
front=ray_depth(vC,m.faces,px,K);hit=np.isfinite(front);surface_error=abs(front-source_depth)
(O/'native-A-unqualified-record.json').write_text(json.dumps(native,indent=2)+'\n');(O/'canonical-C-unqualified-record.json').write_text(json.dumps(canonical,indent=2)+'\n')
np.savez_compressed(O/'unfitted-source-geometry.npz',vertices_camera_C=vC,joints_camera_C=jC,vertices_camera_A=vA,joints_camera_A=jA)
report=dict(schema='v2d.native-RGB-hand-prior-source-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),prediction_sha256=sha(prediction_path),GPU_receipt_sha256=sha(rp),input_sha256=sha(ip),camera_conversion_code_sha256=sha(R/'scripts/track3/source_camera_geometry.py'),camera_conversion_all_vertices_and_joints_max_error_m=closure,subject_betas_and_scale_preserved=True,original_flags_changed=False,original_canonical_handoff_modified=False,source_prior_fitted=False,estimated_RGB_all21_equivalent_stereo_px_median=float(np.median(er)),estimated_RGB_heldout_equivalent_stereo_px_median=float(np.median(er[hold])),estimated_RGB_heldout_equivalent_stereo_px_p95=float(np.percentile(er[hold],95)),legacy_IR_heldout_median_px=median,legacy_IR_heldout_p95_px=p95,legacy_estimated_joint_depth_median_error_m=dz,legacy_source_screen_passed=bool(median<=10 and p95<=25 and dz<=.01),heldout_surface_hits=int(hit.sum()),heldout_surface_rays=len(front),heldout_visible_surface_median_error_m=float(np.median(surface_error[hit])) if hit.any() else None,native_reference_contact_gate_m=.001,reference_accepted=False,training_approved=False,physical_task_accepted=False,new_PPO_steps=0)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
