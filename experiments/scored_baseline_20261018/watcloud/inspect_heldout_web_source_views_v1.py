"""Inspect the unchanged failed pixel in both source cameras; no new class labels or fits."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[2];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'reconstruction/modules/v2d_stereo_depth/lib')]
import cv2,numpy as np
from stereo_depth import load_rig
cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();O=S/'e037-heldout-web-source-views-v1';O.mkdir()
cp=S/'dataset/track_3/public/meta/camera_calibration.json';cal=json.loads(cp.read_text());rig=load_rig(str(cp));a=cal['cameras']['ego_cam_a'];c=cal['cameras']['ego_cam_c'];T=np.linalg.inv(np.asarray(a['extrinsics_to_stereo_left']))@np.asarray(c['extrinsics_to_stereo_left']);K=np.asarray(a['intrinsic_matrix']);D=np.asarray(a['distortion_coefficients'])
fit=S/'e037-native-RGB-hard-skin-coverage-fit-v2/source-hand-candidate.npz'
with np.load(fit) as z:px=z['heldout_pixels'].copy();front=z['heldout_front_depth_m'].copy();depth=z['heldout_estimated_source_depth_m'].copy()
missing=np.flatnonzero(~np.isfinite(front));assert len(missing)==1;i=int(missing[0]);u,v=px[i];z=float(depth[i]);xc=np.array([(u-rig.P1[0,2])*z/rig.P1[0,0],(v-rig.P1[1,2])*z/rig.P1[1,1],z]);xa=(xc@rig.R1)@T[:3,:3].T+T[:3,3];uvA=cv2.projectPoints(xa[None],np.zeros(3),np.zeros(3),K,D)[0].reshape(2)
paths=[S/'e037-fresh-source-hand-request-v1/images/000050.png',S/'e037-RGB-source-camera-probe-v1/camera-a-frame-000050.png'];rows=[];crops=[]
for name,path,point,radius in zip(['C_rectified','A_raw_RGB'],paths,[np.array([u,v]),uvA],[35,65]):
 image=cv2.imread(str(path));x,y=np.rint(point).astype(int);h,w=image.shape[:2];lo=np.array([max(0,x-radius),max(0,y-radius)]);hi=np.array([min(w,x+radius+1),min(h,y+radius+1)]);crop=image[lo[1]:hi[1],lo[0]:hi[0]].copy();center=(point-lo)*[500/crop.shape[1],500/crop.shape[0]];crop=cv2.resize(crop,(500,500));cv2.circle(crop,tuple(np.rint(center).astype(int)),13,(0,0,255),2);cv2.line(crop,(int(center[0])-5,int(center[1])),(int(center[0])+5,int(center[1])),(0,0,255),1);cv2.putText(crop,name,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2);crops.append(crop);rows.append(dict(camera=name,source_image_sha256=sha(path),projected_pixel=point.tolist(),source_crop_bounds=[lo.tolist(),hi.tolist()]))
cv2.imwrite(str(O/'unchanged-failed-pixel-source-crops.png'),np.hstack(crops))
zp=S/'e037-RGB-hand-correspondence-v1/RGB-visible-stereo-surface.npz';x,y=np.rint(uvA).astype(int)
with np.load(zp) as data:known=bool(data['valid'][y,x]);registered_front=float(data['depth_m'][y,x]) if known else None
report=dict(schema='v2d.unchanged-heldout-web-source-view-inspection.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_candidate_sha256=sha(fit),calibration_sha256=sha(cp),heldout_index=i,source_C_pixel=[int(u),int(v)],estimated_C_surface_depth_m=z,estimated_A_projected_depth_m=float(xa[2]),registered_C_point_cloud_A_front_depth_m=registered_front,registered_front_is_estimated_and_not_A_depth_measurement=True,possible_A_self_occlusion_from_registered_C_cloud_m=float(xa[2]-registered_front) if known else None,source_views=rows,source_exposure_synchrony_not_verified=True,per_pixel_registration_uncertainty_not_qualified=True,skin_or_object_class_not_verified=True,no_heldout_pixel_or_failure_label_changed=True,no_new_source_inference_or_fit=True,physical_task_accepted=False)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
