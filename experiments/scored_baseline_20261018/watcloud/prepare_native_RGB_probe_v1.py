"""Prepare one calibrated color input and test camera-frame MANO conversion before GPU work."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[2];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'reconstruction/modules/v2d_stereo_depth/lib'),str(S/'compute')]
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from stereo_depth import load_rig
from numpy_mano_source_geometry_v2 import NumpyMano
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();O=S/'e037-native-RGB-hand-input-v1';O.mkdir();cv2.setNumThreads(1)
calpath=S/'dataset/track_3/public/meta/camera_calibration.json';cal=json.loads(calpath.read_text());rig=load_rig(str(calpath));a=cal['cameras']['ego_cam_a'];c=cal['cameras']['ego_cam_c'];K=np.array(a['intrinsic_matrix']);D=np.array(a['distortion_coefficients']);size=tuple(a['resolution'])
ap=S/'e037-RGB-source-camera-probe-v1/camera-a-frame-000050.png';image=cv2.imread(str(ap));assert image.shape[:2]==size[::-1]
source_image_report=json.loads((ap.parent/'report.json').read_text());assert sha(ap)==source_image_report['image_sha256']
mx,my=cv2.initUndistortRectifyMap(K,D,np.eye(3),K,size,cv2.CV_32FC1);undistorted=cv2.remap(image,mx,my,cv2.INTER_LINEAR);valid=(mx>=0)&(mx<size[0]-1)&(my>=0)&(my<size[1]-1)
target=O/'camera-a-pinhole-frame-000050.png';cv2.imwrite(str(target),undistorted);cv2.imwrite(str(O/'calibrated-source-valid.png'),valid.astype(np.uint8)*255)
mpath=S/'e037-RGB-hand-candidate-probe-v1/report.json';candidates=[x for x in json.loads(mpath.read_text())['candidates'] if x['crop']=='full'];association=json.loads((S/'e037-RGB-hand-correspondence-v1/report.json').read_text());raw=[x for x in association['predictions_compared'] if x['prior']=='raw_fresh_prior'];selected=min(raw,key=lambda x:x['palm_anchor_inferred_image_error_RGB_px']['median'])['RGB_candidate_index'];xy=np.asarray(candidates[selected]['landmarks_px'])
points=cv2.undistortPointsIter(xy[:,None],K,D,None,K,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(-1,2)
ray=np.c_[(points[:,0]-K[0,2])/K[0,0],(points[:,1]-K[1,2])/K[1,1],np.ones(len(points))];roundtrip=cv2.projectPoints(ray,np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2);max_roundtrip=float(abs(roundtrip-xy).max());assert max_roundtrip<1e-6
lo=points.min(0);hi=points.max(0);center=(lo+hi)/2;crop_size=float(max(hi-lo)*1.5);x0,y0=np.floor(center-crop_size/2).astype(int);x1,y1=np.ceil(center+crop_size/2).astype(int);assert 0<=x0<x1<size[0] and 0<=y0<y1<size[1];assert valid[y0:y1,x0:x1].all()
T_A_C=np.linalg.inv(np.array(a['extrinsics_to_stereo_left']))@np.array(c['extrinsics_to_stereo_left']);T_Crect_A=np.eye(4);T_Crect_A[:3,:3]=rig.R1@np.linalg.inv(T_A_C)[:3,:3];T_Crect_A[:3,3]=rig.R1@np.linalg.inv(T_A_C)[:3,3]
rp=S/'e037-watcloud-fresh-hand-source-audit-v1/unqualified-handoff/priors/right/000050.json';record=json.loads(rp.read_text());assert record['is_right'];m=NumpyMano(S/'models/mano/models/MANO_RIGHT.pkl');pose=np.r_[record['mano']['global_orient'],record['mano']['hand_pose']];beta=record['mano']['betas'];J0=m.shaped(beta)[1][0];v,j,_=m.record(record)
tests=[]
for label,Q,b in [('synthetic',Rotation.from_rotvec([.31,-.23,.19]).as_matrix(),np.array([.07,-.03,.08])),('fixed_factory',T_Crect_A[:3,:3],T_Crect_A[:3,3])]:
 q=pose.copy();q[:3]=Rotation.from_matrix(Q@Rotation.from_rotvec(pose[:3]).as_matrix()).as_rotvec();t=Q@(np.asarray(record['cam_t'])+J0)+b-J0;vv,jj,_=m.record(record,pose=q,translation=t);expected_v=v@Q.T+b;expected_j=j@Q.T+b;error=max(float(abs(vv-expected_v).max()),float(abs(jj-expected_j).max()));assert error<(1e-12 if label=='synthetic' else 1e-6)
 tests.append(dict(name=label,all_778_vertices_and_21_joints_max_error_m=error,betas_scale_and_fingers_unchanged=True,wrist_pivot_used=True))
payload=dict(schema='v2d.native-RGB-source-hand-probe-input.v1',frame=50,timestamp_s=2.5,source_camera='ego_cam_a',camera=dict(fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),width=size[0],height=size[1]),image=target.name,image_sha256=sha(target),source_raw_image_sha256=sha(ap),calibration_sha256=sha(calpath),crop=dict(cx=float(center[0]),cy=float(center[1]),size=crop_size),inferred_keypoints=points.tolist(),RGB_candidate_index=selected,identity_is_inferred_not_verified=True,fixed_hand_betas=beta,fixed_hand_scale=record['hand_scale'],fixed_source_record_sha256=sha(rp),Crect_from_A=T_Crect_A.tolist(),original_direct_metric_visual_flags_unchanged=True,reference_accepted=False,training_approved=False,physical_task_accepted=False)
(O/'input.json').write_text(json.dumps(payload,indent=2)+'\n')
receipt=dict(schema='v2d.native-RGB-source-hand-probe-preflight.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_sha256=sha(O/'input.json'),source_detector_report_sha256=sha(mpath),camera_calibration_and_distortion_unchanged=True,input_is_undistorted_with_own_RGB_pinhole_not_stereo_K=True,crop_source_valid_fraction=1.,source_detector_undistort_roundtrip_max_px=max_roundtrip,rigid_pose_conversion_controls=tests,native_reference_contact_gate_m=.001,existing_IR_failed_controls_preserved=True,new_PPO_steps=0,source_qualified_reference=False,physical_task_accepted=False)
(O/'preflight.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
