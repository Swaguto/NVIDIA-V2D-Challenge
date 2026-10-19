"""Fixed-factory RGB/stereo correspondence audit; no calibration fitting or depth inference."""
from pathlib import Path
import datetime, hashlib, json, sys, time
S=Path(__file__).resolve().parents[2]
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'reconstruction/modules/v2d_stereo_depth/lib'),str(R/'scripts/track3')]
import cv2, numpy as np
from stereo_depth import load_rig
from rgb_stereo import rectified_to_rgb_map
cv2.setNumThreads(1)
O=S/'e037-RGB-stereo-registration-v2';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
started=time.monotonic()
calpath=S/'dataset/track_3/public/meta/camera_calibration.json';cal=json.loads(calpath.read_text());rig=load_rig(str(calpath))
apath=S/'e037-RGB-source-camera-probe-v1/camera-a-frame-000050.png'
cpath=S/'e037-fresh-source-hand-request-v1/images/000050.png'
dpath=S/'fast-stereo-cpu-full-e037-v1/000050-depth.npz'
assert sha(apath)==json.loads((apath.parent/'report.json').read_text())['image_sha256']
depth_report=json.loads((dpath.parent/'depth-report.json').read_text())
with np.load(dpath) as z:
 depth=z['depth_m'].copy();valid=z['left_right_consistent'].copy()&np.isfinite(depth)&(depth>.05)&(depth<3)
 assert int(z['frame_index'])==50 and abs(float(z['timestamp_s'])-2.5)<1e-9
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz') as z:
 assert np.allclose(z['intrinsics'],rig.P1[:3,:3],atol=1e-5,rtol=0)
A=cv2.imread(str(apath));C=cv2.imread(str(cpath));assert C.shape[:2]==depth.shape
a=cal['cameras']['ego_cam_a'];c=cal['cameras']['ego_cam_c']
KA=np.array(a['intrinsic_matrix']);DA=np.array(a['distortion_coefficients']);KC=np.array(c['intrinsic_matrix']);DC=np.array(c['distortion_coefficients'])
T=np.linalg.inv(np.array(a['extrinsics_to_stereo_left']))@np.array(c['extrinsics_to_stereo_left'])
def project(x,K,D):return cv2.projectPoints(np.ascontiguousarray(x,float),np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
def bearing(uv,K,D):
 u=cv2.undistortPointsIter(np.ascontiguousarray(uv[:,None],float),K,D,None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(-1,2)
 return np.c_[u,np.ones(len(u))]
def quant(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return dict(n=int(len(x)),median=float(np.median(x)) if len(x) else None,p95=float(np.percentile(x,95)) if len(x) else None,maximum=float(np.max(x)) if len(x) else None)
sift=cv2.SIFT_create(nfeatures=5000)
kc,dc=sift.detectAndCompute(cv2.cvtColor(C,cv2.COLOR_BGR2GRAY),None)
ka,da=sift.detectAndCompute(cv2.cvtColor(A,cv2.COLOR_BGR2GRAY),None)
matcher=cv2.BFMatcher(cv2.NORM_L2)
forward={m.queryIdx:m for m,n in matcher.knnMatch(dc,da,k=2) if m.distance<.7*n.distance}
reverse={m.queryIdx:m for m,n in matcher.knnMatch(da,dc,k=2) if m.distance<.7*n.distance}
matches=[m for i,m in sorted(forward.items()) if m.trainIdx in reverse and reverse[m.trainIdx].trainIdx==i]
assert len(matches)>=8,'Insufficient independent appearance correspondences'
uc=np.array([kc[m.queryIdx].pt for m in matches]);ua=np.array([ka[m.trainIdx].pt for m in matches])
xc_rect=np.c_[(uc[:,0]-rig.P1[0,2])/rig.P1[0,0],(uc[:,1]-rig.P1[1,2])/rig.P1[1,1],np.ones(len(uc))]
xc_raw=xc_rect@rig.R1
raw_c=project(xc_raw,KC,DC)
cross=np.array([[0,-T[2,3],T[1,3]],[T[2,3],0,-T[0,3]],[-T[1,3],T[0,3],0]])
E=cross@T[:3,:3]
hypotheses=[];arrays={}
for flipC,flipA,label in [(False,False,'as_published'),(True,True,'extra_180_both_raw_pixel_grids'),(True,False,'extra_180_C_only'),(False,True,'extra_180_A_only')]:
 uc0=np.array(c['resolution'])-1-raw_c if flipC else raw_c
 ua0=np.array(a['resolution'])-1-ua if flipA else ua
 bc=bearing(uc0,KC,DC);ba=bearing(ua0,KA,DA)
 line=bc@E.T
 residual=(ba*line).sum(1)
 # Orthogonal distance in a virtual undistorted RGB pixel grid, not raw distorted pixels.
 normal=line[:,:2]/np.array([KA[0,0],KA[1,1]])
 ep=np.abs(residual)/np.linalg.norm(normal,axis=1)
 arrays[label+'_epipolar_px']=ep
 hypotheses.append(dict(name=label,epipolar_virtual_RGB_px=quant(ep),within_2px=int((ep<=2).sum()),within_5px=int((ep<=5).sum()),factory_rotation_or_intrinsics_fitted=False))
mapping=rectified_to_rgb_map(cal,rig,depth,valid)
iy=np.rint(uc[:,1]).astype(int);ix=np.rint(uc[:,0]).astype(int)
known=valid[iy,ix]&(mapping[iy,ix,0]>=0)
z=depth[iy,ix].astype(float)
x=xc_rect*z[:,None]
xA=(x@rig.R1)@T[:3,:3].T+T[:3,3]
pred=project(xA,KA,DA)
error=np.linalg.norm(pred-ua,axis=1)
ep=arrays['as_published_epipolar_px'];source_consistent=known&(ep<=2)
reprojection_vs_production=np.linalg.norm(pred-mapping[iy,ix],axis=1)
# Round-trip control uses fixed depth/geometry, no source matching or calibration optimization.
backA=bearing(pred,KA,DA)*xA[:,2,None]
backC=((backA-T[:3,3])@np.linalg.inv(T[:3,:3]).T)@rig.R1.T
round_trip=float(np.max(abs(backC-x)))
assert round_trip<1e-6
raw_round=float(np.max(abs(project(bearing(ua,KA,DA),KA,DA)-ua)))
print('DISTORTION_ROUND_TRIP_PX',raw_round,flush=True)
arrays.update(stereo_feature_pixels=uc,RGB_feature_pixels=ua,RGB_predicted_pixels=pred,depth_known=known,as_published_epipolar_consistent=source_consistent,RGB_depth_projection_error_px=error)
np.savez_compressed(O/'correspondences.npz',**arrays)
display=np.vstack([cv2.resize(C,(1014,634)),cv2.resize(A,(1014,760))])
for i in range(len(uc)):
 col=(0,200,0) if source_consistent[i] and error[i]<=5 else (0,120,255)
 p=tuple(np.rint(uc[i]*[1014/1280,634/800]).astype(int));q=tuple(np.rint(ua[i]*.5+[0,634]).astype(int))
 cv2.circle(display,p,3,col,1);cv2.circle(display,q,3,col,1)
cv2.imwrite(str(O/'appearance-correspondences.png'),display)
report=dict(schema='v2d.fixed-factory-RGB-stereo-registration.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_hashes={str(p.relative_to(S)):sha(p) for p in [apath,cpath,dpath,calpath]},frame=50,depth_source='existing learned stereo estimate with existing left-right consistency',new_depth_inference=False,source_features=dict(method='SIFT mutual two-way ratio',ratio=.7,max_features=5000,C_keypoints=len(kc),A_keypoints=len(ka),mutual_matches=len(matches)),hypotheses=hypotheses,as_published_known_depth_matches=int(known.sum()),as_published_epipolar_consistent_known_depth_matches=int(source_consistent.sum()),as_published_known_depth_error_RGB_px=quant(error[known]),as_published_epipolar_consistent_depth_error_RGB_px=quant(error[source_consistent]),all_match_depth_error_RGB_px=quant(error),RGB_inverse_numeric_closure_passed=raw_round<1e-6,calibration_registration_qualified=False,round_trip_3D_max_m=round_trip,RGB_distortion_round_trip_max_px=raw_round,production_map_subpixel_sampling_difference_px=quant(reprojection_vs_production[known]),production_map_float32_rounding_is_not_calibration_error=True,RGB_from_C=T.tolist(),rectification_R1=rig.R1.tolist(),source_calibration_unchanged=True,physical_exposure_synchrony_verified=False,physical_task_accepted=False,hidden_reference_labels_read=False,native_reference_contact_gate_m=.001,gate_changed=False,wall_seconds=time.monotonic()-started)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

assert raw_round<1e-6,'Retain arrays before inverse-distortion numeric closure failure'
