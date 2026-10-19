"""Compare saved RGB hand predictions and forward-projected priors; no new detections or fit."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[2];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'reconstruction/modules/v2d_stereo_depth/lib'),str(S/'compute')]
import cv2,numpy as np
from stereo_depth import load_rig
from numpy_mano_source_geometry_v2 import NumpyMano
cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.monotonic()
O=S/'e037-RGB-hand-correspondence-v1';O.mkdir(exist_ok=True)
cp=S/'dataset/track_3/public/meta/camera_calibration.json';cal=json.loads(cp.read_text());rig=load_rig(str(cp))
a=cal['cameras']['ego_cam_a'];c=cal['cameras']['ego_cam_c'];K=np.array(a['intrinsic_matrix']);D=np.array(a['distortion_coefficients'])
T=np.linalg.inv(np.array(a['extrinsics_to_stereo_left']))@np.array(c['extrinsics_to_stereo_left'])
ap=S/'e037-RGB-source-camera-probe-v1/camera-a-frame-000050.png';mpath=S/'e037-RGB-hand-candidate-probe-v1/report.json'
mp=json.loads(mpath.read_text());candidates=[p for p in mp['candidates'] if p['crop']=='full'];assert len(candidates)==2
rp=S/'e037-watcloud-fresh-hand-source-audit-v1/unqualified-handoff/priors/right/000050.json';record=json.loads(rp.read_text());m=NumpyMano(S/'models/mano/models/MANO_RIGHT.pkl')
def project(points):
 xyz=(points@rig.R1)@T[:3,:3].T+T[:3,3]
 return cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
def stats(x):return dict(median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(np.max(x)))
rows=[];joint_sets={};sources={str(p.relative_to(S)):sha(p) for p in [cp,ap,mpath,rp]}
for name,fit in [('raw_fresh_prior',None),('bounded_old_IR_image_fit',S/'e037-watcloud-fresh-hand-source-fit-v1/source-hand-candidate.npz'),('dense_skin_count_normalized_fit',S/'e037-watcloud-dense-skin-source-fit-v2/source-hand-candidate.npz')]:
 if fit is None:v,j,_=m.record(record)
 else:
  sources[str(fit.relative_to(S))]=sha(fit)
  with np.load(fit) as z:
   if 'frame_indices' in z:i=int(np.flatnonzero(z['frame_indices']==50)[0]);pose=z['mano_pose'][i];trans=z['camera_translation'][i]
   else:pose=z['mano_pose'];trans=z['camera_translation']
  v,j,_=m.record(record,pose=pose,translation=trans)
 uv=project(j);joint_sets[name]=uv
 for n,candidate in enumerate(candidates):
  points=np.asarray(candidate['landmarks_px']);delta=np.linalg.norm(uv-points,axis=1)
  rows.append(dict(prior=name,RGB_candidate_index=n,detector_handedness_label_not_identity=candidate['handedness_label'],all_21_inferred_image_error_RGB_px=stats(delta),palm_anchor_inferred_image_error_RGB_px=stats(delta[[0,1,5,9,13,17]]),physical_or_source_qualification=False))
# Register only existing valid stereo surface samples; nearest-pixel z-buffer, no hole filling.
dp=S/'fast-stereo-cpu-full-e037-v1/000050-depth.npz';sources[str(dp.relative_to(S))]=sha(dp)
with np.load(dp) as z:depth=z['depth_m'].copy();valid=z['left_right_consistent'].copy()&np.isfinite(depth)&(depth>.05)&(depth<3)
y,x=np.nonzero(valid);z=depth[y,x].astype(float)
xyz=np.c_[(x-rig.P1[0,2])*z/rig.P1[0,0],(y-rig.P1[1,2])*z/rig.P1[1,1],z]
rgbxyz=(xyz@rig.R1)@T[:3,:3].T+T[:3,3]
uv=cv2.projectPoints(rgbxyz,np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
u=np.rint(uv[:,0]).astype(int);v=np.rint(uv[:,1]).astype(int);w,h=a['resolution'];good=np.isfinite(uv).all(1)&(rgbxyz[:,2]>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
flat=np.full(h*w,np.inf);index=v[good]*w+u[good];np.minimum.at(flat,index,rgbxyz[good,2]);rgbdepth=flat.reshape(h,w);visibility=np.isfinite(rgbdepth)
np.savez_compressed(O/'RGB-visible-stereo-surface.npz',depth_m=np.where(visibility,rgbdepth,0).astype(np.float32),valid=visibility,frame_index=np.array(50),timestamp_s=np.array(2.5))
support=[]
for n,candidate in enumerate(candidates):
 pts=np.asarray(candidate['landmarks_px']);uu=np.rint(pts[:,0]).astype(int);vv=np.rint(pts[:,1]).astype(int)
 inside=(uu>=0)&(uu<w)&(vv>=0)&(vv<h);supported=np.zeros(21,bool);supported[inside]=visibility[vv[inside],uu[inside]]
 surface=np.full(21,np.nan);surface[supported]=rgbdepth[vv[supported],uu[supported]]
 support.append(dict(RGB_candidate_index=n,visible_surface_sample_count=int(supported.sum()),visible_surface_depth_m=[float(x) if np.isfinite(x) else None for x in surface],joint_center_depth_not_observed=True,all_source_3D_joint_flags_remain_unknown=True))
# Diagnostic overlay: learned predictions, not reconstructed source truth.
canvas=cv2.imread(str(ap));connections=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
for n,candidate in enumerate(candidates):
 pts=np.rint(candidate['landmarks_px']).astype(int);colour=(255,0,255) if n==0 else (255,200,0)
 for i,j in connections:cv2.line(canvas,tuple(pts[i]),tuple(pts[j]),colour,2)
 for i,point in enumerate(pts):cv2.circle(canvas,tuple(point),4,colour,-1)
pts=np.rint(joint_sets['raw_fresh_prior']).astype(int)
for i,j in connections:cv2.line(canvas,tuple(pts[i]),tuple(pts[j]),(0,220,0),2)
for line,text in enumerate(['RGB predictions: magenta/cyan; raw IR-derived prior: green','Estimated source geometry only; no metric or physical success qualification']):cv2.putText(canvas,text,(20,35+30*line),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
cv2.imwrite(str(O/'RGB-predicted-hands-and-prior.png'),cv2.resize(canvas,(1014,760)))
report=dict(schema='v2d.RGB-hand-source-correspondence.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_hashes=sources,frame=50,camera='ego_cam_a',full_frame_candidates_only=True,crop_detections_are_duplicates_not_independent=True,new_detector_or_depth_inference=False,source_grid_and_distortion_preserved=True,registered_surface_pixels=int(visibility.sum()),registered_surface_is_estimated_not_truth=True,registered_depth_holes_filled=False,registered_depth_pixels_are_visible_surface_not_joint_centers=True,predictions_compared=rows,surface_support=support,RGB_hand_identity_qualified=False,whole_hand_metric_accuracy_qualified=False,physical_task_accepted=False,native_reference_contact_gate_m=.001,gates_changed=False,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
