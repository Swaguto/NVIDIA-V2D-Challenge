"""Screen paired source detections; no hole filling, fitting or flag promotion.

Landmark stereo is tested independently from the model's relative 3D output.
Strict predeclared epipolar/cheirality/range/estimated-depth checks preserve
per-point unknowns; matching model predictions do not establish ground truth.
"""
import hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'reconstruction/modules/v2d_stereo_depth/lib'))
from stereo_depth import load_rig
OUT=S/'mediapipe-stereo-hand-candidates-v1';OUT.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=S/'mediapipe-source-hand-candidates-full-v1/report.json';raw=json.load(open(P));assert raw['all_selected_frames_complete'] and raw['source_closure_verified_after']
rig=load_rig(str(S/'dataset/track_3/public/meta/camera_calibration.json'));D=S/'fast-stereo-cpu-full-e037-v1';depthlock=json.load(open(D/'depth-report.json'))
thresholds={'candidate_dedup_median_px':5.,'epipolar_median_px':1.,'epipolar_p95_px':3.,'minimum_positive_disparity_px':3.,'minimum_depth_m':.15,'maximum_depth_m':1.4,'maximum_independent_estimated_depth_difference_m':.01,'minimum_range_valid_points':16,'minimum_depth_checked_points':8,'minimum_hand_triangle_area_m2':.00001}
report={'schema':'v2d.stereo-hand-candidate-screen.v1','producer_sha256':sha(Path(__file__)),'raw_detector_report_sha256':sha(P),'calibration_sha256':sha(S/'dataset/track_3/public/meta/camera_calibration.json'),'thresholds':thresholds,'source_flags_changed':False,'original_hand_records_replaced':False,'metric_ground_truth_certified':False,'model_hand_relative_world_used':False,'missing_landmarks_filled':False,'physical_task_accepted':False,'rows':[]};save=lambda:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');started=time.monotonic();save()
# Synthetic camera projection control of triangulation with this exact rig.
xyz=np.array([[.03,.02,.6],[-.04,-.03,1.1],[.05,.05,.8]],dtype=float)
def project(points,P):
 homogeneous=np.c_[points,np.ones(len(points))]@P.T
 return homogeneous[:,:2]/homogeneous[:,2:]
a,b=project(xyz,rig.P1),project(xyz,rig.P2);h=cv2.triangulatePoints(rig.P1,rig.P2,a.T,b.T);np.testing.assert_allclose((h[:3]/h[3]).T,xyz,atol=1e-10,rtol=0);report['synthetic_projection_control_passed']=True
qualified=[];depthseen={}
try:
 for row in raw['rows']:
  frame=row['frame'];views=[]
  for v in row['candidate_views']:
   unique=[]
   for candidate in sorted(v['candidates'],key=lambda c:-c['handedness_probability']):
    points=np.array(candidate['landmarks_px']);assert points.shape==(21,2) and np.isfinite(points).all()
    if any(np.median(np.linalg.norm(points-np.array(other['landmarks_px']),axis=1))<=thresholds['candidate_dedup_median_px'] for other in unique):continue
    unique.append(candidate)
   views.append(unique)
  rec={'frame':frame,'dedup_hands_per_view':[len(v) for v in views],'pairs':[]}
  if all(views):
   path=D/f'{frame:06d}-depth.npz';assert sha(path)==depthlock['complete'][frame]['sha256'];depthseen[str(path.relative_to(S))]=sha(path)
   with np.load(path) as z:estimated=z['depth_m'].copy();valid=z['left_right_consistent'].copy()
   for li,left in enumerate(views[0]):
    for ri,right in enumerate(views[1]):
     l=np.array(left['landmarks_px']);r=np.array(right['landmarks_px']);epi=abs(l[:,1]-r[:,1]);disparity=l[:,0]-r[:,0]-(rig.P1[0,2]-rig.P2[0,2]);entry={'left_index':li,'right_index':ri,'left_model_label':left['handedness_label'],'right_model_label':right['handedness_label'],'epipolar_median_px':float(np.median(epi)),'epipolar_p95_px':float(np.quantile(epi,.95)),'qualified':False}
     if left['handedness_label']!=right['handedness_label']:entry['reason']='model handedness inconsistent'
     elif np.median(epi)>thresholds['epipolar_median_px'] or np.quantile(epi,.95)>thresholds['epipolar_p95_px']:entry['reason']='epipolar residual exceeds unchanged predeclared limits'
     else:
      h=cv2.triangulatePoints(rig.P1,rig.P2,l.T,r.T);points=(h[:3]/h[3]).T
      inside=(l[:,0]>=0)&(l[:,0]<rig.size[0])&(l[:,1]>=0)&(l[:,1]<rig.size[1])&(r[:,0]>=0)&(r[:,0]<rig.size[0])&(r[:,1]>=0)&(r[:,1]<rig.size[1])
      range_valid=inside&np.isfinite(points).all(axis=1)&(disparity>=thresholds['minimum_positive_disparity_px'])&(points[:,2]>=thresholds['minimum_depth_m'])&(points[:,2]<=thresholds['maximum_depth_m'])&(epi<=thresholds['epipolar_p95_px'])
      ix=np.clip(np.rint(l[:,0]).astype(int),0,rig.size[0]-1);iy=np.clip(np.rint(l[:,1]).astype(int),0,rig.size[1]-1);checked=range_valid&valid[iy,ix]&np.isfinite(estimated[iy,ix]);depth_error=abs(points[:,2]-estimated[iy,ix]);supported=checked&(depth_error<=thresholds['maximum_independent_estimated_depth_difference_m'])
      entry.update(range_valid_points=int(range_valid.sum()),independent_depth_checked_points=int(checked.sum()),depth_supported_points=int(supported.sum()),independent_depth_median_difference_m=float(np.median(depth_error[checked])) if checked.any() else None,wrist_supported=bool(supported[0]))
      triangle=float(np.linalg.norm(np.cross(points[5]-points[0],points[9]-points[0]))/2) if np.isfinite(points[[0,5,9]]).all() else 0.
      if range_valid.sum()<thresholds['minimum_range_valid_points']:entry['reason']='too few in-image finite positive-range landmarks'
      elif not supported[0] or supported.sum()<thresholds['minimum_depth_checked_points']:entry['reason']='insufficient independent estimated-depth support, including wrist'
      elif triangle<thresholds['minimum_hand_triangle_area_m2']:entry['reason']='degenerate wrist/knuckle triangle'
      else:
       entry.update(qualified=True,reason='stereo/depth-screened candidate only; anatomy/contact/temporal fit unverified')
       qualified.append({'frame':frame,'left_index':li,'right_index':ri,'model_label':left['handedness_label'],'landmarks_camera_m':np.where(supported[:,None],points,np.nan).tolist(),'landmark_supported':supported.tolist(),'left_uv_px':l.tolist(),'right_uv_px':r.tolist(),'source_confidence_promoted':False})
     rec['pairs'].append(entry)
  report['rows'].append(rec)
 report['qualified_candidate_pairs']=len(qualified);report['qualified_frames']=sorted(set(v['frame'] for v in qualified));report['qualified_carry_frames']=[f for f in report['qualified_frames'] if 25<=f<=99];report['all_frames_complete']=len(report['rows'])==151
 # JSON missing coordinates are explicit nulls, never zero-filled coordinates.
 for pair in qualified:pair['landmarks_camera_m']=[[float(v) if np.isfinite(v) else None for v in point] for point in pair['landmarks_camera_m']]
 (OUT/'unpromoted-candidate-points.json').write_text(json.dumps({'candidates':qualified},indent=2,allow_nan=False)+'\n');report['candidate_points_sha256']=sha(OUT/'unpromoted-candidate-points.json')
 for name,h in depthseen.items():assert sha(S/name)==h
 assert sha(P)==report['raw_detector_report_sha256'];report['inputs_verified_after']=True
except BaseException as error:report['error']=repr(error);raise
finally:
 report['wall_seconds']=time.monotonic()-started;save();print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)
