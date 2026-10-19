"""Reuse saved features; solve only the fixed distortion inverse's failed numeric roots."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[2];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'reconstruction/modules/v2d_stereo_depth/lib')]
import cv2,numpy as np
from scipy.optimize import least_squares
from stereo_depth import load_rig
cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
O=S/'e037-RGB-stereo-registration-v3';O.mkdir(exist_ok=True);start=time.monotonic()
cp=S/'dataset/track_3/public/meta/camera_calibration.json';cal=json.loads(cp.read_text());rig=load_rig(str(cp))
a=cal['cameras']['ego_cam_a'];c=cal['cameras']['ego_cam_c'];KA=np.array(a['intrinsic_matrix']);DA=np.array(a['distortion_coefficients']);KC=np.array(c['intrinsic_matrix']);DC=np.array(c['distortion_coefficients'])
T=np.linalg.inv(np.array(a['extrinsics_to_stereo_left']))@np.array(c['extrinsics_to_stereo_left'])
def project(x,K,D):return cv2.projectPoints(np.ascontiguousarray(x,float),np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
def inverse(uv,K,D):
 xy=cv2.undistortPointsIter(np.ascontiguousarray(uv[:,None],float),K,D,None,None,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(-1,2)
 initial=np.linalg.norm(project(np.c_[xy,np.ones(len(xy))],K,D)-uv,axis=1)
 corrected=[]
 for i in np.flatnonzero(initial>1e-6):
  fit=least_squares(lambda q:project(np.r_[q,1.][None],K,D)[0]-uv[i],xy[i],method='lm',ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=100)
  xy[i]=fit.x;corrected.append(dict(index=int(i),initial_error_px=float(initial[i]),nfev=fit.nfev,jacobian_condition_number=float(np.linalg.cond(fit.jac))))
 error=np.linalg.norm(project(np.c_[xy,np.ones(len(xy))],K,D)-uv,axis=1)
 assert np.isfinite(xy).all() and error.max()<1e-6,'Fixed-calibration inverse closure must hold'
 return np.c_[xy,np.ones(len(xy))],dict(points=len(xy),initial_max_px=float(initial.max()),final_max_px=float(error.max()),corrected_roots=corrected)
def quant(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return dict(n=int(len(x)),median=float(np.median(x)) if len(x) else None,p95=float(np.percentile(x,95)) if len(x) else None,maximum=float(np.max(x)) if len(x) else None)
parent=S/'e037-RGB-stereo-registration-v2'
with np.load(parent/'correspondences.npz') as z:uc=z['stereo_feature_pixels'].copy();ua=z['RGB_feature_pixels'].copy();known=z['depth_known'].copy();error=z['RGB_depth_projection_error_px'].copy()
cr=np.c_[(uc[:,0]-rig.P1[0,2])/rig.P1[0,0],(uc[:,1]-rig.P1[1,2])/rig.P1[1,1],np.ones(len(uc))]
rawc=project(cr@rig.R1,KC,DC)
tx=np.array([[0,-T[2,3],T[1,3]],[T[2,3],0,-T[0,3]],[-T[1,3],T[0,3],0]])
E=tx@T[:3,:3];hyp=[];out={}
for fc,fa,name in [(False,False,'as_published'),(True,True,'extra_180_both'),(True,False,'extra_180_C'),(False,True,'extra_180_A')]:
 bc,cc=inverse(np.array(c['resolution'])-1-rawc if fc else rawc,KC,DC)
 ba,ac=inverse(np.array(a['resolution'])-1-ua if fa else ua,KA,DA)
 line=bc@E.T;ep=abs((ba*line).sum(1))/np.linalg.norm(line[:,:2]/[KA[0,0],KA[1,1]],axis=1)
 out[name+'_epipolar_px']=ep
 hyp.append(dict(name=name,epipolar_virtual_RGB_px=quant(ep),within_2px=int((ep<=2).sum()),within_5px=int((ep<=5).sum()),C_numeric_inverse=cc,A_numeric_inverse=ac))
consistent=known&(out['as_published_epipolar_px']<=2)
np.savez_compressed(O/'fixed-inverse-correspondences.npz',**out,depth_known=known,epipolar_consistent_known_depth=consistent,RGB_depth_projection_error_px=error)
report=dict(schema='v2d.fixed-factory-RGB-stereo-inverse-control.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_correspondences_sha256=sha(parent/'correspondences.npz'),source_report_sha256=sha(parent/'report.json'),calibration_sha256=sha(cp),appearance_matches_reused=True,new_detector_or_depth_inference=False,calibration_fitted=False,fixed_14_coefficient_inverse_roots_only=True,hypotheses=hyp,known_depth_matches=int(known.sum()),epipolar_consistent_known_depth_matches=int(consistent.sum()),consistent_depth_RGB_projection_error_px=quant(error[consistent]),factory_mapping_supported_on_some_source_features=True,whole_hand_metric_accuracy_qualified=False,physical_exposure_synchrony_verified=False,physical_task_accepted=False,original_gate_tolerances_changed=False,native_reference_contact_gate_m=.001,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='hypotheses'},indent=2));print([(h['name'],h['epipolar_virtual_RGB_px'],len(h['A_numeric_inverse']['corrected_roots'])) for h in hyp])
