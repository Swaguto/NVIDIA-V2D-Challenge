"""Test fixed-shape image-pose identifiability; no depth/hand targets in optimization."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'scripts/track3')]
import cv2,numpy as np,trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from source_hand_depth import front_surface_depth
cv2.setNumThreads(1);O=S/'e037-object-silhouette-pose-probe-v1';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.monotonic()
P=S/'plane-aligned-native-reference-v1/frozen';manifest=json.loads((P/'manifest.json').read_text());mesh_path=P/manifest['objects'][0]['mesh'];assert sha(mesh_path)==manifest['files'][manifest['objects'][0]['mesh']]
mesh=trimesh.load(mesh_path,force='mesh',process=False);vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
selection_path=S/'plane-aligned-pose-temporal-path-v4/report.json';pool_path=S/'plane-aligned-pose-pools-v7/report.json';selection=json.loads(selection_path.read_text());pool=json.loads(pool_path.read_text());assert sha(pool_path)==selection['candidate_pool_report_sha256'];index=selection['selected_pool_indices'][50];seed=np.asarray(pool['frames'][50]['retained_candidates'][index]['camera_from_object'])
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:K=z['intrinsics'].copy()
mask_path=S/'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/000050.png';mask=cv2.imread(str(mask_path),0)>0
image_path=S/'e037-fresh-source-hand-request-v1/images/000050.png';source=cv2.imread(str(image_path))
depth_path=S/'fast-stereo-cpu-full-e037-v1/000050-depth.npz'
with np.load(depth_path) as z:depth=z['depth_m'].copy()
prior_probe=S/'e037-visible-object-pose-probe-v1/unqualified-object-pose-proposals.npz'
with np.load(prior_probe) as z:heldout=z['heldout_pixels_C'].copy()
hand_path=S/'e037-source-hand-object-registration-v1/registered-proxy-and-source-samples.npz'
with np.load(hand_path) as z:tips=z['hand_joints_C'][[4,8,12,16,20]].copy()
yy,xx=np.indices(mask.shape);train=(xx+yy)%2==0;test=~train
def pose_of(x):
 p=seed.copy();p[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@seed[:3,:3];p[:2,3]=seed[:2,3]+x[3:];return p
def raster(pose):
 v=vertices@pose[:3,:3].T+pose[:3,3];assert (v[:,2]>0).all()
 uv=np.c_[K[0,0]*v[:,0]/v[:,2]+K[0,2],K[1,1]*v[:,1]/v[:,2]+K[1,2]]
 out=np.zeros(mask.shape,np.uint8)
 for tri in faces:cv2.fillConvexPoly(out,np.rint(uv[tri]).astype(np.int32),255)
 return out>0,v
def iou(pred,which):return float((pred&mask&which).sum()/( (pred|mask)&which).sum())
calls=0
def objective(x):
 global calls
 assert time.monotonic()-start<210,'CPU experiment cap'
 calls+=1
 pred,_=raster(pose_of(x));value=1-iou(pred,train)
 if calls%50==0:(O/'progress.json').write_text(json.dumps(dict(evaluations=calls,latest_train_IoU=1-value,wall_seconds=time.monotonic()-start))+'\n')
 return value
fit=minimize(objective,np.zeros(5),method='Powell',bounds=[(-.35,.35)]*3+[(-.05,.05)]*2,options={'maxfev':800,'maxiter':30,'xtol':1e-4,'ftol':1e-5})
candidate=pose_of(fit.x)
rows=[];panels=[]
for name,pose in [('frozen_selected',seed),('unqualified_image_pose',candidate)]:
 pred,v=raster(pose);front=front_surface_depth(v,faces,heldout,K);hit=np.isfinite(front);z=depth[heldout[:,1],heldout[:,0]];residual=abs(front[hit]-z[hit]);_,dist,_=trimesh.proximity.closest_point_naive(trimesh.Trimesh(v,faces,process=False),tips)
 rows.append(dict(name=name,train_mask_IoU=iou(pred,train),checkerboard_heldout_mask_IoU=iou(pred,test),all_mask_IoU=iou(pred,np.ones(mask.shape,bool)),inferred_object_mask_coverage=float((pred&mask).sum()/mask.sum()),heldout_estimated_surface_rays=len(heldout),heldout_front_hits=int(hit.sum()),heldout_front_misses=int((~hit).sum()),hit_only_estimated_surface_median_abs_depth_error_m=float(np.median(residual)) if hit.any() else None,hit_only_estimated_surface_p95_abs_depth_error_m=float(np.percentile(residual,95)) if hit.any() else None,inferred_tip_distances_m=dist.tolist(),inferred_tip_median_distance_m=float(np.median(dist))))
 panel=source.copy();contours,_=cv2.findContours(pred.astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,(0,200,255),1);panel=panel[160:390,350:900];cv2.putText(panel,f'{name}: IoU{rows[-1]["all_mask_IoU"]:.3f}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,240),1);panels.append(panel)
cv2.imwrite(str(O/'frozen-and-unqualified-image-pose.png'),np.vstack(panels))
np.savez_compressed(O/'unqualified-image-pose.npz',frame_index=np.array(50),camera_from_object=candidate,heldout_estimated_depth_pixels_C=heldout)
report=dict(schema='v2d.fixed-mesh-image-pose-identifiability-probe.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_hashes={str(p.relative_to(S)):sha(p) for p in [P/'manifest.json',mesh_path,selection_path,pool_path,B,mask_path,image_path,depth_path,prior_probe,hand_path]},frame=50,selected_seed=index,solver_success=bool(fit.success),solver_status=int(fit.status),solver_message=str(fit.message),objective_calls=calls,parameter_delta=fit.x.tolist(),depth_translation_fixed=True,mesh_shape_and_scale_fixed=True,rotation_component_bounds_rad=.35,xy_translation_component_bounds_m=.05,maxfev=800,hand_or_depth_values_used_in_solver=False,checkerboard_partition_declared_before_fit=True,checkerboard_holdout_is_not_an_independent_episode=True,entire_original_mask_including_suspect_components_retained=True,source_masks_depth_and_hand_are_estimates=True,rows=rows,full_source_reference_qualified=False,motion_limits_or_contact_certified=False,old_controls_relabelled_or_removed=False,new_candidate_promoted=False,new_GPU_work=False,new_PPO_steps=0,physical_task_accepted=False,required_native_reference_contact_gate_m=.001,wall_seconds=time.monotonic()-start)
for rel,h in report['source_hashes'].items():assert sha(S/rel)==h,rel
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
