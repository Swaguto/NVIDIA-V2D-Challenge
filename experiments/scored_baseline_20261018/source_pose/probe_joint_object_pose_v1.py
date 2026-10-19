"""One bounded fixed-mesh joint silhouette/first-surface pose experiment.

The finite missing-ray loss is an optimizer device only. Reports retain true
infinite residuals and missing coverage through the typed surface operator.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'scripts/track3')]
import cv2,numpy as np,trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from source_hand_depth import compare_estimated_surface,front_surface_depth
cv2.setNumThreads(1);O=S/'e037-joint-object-pose-v1';O.mkdir(exist_ok=True);start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=S/'plane-aligned-native-reference-v1/frozen';manifest=json.loads((P/'manifest.json').read_text());mesh_path=P/manifest['objects'][0]['mesh'];assert sha(mesh_path)==manifest['files'][manifest['objects'][0]['mesh']]
mesh=trimesh.load(mesh_path,force='mesh',process=False);vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
pool_path=S/'plane-aligned-pose-pools-v7/report.json';selection_path=S/'plane-aligned-pose-temporal-path-v4/report.json';pool=json.loads(pool_path.read_text());selection=json.loads(selection_path.read_text());assert sha(pool_path)==selection['candidate_pool_report_sha256'];index=selection['selected_pool_indices'][50];seed=np.asarray(pool['frames'][50]['retained_candidates'][index]['camera_from_object'])
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:K=z['intrinsics'].copy()
partition_path=S/'e037-visible-object-pose-probe-v1/unqualified-object-pose-proposals.npz'
with np.load(partition_path) as z:all_train=z['train_pixels_C'].copy();heldout=z['heldout_pixels_C'].copy()
train_index=np.linspace(0,len(all_train)-1,128).astype(int);train=all_train[train_index];assert len(np.unique(train_index))==128
depth_path=S/'fast-stereo-cpu-full-e037-v1/000050-depth.npz'
with np.load(depth_path) as z:depth=z['depth_m'].copy();valid=z['left_right_consistent'].copy()
mask_path=S/'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/000050.png';mask=cv2.imread(str(mask_path),0)>0
image_path=S/'e037-fresh-source-hand-request-v1/images/000050.png';source=cv2.imread(str(image_path))
hand_path=S/'e037-source-hand-object-registration-v1/registered-proxy-and-source-samples.npz'
with np.load(hand_path) as z:tips=z['hand_joints_C'][[4,8,12,16,20]].copy()
for px in [all_train,heldout]:assert valid[px[:,1],px[:,0]].all() and mask[px[:,1],px[:,0]].all()
yy,xx=np.indices(mask.shape);checker=(xx+yy)%2==0
def pose_of(x):
 pose=seed.copy();pose[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@seed[:3,:3];pose[:3,3]=seed[:3,3]+x[3:];return pose
def projected(pose):
 v=vertices@pose[:3,:3].T+pose[:3,3];assert (v[:,2]>0).all()
 uv=np.c_[K[0,0]*v[:,0]/v[:,2]+K[0,2],K[1,1]*v[:,1]/v[:,2]+K[1,2]];raster=np.zeros(mask.shape,np.uint8)
 for tri in faces:cv2.fillConvexPoly(raster,np.rint(uv[tri]).astype(np.int32),255)
 return raster>0,v
def iou(pred,part):return float((pred&mask&part).sum()/((pred|mask)&part).sum())
calls=0;best=None;trace=[];depth_scale=.01;missing_loss=float(np.log1p((.05/depth_scale)**2))
def objective(x):
 global calls,best
 assert time.monotonic()-start<205,'Joint CPU probe internal wall cap'
 pose=pose_of(x);pred,v=projected(pose);front=front_surface_depth(v,faces,train,K);hit=np.isfinite(front)
 loss=np.full(len(train),missing_loss);residual=front[hit]-depth[train[hit,1],train[hit,0]];loss[hit]=np.log1p((residual/depth_scale)**2)
 image_loss=1-iou(pred,checker);surface_loss=float(np.mean(loss));total=image_loss+surface_loss;calls+=1
 if best is None or total<best['objective']:
  best=dict(objective=float(total),image_loss=float(image_loss),surface_loss=surface_loss,training_surface_hits=int(hit.sum()),training_surface_misses=int((~hit).sum()),parameters=x.tolist(),evaluation=calls);trace.append(best.copy());(O/'training-best-trace.json').write_text(json.dumps(trace,indent=2)+'\n')
 return total
initial_objective=objective(np.zeros(6))
fit=minimize(objective,np.zeros(6),method='Powell',bounds=[(-.35,.35)]*3+[(-.05,.05)]*3,options={'maxfev':600,'maxiter':30,'xtol':1e-4,'ftol':1e-4})
candidate=pose_of(fit.x);rows=[];panels=[]
for name,pose in [('frozen_selected',seed),('unqualified_joint_pose',candidate)]:
 pred,v=projected(pose);cs=[]
 for label,px in [('all_training',all_train),('heldout',heldout)]:
  c=compare_estimated_surface(v,faces,px,K,depth[px[:,1],px[:,0]],np.ones(len(px),bool),evidence_kind='estimated_visible_surface_depth');hit=c.model_hit;r=np.abs(c.residual_m[hit]);cs.append((label,dict(expected_rays=len(px),model_hits=int(hit.sum()),model_misses=int((~hit).sum()),hit_only_median_abs_depth_error_m=float(np.median(r)) if hit.any() else None,hit_only_p95_abs_depth_error_m=float(np.percentile(r,95)) if hit.any() else None,all_expected_rays_covered=bool(hit.all()),missed_expected_residuals_remain_infinite=bool(np.isinf(c.residual_m[~hit]).all()))))
 _,distance,_=trimesh.proximity.closest_point_naive(trimesh.Trimesh(v,faces,process=False),tips)
 rows.append(dict(name=name,train_mask_IoU=iou(pred,checker),checkerboard_heldout_mask_IoU=iou(pred,~checker),all_mask_IoU=iou(pred,np.ones(mask.shape,bool)),inferred_mask_coverage=float((pred&mask).sum()/mask.sum()),depth=dict(cs),inferred_tip_distances_m=distance.tolist(),inferred_tip_median_distance_m=float(np.median(distance))))
 panel=source.copy();contours,_=cv2.findContours(pred.astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,(0,200,255),1);panel=panel[160:390,350:900];cv2.putText(panel,f'{name}: IoU{rows[-1]["all_mask_IoU"]:.3f}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,240),1);panels.append(panel)
cv2.imwrite(str(O/'frozen-and-joint-pose.png'),np.vstack(panels))
np.savez_compressed(O/'unqualified-joint-object-pose.npz',frame_index=np.array(50),camera_from_object=candidate,training_subsample_indices=train_index,training_pixels_C=train,all_training_pixels_C=all_train,heldout_pixels_C=heldout)
report=dict(schema='v2d.joint-first-surface-and-image-pose-probe.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),surface_operator_sha256=sha(R/'scripts/track3/source_hand_depth.py'),source_hashes={str(p.relative_to(S)):sha(p) for p in [P/'manifest.json',mesh_path,pool_path,selection_path,B,partition_path,depth_path,mask_path,image_path,hand_path]},frame=50,seed_pool_index=index,solver_success=bool(fit.success),solver_status=int(fit.status),solver_message=str(fit.message),objective_calls=calls,initial_training_objective=initial_objective,final_training_objective=float(fit.fun),training_best=best,parameter_delta=fit.x.tolist(),objective='Even-pixel mask1-IoU plus mean Cauchy first-surface depth loss on128fixed training pixels; missing rays use predeclared optimizer loss only',depth_loss_scale_m=depth_scale,optimization_only_missing_ray_loss=missing_loss,optimization_only_missing_ray_equivalent_m=.05,rotation_component_bounds_rad=.35,translation_component_bounds_m=.05,maxfev=600,training_depth_subsample_128_of_original_512=True,all_512_training_and_512_heldout_measurements_retained_in_reports=True,heldout_values_used_by_solver=False,hand_points_used_by_solver=False,mesh_shape_or_scale_changed=False,source_authority_or_confidence_changed=False,rows=rows,full_source_reference_qualified=False,motion_or_native_contact_or_grasp_certified=False,all_previous_controls_retained=True,new_candidate_promoted=False,required_native_reference_contact_gate_m=.001,new_GPU_work=False,new_PPO_steps=0,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
for rel,h in report['source_hashes'].items():assert sha(S/rel)==h,rel
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
