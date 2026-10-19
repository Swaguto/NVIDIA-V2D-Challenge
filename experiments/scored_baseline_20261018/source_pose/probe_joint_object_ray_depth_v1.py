"""Fit one camera-ray translation from training first-surface evidence only.

Reuse the saved image-only pose without rerunning that fit. Source depth is
estimated; a root of its hit-only median is not a physical or coverage gate.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'scripts/track3')]
import cv2,numpy as np,trimesh
from scipy.optimize import brentq
from source_hand_depth import compare_estimated_surface
cv2.setNumThreads(1);O=S/'e037-joint-object-ray-depth-v1';O.mkdir(exist_ok=True);start=time.monotonic()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=S/'plane-aligned-native-reference-v1/frozen';manifest=json.loads((P/'manifest.json').read_text());mesh_path=P/manifest['objects'][0]['mesh'];assert sha(mesh_path)==manifest['files'][manifest['objects'][0]['mesh']]
mesh=trimesh.load(mesh_path,force='mesh',process=False);vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:K=z['intrinsics'].copy()
image_pose_path=S/'e037-object-silhouette-pose-probe-v1/unqualified-image-pose.npz'
with np.load(image_pose_path) as z:seed=z['camera_from_object'].copy()
partition_path=S/'e037-visible-object-pose-probe-v1/unqualified-object-pose-proposals.npz'
with np.load(partition_path) as z:train=z['train_pixels_C'].copy();heldout=z['heldout_pixels_C'].copy()
depth_path=S/'fast-stereo-cpu-full-e037-v1/000050-depth.npz'
with np.load(depth_path) as z:depth=z['depth_m'].copy();valid=z['left_right_consistent'].copy()
mask_path=S/'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/000050.png';mask=cv2.imread(str(mask_path),0)>0
image_path=S/'e037-fresh-source-hand-request-v1/images/000050.png';source=cv2.imread(str(image_path))
hand_path=S/'e037-source-hand-object-registration-v1/registered-proxy-and-source-samples.npz'
with np.load(hand_path) as z:tips=z['hand_joints_C'][[4,8,12,16,20]].copy()
prior_report_path=S/'e037-object-silhouette-pose-probe-v1/report.json';prior_report=json.loads(prior_report_path.read_text())
for px in [train,heldout]:
 assert valid[px[:,1],px[:,0]].all() and mask[px[:,1],px[:,0]].all()
 assert np.isfinite(depth[px[:,1],px[:,0]]).all()
ray=seed[:3,3]/seed[2,3];assert abs(ray[2]-1)<1e-12
def pose_of(dz):
 pose=seed.copy();pose[:3,3]=seed[:3,3]+ray*dz;return pose
def compare(pose,px):
 v=vertices@pose[:3,:3].T+pose[:3,3]
 return compare_estimated_surface(v,faces,px,K,depth[px[:,1],px[:,0]],np.ones(len(px),bool),evidence_kind='estimated_visible_surface_depth')
trace=[]
def training_residual(dz):
 assert time.monotonic()-start<200,'Training depth probe wall cap'
 c=compare(pose_of(dz),train);hits=c.model_hit
 # Misses are retained and reported separately; the median never certifies them.
 assert hits.sum()>=len(train)//2,'Too few first-surface intersections for the scalar training statistic'
 value=float(np.median(c.residual_m[hits]))
 trace.append(dict(dz_m=float(dz),training_hits=int(hits.sum()),training_misses=int((~hits).sum()),hit_only_signed_median_residual_m=value))
 (O/'training-trace.json').write_text(json.dumps(trace,indent=2)+'\n')
 return value
low,high=-.05,.05
bracket=[];root=None;solver=None
try:
 a,b=training_residual(low),training_residual(high);bracket=[a,b]
 assert a*b<=0,'Training median root is not bracketed within the declared translation bounds'
 root,solver=brentq(training_residual,low,high,xtol=1e-7,rtol=1e-10,maxiter=24,full_output=True)
 final_residual=training_residual(root)
 solver_ok=bool(solver.converged and abs(final_residual)<1e-5)
except (AssertionError,ValueError,RuntimeError) as error:
 solver_ok=False;failure=repr(error)
else:failure=None
candidate=pose_of(root) if root is not None else None
yy,xx=np.indices(mask.shape);checker=(xx+yy)%2==0
rows=[];panels=[]
for name,pose in [('saved_rejected_image_pose',seed)]+([('unqualified_joint_ray_pose',candidate)] if candidate is not None else []):
 c=compare(pose,train);h=compare(pose,heldout);v=vertices@pose[:3,:3].T+pose[:3,3]
 uv=np.c_[K[0,0]*v[:,0]/v[:,2]+K[0,2],K[1,1]*v[:,1]/v[:,2]+K[1,2]];raster=np.zeros(mask.shape,np.uint8)
 for tri in faces:cv2.fillConvexPoly(raster,np.rint(uv[tri]).astype(np.int32),255)
 pred=raster>0
 def iou(part):return float((pred&mask&part).sum()/((pred|mask)&part).sum())
 def summary(comp):
  hits=comp.model_hit;res=np.abs(comp.residual_m[hits]);return dict(expected_rays=len(hits),model_hits=int(hits.sum()),model_misses=int((~hits).sum()),hit_only_signed_median_residual_m=float(np.median(comp.residual_m[hits])) if hits.any() else None,hit_only_median_abs_depth_error_m=float(np.median(res)) if hits.any() else None,hit_only_p95_abs_depth_error_m=float(np.percentile(res,95)) if hits.any() else None,all_expected_rays_covered=bool(hits.all()),missed_expected_residuals_remain_infinite=bool(np.isinf(comp.residual_m[~hits]).all()))
 _,dist,_=trimesh.proximity.closest_point_naive(trimesh.Trimesh(v,faces,process=False),tips)
 rows.append(dict(name=name,train_mask_IoU=iou(checker),checkerboard_heldout_mask_IoU=iou(~checker),all_mask_IoU=iou(np.ones(mask.shape,bool)),inferred_mask_coverage=float((pred&mask).sum()/mask.sum()),train_depth=summary(c),heldout_depth=summary(h),inferred_tip_distances_m=dist.tolist(),inferred_tip_median_distance_m=float(np.median(dist))))
 panel=source.copy();contours,_=cv2.findContours(raster,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,(0,200,255),1);panel=panel[160:390,350:900];cv2.putText(panel,f'{name}: IoU{rows[-1]["all_mask_IoU"]:.3f}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,240),1);panels.append(panel)
cv2.imwrite(str(O/'image-control-and-joint-ray-pose.png'),np.vstack(panels))
if candidate is not None:np.savez_compressed(O/'unqualified-joint-object-pose.npz',frame_index=np.array(50),camera_from_object=candidate,train_pixels_C=train,heldout_pixels_C=heldout,translation_ray_C=ray,translation_delta_z_m=np.array(root))
report=dict(schema='v2d.joint-source-image-and-first-surface-depth-probe.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),surface_operator_sha256=sha(R/'scripts/track3/source_hand_depth.py'),source_hashes={str(p.relative_to(S)):sha(p) for p in [P/'manifest.json',mesh_path,B,image_pose_path,partition_path,depth_path,mask_path,image_path,hand_path,prior_report_path]},frame=50,solver_success=solver_ok,solver_failure=failure,solver_flag=str(solver.flag) if solver is not None else None,solver_iterations=int(solver.iterations) if solver is not None else None,declared_training_bracket_m=[low,high],training_bracket_signed_residuals_m=bracket,translation_delta_z_m=float(root) if root is not None else None,translation_ray_C=ray.tolist(),rotation_shape_scale_and_hand_fixed=True,all_training_and_heldout_pixels_preserved=True,heldout_values_used_by_solver=False,hand_points_used_by_solver=False,existing_image_fit_rerun=False,training_objective='Signed median of estimated visible first-surface residuals over intersected training rays; misses explicitly retained and never certified by this statistic',source_authority_flags_changed=False,source_image_and_depth_are_estimates=True,rows=rows,all_earlier_controls_retained=True,numerical_or_contact_gates_relaxed=False,full_source_reference_qualified=False,motion_or_native_contact_or_grasp_certified=False,new_candidate_promoted=False,required_native_reference_contact_gate_m=.001,new_GPU_work=False,new_PPO_steps=0,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
for rel,h in report['source_hashes'].items():assert sha(S/rel)==h,rel
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
