"""New one-frame estimated visible-cloud pose experiment; fixed own mesh and scale."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'scripts/track3')]
import cv2,numpy as np,trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from source_hand_depth import front_surface_depth
cv2.setNumThreads(1)
O=S/'e037-visible-object-pose-probe-v1';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=S/'plane-aligned-native-reference-v1/frozen';manifest=json.loads((P/'manifest.json').read_text());mesh_path=P/manifest['objects'][0]['mesh'];assert sha(mesh_path)==manifest['files'][manifest['objects'][0]['mesh']]
mesh=trimesh.load(mesh_path,force='mesh',process=False);vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces);tree=cKDTree(vertices)
pool_path=S/'plane-aligned-pose-pools-v7/report.json';selection_path=S/'plane-aligned-pose-temporal-path-v4/report.json';pool=json.loads(pool_path.read_text());selection=json.loads(selection_path.read_text());assert sha(pool_path)==selection['candidate_pool_report_sha256']
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:K=z['intrinsics'].copy()
depth_path=S/'fast-stereo-cpu-full-e037-v1/000050-depth.npz'
with np.load(depth_path) as z:depth=z['depth_m'].copy();valid=z['left_right_consistent'].copy()
mask_path=S/'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/000050.png';mask=cv2.imread(str(mask_path),0)>0
source_path=S/'e037-fresh-source-hand-request-v1/images/000050.png';source=cv2.imread(str(source_path))
valid &= mask & np.isfinite(depth) & (depth>0)
all_yx=np.argwhere(valid);parity=all_yx.sum(1)%2
parts=[]
for bit in [0,1]:
 p=all_yx[parity==bit];assert len(p)>512;parts.append(p[np.linspace(0,len(p)-1,512).astype(int)][:,::-1])
train,test=parts
def backproject(px):
 z=depth[px[:,1],px[:,0]]
 return np.c_[(px[:,0]-K[0,2])*z/K[0,0],(px[:,1]-K[1,2])*z/K[1,1],z]
cloud=backproject(train);test_z=depth[test[:,1],test[:,0]]
start=time.monotonic();rows=[];poses=[];panels=[]
hand_path=S/'e037-source-hand-object-registration-v1/registered-proxy-and-source-samples.npz'
with np.load(hand_path) as z:tips=z['hand_joints_C'][[4,8,12,16,20]].copy()
def project(v):return np.c_[K[0,0]*v[:,0]/v[:,2]+K[0,2],K[1,1]*v[:,1]/v[:,2]+K[1,2]]
def measure(pose):
 v=vertices@pose[:3,:3].T+pose[:3,3];assert (v[:,2]>0).all()
 raster=np.zeros(mask.shape,np.uint8);uv=project(v)
 for tri in faces:cv2.fillConvexPoly(raster,np.rint(uv[tri]).astype(int),255)
 pred=raster>0;front=front_surface_depth(v,faces,test,K);hit=np.isfinite(front)
 _,dist,_=trimesh.proximity.closest_point_naive(trimesh.Trimesh(v,faces,process=False),tips)
 residual=abs(front[hit]-test_z[hit])
 return dict(inferred_source_mask_IoU=float((pred&mask).sum()/(pred|mask).sum()),inferred_source_mask_coverage=float((pred&mask).sum()/mask.sum()),heldout_estimated_surface_rays=len(test),heldout_front_hits=int(hit.sum()),heldout_front_misses=int((~hit).sum()),hit_only_estimated_surface_depth_median_abs_error_m=float(np.median(residual)) if hit.any() else None,hit_only_estimated_surface_depth_p95_abs_error_m=float(np.percentile(residual,95)) if hit.any() else None,inferred_tip_distances_m=dist.tolist(),inferred_tip_median_distance_m=float(np.median(dist))),raster
for i,c in enumerate(pool['frames'][50]['retained_candidates']):
 assert time.monotonic()-start<200
 seed=np.asarray(c['camera_from_object']);r0=seed[:3,:3];t0=seed[:3,3]
 def unpack(x):
  pose=np.eye(4);pose[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@r0;pose[:3,3]=t0+x[3:];return pose
 def residual(x):
  pose=unpack(x);points=(cloud-pose[:3,3])@pose[:3,:3];_,inds=tree.query(points);return (points-vertices[inds]).ravel()
 # Fixed, predeclared diagnostic bounds; these do not certify a full motion path.
 fit=least_squares(residual,np.zeros(6),bounds=([-0.35]*3+[-0.05]*3,[0.35]*3+[0.05]*3),loss='huber',f_scale=.01,max_nfev=120,ftol=1e-8,xtol=1e-8,gtol=1e-8)
 pose=unpack(fit.x);before,_=measure(seed);after,raster=measure(pose)
 row=dict(seed_pool_index=i,selected_seed_in_frozen_path=i==selection['selected_pool_indices'][50],fit_success=bool(fit.success),fit_status=int(fit.status),fit_message=str(fit.message),nfev=int(fit.nfev),njev=int(fit.njev),parameter_delta=fit.x.tolist(),active_bounds=fit.active_mask.tolist(),train_vertex_point_residual_RMS_m=float(np.sqrt(np.mean(residual(fit.x)**2))),before=before,after=after)
 rows.append(row);poses.append(pose)
 panel=source.copy();contours,_=cv2.findContours(raster,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,(0,200,255),1);panel=panel[160:390,350:900];cv2.putText(panel,f'new pose seed{i}: IoU {after["inferred_source_mask_IoU"]:.3f}, hits {after["heldout_front_hits"]}/512',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,240),1);panels.append(panel)
 (O/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
cv2.imwrite(str(O/'fixed-mesh-estimated-visible-cloud-pose-proposals.png'),np.vstack(panels))
np.savez_compressed(O/'unqualified-object-pose-proposals.npz',frame_index=np.array(50),camera_from_object=np.asarray(poses),train_pixels_C=train,heldout_pixels_C=test,train_estimated_points_C=cloud,heldout_estimated_surface_depth_m=test_z)
report=dict(schema='v2d.estimated-visible-object-cloud-pose-probe.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_hashes={str(p.relative_to(S)):sha(p) for p in [P/'manifest.json',mesh_path,pool_path,selection_path,B,depth_path,mask_path,source_path,hand_path]},frame=50,train_pixels=len(train),heldout_pixels=len(test),checkerboard_split_fixed_before_fit=True,heldout_pixels_used_by_solver=False,hand_points_used_by_solver=False,mesh_or_scale_changed=False,original_source_flags_changed=False,source_mask_depth_and_hand_are_inferred=True,loss='Huber 0.01m, estimated visible cloud to nearest fixed own-mesh vertex',rotation_component_bounds_rad=.35,translation_component_bounds_m=.05,nfev_cap_per_seed=120,rows=rows,old_fitter_or_temporal_cohort_rerun=False,full_source_reference_qualified=False,motion_quality_or_physical_grasp_certified=False,all_original_failed_controls_retained=True,required_native_reference_contact_gate_m=.001,new_candidate_promoted=False,new_GPU_work=False,new_PPO_steps=0,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
for rel,h in report['source_hashes'].items():assert sha(S/rel)==h,rel
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
