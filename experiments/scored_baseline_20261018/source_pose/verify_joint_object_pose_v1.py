"""Independently check the fixed new pose, old source gates and unsolved coverage."""
from pathlib import Path
import datetime,hashlib,json,sys,warnings
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'scripts/track3')]
import cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from select_pose_hypothesis import observed_fit
from foundationpose_track import pose_rejection
from temporal_pose_path import motion_summary
from camera_prior import transport
from source_hand_depth import compare_estimated_surface
cv2.setNumThreads(1);O=S/'e037-joint-object-pose-verification-v1';O.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=S/'plane-aligned-native-reference-v1/frozen';manifest=json.loads((P/'manifest.json').read_text());mesh_path=P/manifest['objects'][0]['mesh'];mesh=trimesh.load(mesh_path,force='mesh',process=False);v=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
candidate_path=S/'e037-joint-object-pose-v1/unqualified-joint-object-pose.npz';report_path=S/'e037-joint-object-pose-v1/report.json';report=json.loads(report_path.read_text());assert report['solver_success']
with np.load(candidate_path) as z:pose=z['camera_from_object'].copy();heldout=z['heldout_pixels_C'].copy()
assert np.isfinite(pose).all() and abs(np.linalg.det(pose[:3,:3])-1)<1e-12
assert np.max(abs(np.einsum('ki,kj->ij',pose[:3,:3],pose[:3,:3],optimize=False)-np.eye(3)))<1e-12
with warnings.catch_warnings(record=True) as messages:
 warnings.simplefilter('always')
 matmul=v@pose[:3,:3].T+pose[:3,3]
explicit=np.einsum('nj,ij->ni',v,pose[:3,:3],optimize=False)+pose[:3,3]
columns=np.column_stack([v[:,0]*pose[i,0]+v[:,1]*pose[i,1]+v[:,2]*pose[i,2]+pose[i,3] for i in range(3)])
assert np.isfinite(matmul).all() and np.isfinite(explicit).all() and np.isfinite(columns).all()
numerical=dict(matmul_explicit_max_component_error_m=float(abs(matmul-explicit).max()),explicit_scalar_columns_max_component_error_m=float(abs(explicit-columns).max()),all_coordinates_finite=True,warnings_in_this_control=[str(x.message) for x in messages],fit_log_warnings_retained=True,warning_cause_not_proven=True)
assert numerical['matmul_explicit_max_component_error_m']<1e-12 and numerical['explicit_scalar_columns_max_component_error_m']<1e-12
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:K=z['intrinsics'].copy();original_camera=z['camera_c2w'].copy()
depth_path=S/'fast-stereo-cpu-full-e037-v1/000050-depth.npz'
with np.load(depth_path) as z:depth=z['depth_m'].copy();valid=z['left_right_consistent'].copy()
mask_path=S/'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/000050.png';mask=cv2.imread(str(mask_path),0)>0
track_path=S/'plane-aligned-pose-temporal-path-v4/inferred-object-trajectory.npz';track_report_path=S/'plane-aligned-pose-temporal-path-v4/report.json';track_report=json.loads(track_report_path.read_text())
with np.load(track_path) as z:before_world=z['candidate_world_from_object'].copy();cameras=z['candidate_camera_c2w'].copy();camera_poses=z['candidate_camera_from_object'].copy();times=z['timestamps_s'].copy();original_object=z['original_camera_from_object'].copy()
old_source=track_report['rows'][50]['source_fit_after'];new_source=observed_fit(v,faces,cKDTree(v),pose,depth,valid,mask,K)
prior=transport(camera_poses[49],np.linalg.inv(cameras[49]),np.linalg.inv(cameras[50]));reject,gate=pose_rejection(pose,depth,valid,(mask*255).astype(np.uint8),K,np.stack([v.min(0),v.max(0)]),True,prior)
source_local_comparison_passed=bool(reject is None and new_source['quality']>old_source['quality'] and new_source['silhouette_iou']>=old_source['silhouette_iou'] and new_source['median_measured_vertex_distance_m']<old_source['median_measured_vertex_distance_m'])
after_world=before_world.copy();after_world[50]=np.einsum('ij,jk->ik',cameras[50],pose,optimize=False)
before=motion_summary(before_world,times);after=motion_summary(after_world,times);original=motion_summary(np.einsum('tij,tjk->tik',original_camera,original_object,optimize=False),times)
motion_over=[k for k in original if after[k]>original[k]+1e-9]
uv=np.c_[K[0,0]*explicit[:,0]/explicit[:,2]+K[0,2],K[1,1]*explicit[:,1]/explicit[:,2]+K[1,2]];raster=np.zeros(mask.shape,np.uint8)
for tri in faces:cv2.fillConvexPoly(raster,np.rint(uv[tri]).astype(np.int32),255)
pred=raster>0;yy,xx=np.indices(mask.shape);test=(xx+yy)%2==1;iou=float((pred&mask&test).sum()/((pred|mask)&test).sum());expected=report['rows'][1];assert abs(iou-expected['checkerboard_heldout_mask_IoU'])<1e-15
c=compare_estimated_surface(explicit,faces,heldout,K,depth[heldout[:,1],heldout[:,0]],np.ones(len(heldout),bool),evidence_kind='estimated_visible_surface_depth');hit=c.model_hit;r=np.abs(c.residual_m[hit]);median=float(np.median(r));p95=float(np.percentile(r,95));assert int(hit.sum())==expected['depth']['heldout']['model_hits'];assert abs(median-expected['depth']['heldout']['hit_only_median_abs_depth_error_m'])<1e-12
count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8);component_rows=[]
for label in range(1,count):
 sample=labels[heldout[:,1],heldout[:,0]]==label
 component_rows.append(dict(component_label=label,inferred_mask_pixels=int(stats[label,cv2.CC_STAT_AREA]),bbox_xywh=stats[label,:4].tolist(),centroid_C=centroids[label].tolist(),heldout_expected=int(sample.sum()),heldout_model_hits=int((sample&hit).sum()),heldout_model_misses=int((sample&~hit).sum()),component_not_removed_or_relabelled=True))
distance=cv2.distanceTransform((~pred).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);miss_px=heldout[~hit];dist=distance[miss_px[:,1],miss_px[:,0]]
image_path=S/'e037-fresh-source-hand-request-v1/images/000050.png';image=cv2.imread(str(image_path));contours,_=cv2.findContours(raster,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(image,contours,-1,(0,200,255),1)
for p in miss_px:cv2.circle(image,tuple(map(int,p)),2,(0,0,255),1)
cv2.imwrite(str(O/'joint-pose-and-unchanged-missing-heldout-rays.png'),image[160:390,350:920])
np.savez_compressed(O/'heldout-coverage-verification.npz',heldout_pixels_C=heldout,predicted_front_depth_m=c.predicted_depth_m,estimated_source_depth_m=depth[heldout[:,1],heldout[:,0]],residual_m=c.residual_m,model_hit=hit,component_label=labels[heldout[:,1],heldout[:,0]])
output=dict(schema='v2d.joint-object-pose-independent-source-verification.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_hashes={str(p.relative_to(S)):sha(p) for p in [P/'manifest.json',mesh_path,candidate_path,report_path,B,depth_path,mask_path,track_path,track_report_path,image_path]},numerical=numerical,checkerboard_heldout_mask_IoU=iou,heldout_depth_hits=int(hit.sum()),heldout_depth_expected=len(hit),hit_only_heldout_depth_median_abs_error_m=median,hit_only_heldout_depth_p95_abs_error_m=p95,existing_source_fit_before=old_source,existing_source_fit_candidate=new_source,existing_tracker_rejection=reject,existing_tracker_gate_diagnostics=gate,existing_single_frame_source_comparison_passed=source_local_comparison_passed,hypothetical_one_frame_splice_motion=dict(original_limits_from_original_track=original,frozen_selected_track=before,hypothetical_splice=after,over_original_motion_caps=motion_over),actual_track_modified=False,components=component_rows,missing_ray_distance_to_projected_mesh_px_quantiles=np.quantile(dist,[0,.5,.95,1]).tolist(),unresolved_primary_shape_pose_or_evidence_mismatch=True,source_flags_unchanged=True,no_sample_deleted_or_relabelled=True,any_future_mutation_requires_new_native_certificates=True,required_native_reference_contact_gate_m=.001,full_source_reference_qualified=False,physical_task_accepted=False,new_GPU_work=False,new_PPO_steps=0)
for rel,h in output['source_hashes'].items():assert sha(S/rel)==h,rel
(O/'report.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
