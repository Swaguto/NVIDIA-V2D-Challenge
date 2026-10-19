"""Train-only constrained translation repair for the two source-rejected knots."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'scripts/track3')]
import cv2,numpy as np,trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from source_hand_depth import front_surface_depth
from select_pose_hypothesis import observed_fit
from temporal_pose_path import motion_summary
from foundationpose_track import pose_rejection
from camera_prior import transport
cv2.setNumThreads(1);O=S/'e037-joint-pose-translation-repair-v1';O.mkdir(exist_ok=True);start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proposal_path=S/'e037-joint-pose-temporal-compatibility-v1/unqualified-smooth-object-proposal.npz';proposal_report_path=S/'e037-joint-pose-temporal-compatibility-v1/report.json';prior=json.loads(proposal_report_path.read_text());assert prior['source_nonregression_failed_frames']==[47,49] and prior['all_original_motion_caps_passed']
with np.load(proposal_path) as z:world=z['world_from_object'].copy();camera=z['camera_from_object'].copy();times=z['timestamps_s'].copy();weights=z['proposal_weights'].copy()
original_proposal=world.copy()
track_path=S/'plane-aligned-pose-temporal-path-v4/inferred-object-trajectory.npz';track_report_path=S/'plane-aligned-pose-temporal-path-v4/report.json';frozen=json.loads(track_report_path.read_text())
with np.load(track_path) as z:cameras=z['candidate_camera_c2w'].copy();original_poses=z['candidate_camera_from_object'].copy()
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:K=z['intrinsics'].copy()
P=S/'plane-aligned-native-reference-v1/frozen';manifest=json.loads((P/'manifest.json').read_text());mesh_path=P/manifest['objects'][0]['mesh'];assert sha(mesh_path)==manifest['files'][manifest['objects'][0]['mesh']]
mesh=trimesh.load(mesh_path,force='mesh',process=False);v=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces);tree=cKDTree(v);yy,xx=np.indices((800,1280));checker=(yy+xx)%2==0
motion_caps=prior['original_motion_caps'];motion_keys=list(motion_caps);rows=[];input_hashes={}
for frame in [47,49]:
 assert time.monotonic()-start<190
 dp=S/f'fast-stereo-cpu-full-e037-v1/{frame:06d}-depth.npz';mp=S/f'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/{frame:06d}.png'
 assert sha(dp)==prior['changed_frame_input_hashes'][str(dp.relative_to(S))] and sha(mp)==prior['changed_frame_input_hashes'][str(mp.relative_to(S))]
 with np.load(dp) as z:depth=z['depth_m'].copy();valid=z['left_right_consistent'].copy()
 mask=cv2.imread(str(mp),0)>0;eligible=mask&valid&np.isfinite(depth)&(depth>0);yx=np.argwhere(eligible);parity=yx.sum(1)%2;partitions=[]
 for bit in [0,1]:
  p=yx[parity==bit];assert len(p)>512;partitions.append(p[np.linspace(0,len(p)-1,512).astype(int)][:,::-1])
 train,heldout=partitions;subset=train[np.linspace(0,511,128).astype(int)];z=depth[train[:,1],train[:,0]];points=np.c_[(train[:,0]-K[0,2])*z/K[0,0],(train[:,1]-K[1,2])*z/K[1,1],z]
 seed=camera[frame].copy();saved_world=world.copy();cache={};calls=0
 def evaluate_pose(pose):
  verts=np.einsum('nj,ij->ni',v,pose[:3,:3],optimize=False)+pose[:3,3];uv=np.c_[K[0,0]*verts[:,0]/verts[:,2]+K[0,2],K[1,1]*verts[:,1]/verts[:,2]+K[1,2]];raster=np.zeros(mask.shape,np.uint8)
  for tri in faces:cv2.fillConvexPoly(raster,np.rint(uv[tri]).astype(np.int32),255)
  pred=raster>0;iou=float((pred&mask&checker).sum()/((pred|mask)&checker).sum());front=front_surface_depth(verts,faces,subset,K);hit=np.isfinite(front);loss=np.full(128,np.log1p(25.));loss[hit]=np.log1p(((front[hit]-depth[subset[hit,1],subset[hit,0]])/.01)**2)
  local=np.einsum('ni,ij->nj',points-pose[:3,3],pose[:3,:3],optimize=False);median=float(np.median(tree.query(local,workers=1)[0]));return iou,int(hit.sum()),float(loss.mean()),median
 base_iou,base_hits,_,base_distance=evaluate_pose(original_poses[frame])
 def evaluate(x):
  nonlocal_dummy=None
  global_unused=None
  key=tuple(x)
  if cache.get('key')==key:return cache['value']
  assert time.monotonic()-start<200,'Translation repair CPU cap'
  pose=seed.copy();pose[:3,3]+=np.asarray(x)*.005;iou,hits,surface,distance=evaluate_pose(pose);hypothetical=saved_world.copy();hypothetical[frame]=np.einsum('ij,jk->ik',cameras[frame],pose,optimize=False);motion=motion_summary(hypothetical,times)
  constraints=np.r_[1-np.asarray(x),1+np.asarray(x),(base_distance-distance)/.01,iou-base_iou,(hits-base_hits)/128.,[(motion_caps[k]-motion[k])/max(motion_caps[k],1e-9) for k in motion_keys]]
  value=dict(objective=1-iou+surface,constraints=constraints,pose=pose,motion=motion,train_iou=iou,train_vertex_distance_m=distance,train_hits=hits)
  cache.update(key=key,value=value);return value
 def objective(x):return evaluate(x)['objective']
 def constraint(x):return evaluate(x)['constraints']
 fit=minimize(objective,np.zeros(3),method='COBYLA',constraints=[dict(type='ineq',fun=constraint)],options={'maxiter':400,'rhobeg':.25,'tol':1e-4,'catol':1e-7})
 final=evaluate(fit.x);pose=final['pose'];train_feasible=bool(fit.success and final['constraints'].min()>=-1e-7)
 measured=observed_fit(v,faces,tree,pose,depth,valid,mask,K);before=frozen['rows'][frame]['source_fit_after'];jump=transport(camera[frame-1],np.linalg.inv(cameras[frame-1]),np.linalg.inv(cameras[frame]));reject,_=pose_rejection(pose,depth,valid,(mask*255).astype(np.uint8),K,np.stack([v.min(0),v.max(0)]),True,jump)
 passed=bool(train_feasible and reject is None and measured['quality']>=before['quality']-1e-10 and measured['silhouette_iou']>=before['silhouette_iou']-1e-10 and measured['median_measured_vertex_distance_m']<=before['median_measured_vertex_distance_m']+1e-10)
 if passed:camera[frame]=pose;world[frame]=np.einsum('ij,jk->ik',cameras[frame],pose,optimize=False)
 rows.append(dict(frame=frame,solver_success=bool(fit.success),solver_status=int(fit.status),solver_message=str(fit.message),nfev=int(fit.nfev),translation_delta_C_m=(fit.x*.005).tolist(),training_feasible=train_feasible,minimum_training_constraint_slack=float(final['constraints'].min()),baseline_train_IoU=base_iou,final_train_IoU=final['train_iou'],baseline_train_mesh_vertex_median_m=base_distance,final_train_mesh_vertex_median_m=final['train_vertex_distance_m'],baseline_training_ray_hits=base_hits,final_training_ray_hits=final['train_hits'],all_source_comparison_before=before,all_source_comparison_proposal=measured,tracker_rejection=reject,all_source_nonregression_passed=passed,proposal_committed_only_inside_new_private_hypothesis=passed))
 np.savez_compressed(O/f'{frame:06d}-unqualified-translation-proposal.npz',camera_from_object=pose,training_pixels_C=train,heldout_pixels_C=heldout)
 input_hashes[str(dp.relative_to(S))]=sha(dp);input_hashes[str(mp.relative_to(S))]=sha(mp);(O/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
final_motion=motion_summary(world,times);over=[k for k in motion_caps if final_motion[k]>motion_caps[k]+1e-9];all_failed=[r['frame'] for r in rows if not r['all_source_nonregression_passed']]
unchanged=np.ones(len(world),bool);unchanged[[47,49]]=False;assert np.array_equal(world[unchanged],original_proposal[unchanged])
np.savez_compressed(O/'unqualified-combined-object-proposal.npz',frame_indices=np.arange(len(times)),timestamps_s=times,camera_from_object=camera,world_from_object=world,proposal_weights=weights)
report=dict(schema='v2d.joint-source-object-constrained-translation-repair.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_hashes={str(p.relative_to(S)):sha(p) for p in [proposal_path,proposal_report_path,track_path,track_report_path,B,P/'manifest.json',mesh_path]},changed_frame_input_hashes=input_hashes,frames=[47,49],translation_component_bounds_m=.005,rotations_and_all_other_knots_fixed=True,training_objective='Same joint even-pixel silhouette and128ray Cauchy10mm source loss; hard training constraints on image,512point vertex median,128ray coverage and all original full motion caps',heldout_values_used_in_optimizer=False,original_4096point_gate_checked_after_optimizer=True,solver_evaluation_cap_per_frame=400,rows=rows,source_failed_frames=all_failed,original_motion_caps=motion_caps,final_motion=final_motion,over_original_motion_caps=over,all19changed_source_frames_passed=not all_failed and not over,source_flag_confidence_or_old_asset_changed=False,old_failed_controls_retained=True,full_reference_promoted=False,physical_task_accepted=False,required_native_reference_contact_gate_m=.001,native_contact_swept_or_grasp_certified=False,new_GPU_work=False,new_PPO_steps=0)
for rel,h in report['source_hashes'].items():assert sha(S/rel)==h,rel
for rel,h in input_hashes.items():assert sha(S/rel)==h,rel
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
