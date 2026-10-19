"""One new peak-frame skin control; mask conflicts become unknown, never labels."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,binary_erosion
from scipy.optimize import least_squares
import trimesh
S=Path(__file__).resolve().parents[2];sys.path.insert(0,str(S/'compute'))
from numpy_mano_source_geometry_v2 import NumpyMano
from audit_source_hand_surface_depth_v1 import ray_depth
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
O=S/'e037-watcloud-dense-skin-source-fit-v1';O.mkdir(exist_ok=True);f=50
request_path=S/'e037-fresh-source-hand-request-v2/request.json';request=json.loads(request_path.read_text());row=next(r for r in request['frames'] if r['frame_idx']==f)
B=Path(request['canonical_bundle']);handmask_path=S/f'e037-watcloud-fresh-hand-v1/outputs/recovery/image-observations/right/{f:06d}-mask.png';objectmask_path=S/f'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/{f:06d}.png';record_path=S/f'e037-watcloud-fresh-hand-source-audit-v1/unqualified-handoff/priors/right/{f:06d}.json'
record=json.loads(record_path.read_text());handmask=np.asarray(Image.open(handmask_path))>0;objectmask=np.asarray(Image.open(objectmask_path))>0
depth_path=S/f'fast-stereo-cpu-full-e037-v1/{f:06d}-depth.npz';depth_report_path=depth_path.parent/'depth-report.json';depth_report=json.loads(depth_report_path.read_text());expected=next(x for x in depth_report['complete'] if x['frame']==f);assert sha(depth_path)==expected['sha256']
with np.load(depth_path,allow_pickle=False) as z:depth=z['depth_m'].copy();lr=z['left_right_consistent'].copy();assert int(z['frame_index'])==f
with np.load(B,allow_pickle=False) as z:K=z['intrinsics'].copy();cam=z['camera_c2w'][f].copy()
m=NumpyMano(S/'models/mano/models/MANO_RIGHT.pkl');v0,j0,_=m.record(record);project=lambda j:np.c_[K[0,0]*j[:,0]/j[:,2]+K[0,2],K[1,1]*j[:,1]/j[:,2]+K[1,2]]
prior=np.r_[record['mano']['global_orient'],record['mano']['hand_pose']];model_uv=project(j0)
# Frozen image-space wrist boundary from the fresh image prior. Its uncertainty
# remains explicit; it is not a human-labelled boundary or metric observation.
wrist=model_uv[0];palm=model_uv[[5,9,13,17]].mean(0);direction=palm-wrist;length=np.linalg.norm(direction);assert length>1
yy,xx=np.indices(depth.shape);inside_wrist=((xx-wrist[0])*direction[0]+(yy-wrist[1])*direction[1])>=-.1*length**2
crop=row['crop'];inside_crop=(abs(xx-crop['cx'])<=crop['size']/2)&(abs(yy-crop['cy'])<=crop['size']/2)
ambiguous_object=binary_dilation(objectmask,iterations=3)
selected=binary_erosion(handmask&~ambiguous_object&inside_wrist&inside_crop,iterations=3)&lr&np.isfinite(depth)&(depth>0)
Image.fromarray(selected.astype(np.uint8)*255).save(O/'inferred-visible-skin-mask.png')
ys,xs=np.nonzero(selected&(xx%7==3)&(yy%7==3));pixels=np.c_[xs,ys];assert len(pixels)>=48,len(pixels)
partition=((xs//7+ys//7)%2)==0
sets=[]
for which in (partition,~partition):
    indices=np.flatnonzero(which);indices=indices[np.linspace(0,len(indices)-1,min(32,len(indices))).round().astype(int)];sets.append(pixels[indices])
train_px,test_px=sets
points=np.asarray(row['keypoints']);observed=points[:,2]>0;hold=np.isin(np.arange(21),[1,4,7,10,13,16,19]);legacy_train=observed&~hold;test=observed&hold
with np.load(S/'e037-source-hand-motion-fit-v1/source-hand-candidate.npz') as z:joint_depth=z['estimated_depth_m'][f].copy()
td=legacy_train&np.isfinite(joint_depth);_,local_j,_=m.record(record,translation=np.zeros(3));z=joint_depth[td];measured=np.c_[(points[td,0]-K[0,2])*z/K[0,0],(points[td,1]-K[1,2])*z/K[1,1],z];initial_t=np.median(measured-local_j[td],axis=0)
def lift(px):
    z=depth[px[:,1],px[:,0]]
    return np.c_[(px[:,0]-K[0,2])*z/K[0,0],(px[:,1]-K[1,2])*z/K[1,1],z]
targets=lift(train_px);test_z=depth[test_px[:,1],test_px[:,0]]
conflict=ambiguous_object[np.rint(points[:,1]).astype(int).clip(0,depth.shape[0]-1),np.rint(points[:,0]).astype(int).clip(0,depth.shape[1]-1)]
image_train=legacy_train&~conflict
def geometry(x):return m.record(record,pose=prior+x[:48],translation=initial_t+x[48:])
def residual(x):
    v,j,_=geometry(x);closest,_,_=trimesh.proximity.closest_point_naive(trimesh.Trimesh(v,m.faces,process=False),targets)
    return np.r_[((project(j)[image_train]-points[image_train,:2])/8).ravel(),((closest-targets)/.015).ravel(),np.sqrt(.03)*x[:48]/.15,np.sqrt(.03)*x[48:]/.15]
started=time.monotonic();bounds=np.r_[np.full(3,.52),np.full(45,.35),np.full(3,.25)];opt=least_squares(residual,np.zeros(51),loss='huber',f_scale=1,bounds=(-bounds,bounds),max_nfev=80,ftol=1e-5,xtol=1e-5)
v,j,_=geometry(opt.x);errors=np.linalg.norm(project(j)-points[:,:2],axis=-1);front=ray_depth(v,m.faces,test_px,K);hit=np.isfinite(front);surface_error=abs(front-test_z);hd=test&np.isfinite(joint_depth);legacy_dz=float(np.median(abs(j[hd,2]-joint_depth[hd])));image_med=float(np.median(errors[test]));image_p95=float(np.quantile(errors[test],.95))
np.savez_compressed(O/'source-hand-candidate.npz',frame_indices=[f],vertices_world=(np.einsum('ni,ji->nj',v,cam[:3,:3])+cam[:3,3])[None],joints_world=(np.einsum('ni,ji->nj',j,cam[:3,:3])+cam[:3,3])[None],mano_pose=(prior+opt.x[:48])[None],camera_translation=(initial_t+opt.x[48:])[None],train_pixels=train_px,heldout_pixels=test_px,heldout_front_depth_m=front,heldout_estimated_source_depth_m=test_z)
report=dict(schema='v2d.fresh-hand-dense-skin-source-control.v1',frame=f,producer_sha256=sha(Path(__file__)),request_sha256=sha(request_path),source_bundle_sha256=sha(B),source_hand_mask_sha256=sha(handmask_path),inferred_object_mask_sha256=sha(objectmask_path),estimated_depth_file_sha256=sha(depth_path),raw_prior_sha256=sha(record_path),mask_object_overlap_pixels=int((handmask&objectmask).sum()),object_ambiguous_source_joint_ids=np.flatnonzero(observed&conflict).tolist(),wrist_boundary_is_inferred_from_frozen_raw_prior=True,selected_skin_pixels=int(selected.sum()),train_surface_pixels=len(train_px),heldout_surface_pixels=len(test_px),heldout_surface_hits=int(hit.sum()),heldout_front_surface_median_error_m=float(np.median(surface_error[hit])) if hit.any() else None,heldout_front_surface_p95_error_m=float(np.quantile(surface_error[hit],.95)) if hit.any() else None,heldout_source_image_median_px=image_med,heldout_source_image_p95_px=image_p95,legacy_joint_depth_median_error_m=legacy_dz,legacy_joint_screen_passed=bool(image_med<=10 and image_p95<=25 and legacy_dz<=.01),dense_surface_screen_passed=bool(hit.all() and np.median(surface_error)<=.01 and image_med<=10 and image_p95<=25),pose_root_component_bound_rad=.52,pose_finger_component_bound_rad=.35,translation_optimization_component_bound_m=.25,all_pose_bounds_unchanged=True,source_confidence_modified=False,unobserved_joints_not_promoted=True,object_overlap_is_ambiguity_not_ground_truth=True,source_depth_is_estimated_not_ground_truth=True,native_reference_contact_tolerance_m=.001,native_reference_gate_passed=None,reference_accepted=False,training_approved=False,physical_task_accepted=False,solver_success=bool(opt.success),solver_function_evaluations=int(opt.nfev),wall_seconds=time.monotonic()-started,candidate_sha256=sha(O/'source-hand-candidate.npz'))
(O/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2,allow_nan=False))
