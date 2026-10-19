"""Bound a fresh native RGB prior with the same saved image/surface observations and numeric gates."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[2];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'reconstruction/modules/v2d_stereo_depth/lib'),str(S/'compute')]
import cv2,numpy as np,trimesh
from scipy.optimize import minimize
from scipy.ndimage import binary_dilation
from PIL import Image
from stereo_depth import load_rig
from numpy_mano_source_geometry_v2 import NumpyMano
from audit_source_hand_surface_depth_v1 import ray_depth
sys.path.insert(0,str(R/'scripts/track3'))
from source_hand_depth import compare_estimated_surface
sys.path.insert(0,str(R/'scripts/track3'))
from source_hand_landmarks import convert_mano_landmarks
cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();started=time.monotonic()
O=S/'e037-native-RGB-hard-skin-coverage-fit-v2';O.mkdir(exist_ok=True);f=50
request_path=S/'e037-fresh-source-hand-request-v2/request.json';request=json.loads(request_path.read_text());row=next(x for x in request['frames'] if x['frame_idx']==f)
calpath=S/'dataset/track_3/public/meta/camera_calibration.json';cal=json.loads(calpath.read_text());rig=load_rig(str(calpath))
a=cal['cameras']['ego_cam_a'];c=cal['cameras']['ego_cam_c'];KA=np.array(a['intrinsic_matrix']);DA=np.array(a['distortion_coefficients'])
T=np.linalg.inv(np.array(a['extrinsics_to_stereo_left']))@np.array(c['extrinsics_to_stereo_left'])
with np.load(request['canonical_bundle']) as z:K=z['intrinsics'].copy();camera_to_world=z['camera_c2w'][f].copy()
assert np.allclose(K,rig.P1[:3,:3],atol=1e-5,rtol=0)
mpath=S/'e037-RGB-hand-candidate-probe-v1/report.json';mp=json.loads(mpath.read_text());candidates=[x for x in mp['candidates'] if x['crop']=='full']
# Fixed association established by the prior's projected location, not detector handedness.
association=S/'e037-RGB-hand-correspondence-v1/report.json';assoc=json.loads(association.read_text());raw=[r for r in assoc['predictions_compared'] if r['prior']=='raw_fresh_prior'];order=sorted(raw,key=lambda r:r['palm_anchor_inferred_image_error_RGB_px']['median']);assert order[0]['palm_anchor_inferred_image_error_RGB_px']['median']*5<order[1]['palm_anchor_inferred_image_error_RGB_px']['median']
rgb_index=order[0]['RGB_candidate_index'];rgb_points=np.asarray(candidates[rgb_index]['landmarks_px'])
record_path=S/'e037-native-RGB-hand-source-audit-v1/canonical-C-unqualified-record.json';record=json.loads(record_path.read_text());prior=np.r_[record['mano']['global_orient'],record['mano']['hand_pose']]
m=NumpyMano(S/'models/mano/models/MANO_RIGHT.pkl')
parent=S/'e037-watcloud-dense-skin-source-fit-v2';fit_path=parent/'source-hand-candidate.npz';parent_report=json.loads((parent/'report.json').read_text())
with np.load(fit_path) as z:train_px=z['train_pixels'].copy();test_px=z['heldout_pixels'].copy();old_pose=z['mano_pose'][0].copy();old_translation=z['camera_translation'][0].copy()
dp=S/'fast-stereo-cpu-full-e037-v1/000050-depth.npz'
with np.load(dp) as z:depth=z['depth_m'].copy()
depth_path=S/'e037-source-hand-motion-fit-v1/source-hand-candidate.npz'
with np.load(depth_path) as z:joint_depth=z['estimated_depth_m'][f].copy()
points=np.asarray(row['keypoints']);supported=points[:,2]>0;hold=np.isin(np.arange(21),[1,4,7,10,13,16,19]);old_train=supported&~hold;test=supported&hold
td=old_train&np.isfinite(joint_depth);_,local_j,_=m.record(record,translation=np.zeros(3));z=joint_depth[td];measured=np.c_[(points[td,0]-K[0,2])*z/K[0,0],(points[td,1]-K[1,2])*z/K[1,1],z];initial_t=np.median(measured-local_j[td],axis=0)
op=S/'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/000050.png';objectmask=binary_dilation(np.asarray(Image.open(op))>0,iterations=3)
conflict=objectmask[np.rint(points[:,1]).astype(int).clip(0,depth.shape[0]-1),np.rint(points[:,0]).astype(int).clip(0,depth.shape[1]-1)]
old_image_train=old_train&~conflict
z=depth[train_px[:,1],train_px[:,0]];targets=np.c_[(train_px[:,0]-K[0,2])*z/K[0,0],(train_px[:,1]-K[1,2])*z/K[1,1],z];test_z=depth[test_px[:,1],test_px[:,0]]
def pc(j):return np.c_[K[0,0]*j[:,0]/j[:,2]+K[0,2],K[1,1]*j[:,1]/j[:,2]+K[1,2]]
def pa(j):return cv2.projectPoints((j@rig.R1)@T[:3,:3].T+T[:3,3],np.zeros(3),np.zeros(3),KA,DA)[0].reshape(-1,2)
def geometry(x):return m.record(record,pose=prior+x[:48],translation=initial_t+x[48:])
rgb_train=~hold
# Convert pixel residual scale by known focal ratios; no calibration adjustment.
rgb_scale=np.array([KA[0,0]/K[0,0],KA[1,1]/K[1,1]])
def rgb_landmarks(vertices,joints):return convert_mano_landmarks(vertices,joints,source_layout='manotorch21',target_layout='hamer_smplx21')
def residual(x):
 v,j,_=geometry(x);closest,_,_=trimesh.proximity.closest_point_naive(trimesh.Trimesh(v,m.faces,process=False),targets)
 return np.r_[((pc(j)[old_image_train]-points[old_image_train,:2])/8).ravel(),((pa(rgb_landmarks(v,j))[rgb_train]-rgb_points[rgb_train])/rgb_scale/8).ravel()*np.sqrt(max(1,int(old_image_train.sum()))/int(rgb_train.sum())),((closest-targets)/.015).ravel()*np.sqrt(max(1,int(old_image_train.sum()))/len(train_px)),np.sqrt(.03)*x[:48]/.15,np.sqrt(.03)*x[48:]/.15]
bounds=np.r_[np.full(3,.52),np.full(45,.35),np.full(3,.25)];warm_source=S/'e037-native-RGB-hard-skin-coverage-fit-v1/source-hand-candidate.npz'
with np.load(warm_source) as warm:x0=np.r_[warm['mano_pose'][0]-prior,warm['camera_translation'][0]-initial_t]
assert np.all(abs(x0)<=bounds+1e-12)
def triangle_coverage(triangles,pixels):
 a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a;area=e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0];good=abs(area)>1e-8
 if not good.any():return np.full(len(pixels),-1.)
 a,e1,e2,area=a[good],e1[good],e2[good],area[good];delta=pixels[:,None,:]-a[None]
 b=(delta[:,:,0]*e2[None,:,1]-delta[:,:,1]*e2[None,:,0])/area[None];c=(e1[None,:,0]*delta[:,:,1]-e1[None,:,1]*delta[:,:,0])/area[None]
 return np.maximum.reduce(np.minimum.reduce([1-b-c,b,c]),axis=1)
control=triangle_coverage(np.array([[[0.,0.],[2.,0.],[0.,2.]]]),np.array([[.5,.5],[2.,2.]]));assert control[0]>0 and control[1]<0
reverse_control=triangle_coverage(np.array([[[0.,2.],[2.,0.],[0.,0.]]]),np.array([[.5,.5],[2.,2.]]));assert np.max(abs(control-reverse_control))<1e-12
margin=1e-5
# Recover the full pre-existing TRAIN partition; no heldout pixels or parity enter constraints.
selection_path=S/'e037-watcloud-dense-skin-source-fit-v2/inferred-visible-skin-mask.png'
selected=np.asarray(Image.open(selection_path))>0;yy,xx=np.indices(selected.shape);ys,xs=np.nonzero(selected&(xx%7==3)&(yy%7==3));partition=((xs//7+ys//7)%2)==0;coverage_px=np.c_[xs[partition],ys[partition]]
training_set=set(map(tuple,coverage_px.tolist()));assert all(tuple(p) in training_set for p in train_px.tolist());assert not any(tuple(p) in training_set for p in test_px.tolist());assert len(coverage_px)>len(train_px)

# Only the unchanged training skin pixels constrain the projected mesh union.
def coverage(x):
 vertices,_,_=geometry(x);triangles=pc(vertices)[m.faces];front=(vertices[m.faces,:,][...,2]>.01).all(1)
 return triangle_coverage(triangles[front],coverage_px)-margin

def objective(x):
 r=residual(x);a=abs(r);return .5*float(np.where(a<=1,r*r,2*a-1).sum())
progress=[]
def callback(x):
 progress.append(dict(parameters=x.tolist(),iteration=len(progress)+1,training_minimum_barycentric_slack=float(coverage(x).min()),objective=objective(x)))
 (O/'solver-progress.json').write_text(json.dumps(progress,indent=2)+'\n')
opt=minimize(objective,x0,method='SLSQP',bounds=list(zip(-bounds,bounds)),constraints=[dict(type='ineq',fun=coverage)],callback=callback,options=dict(maxiter=80,ftol=1e-8,disp=False))
v,j,_=geometry(opt.x);err=np.linalg.norm(pc(j)-points[:,:2],axis=1);surface=compare_estimated_surface(v,m.faces,test_px,K,test_z,np.ones(len(test_z),bool),evidence_kind='estimated_visible_surface_depth');front=surface.predicted_depth_m;hit=surface.model_hit;surface_error=abs(front-test_z);hd=test&np.isfinite(joint_depth)
old_med=float(np.median(err[test]));old_p95=float(np.percentile(err[test],95));legacy_depth=float(np.median(abs(j[hd,2]-joint_depth[hd])))
def rgb_stats(vertices,q):
 delta=(pa(rgb_landmarks(vertices,q))-rgb_points)/rgb_scale;e=np.linalg.norm(delta,axis=1)
 return dict(all_21_estimated_RGB_error_equivalent_stereo_px_median=float(np.median(e)),heldout_7_estimated_RGB_error_equivalent_stereo_px_median=float(np.median(e[hold])),heldout_7_estimated_RGB_error_equivalent_stereo_px_p95=float(np.percentile(e[hold],95)))
old_v,old_j,_=geometry(x0)
np.savez_compressed(O/'source-hand-candidate.npz',frame_indices=[f],mano_pose=(prior+opt.x[:48])[None],camera_translation=(initial_t+opt.x[48:])[None],vertices_world=((v@camera_to_world[:3,:3].T)+camera_to_world[:3,3])[None],joints_world=((j@camera_to_world[:3,:3].T)+camera_to_world[:3,3])[None],train_pixels=train_px,heldout_pixels=test_px,heldout_front_depth_m=front,heldout_estimated_source_depth_m=test_z)
report=dict(schema='v2d.source-RGB-multiview-hand-fit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_hashes={str(p.relative_to(S)):sha(p) for p in [S/'e037-native-RGB-hand-source-audit-v1/report.json',S/'e037-watcloud-native-RGB-v1/outputs/native-RGB/receipt.json',request_path,calpath,mpath,association,record_path,fit_path,parent/'report.json',dp,depth_path,op]},hard_training_coverage_constraints_added=True,hard_coverage_uses_heldout_pixels=False,training_coverage_margin_barycentric=margin,training_minimum_final_barycentric_slack=float(coverage(opt.x).min()),training_actual_ray_hits=int(np.isfinite(ray_depth(v,m.faces,coverage_px,K)).sum()),training_rays=len(coverage_px),original_depth_training_points=len(train_px),source_training_partition_mask_sha256=sha(selection_path),all_new_constraints_belong_to_original_training_partition=True,heldout_partition_used_in_constraints=False,surface_operator_sha256=sha(R/'scripts/track3/source_hand_depth.py'),source_surface_evidence_kind='estimated_visible_surface_depth',heldout_miss_residuals_retained_as_infinite=True,solver='SLSQP with same Huber objective plus training silhouette containment',explicit_RGB_fingertip_adapter_used=True,adapter_sha256=sha(R/'scripts/track3/source_hand_landmarks.py'),legacy_IR_projection_and_depth_semantics_unchanged=True,heldout_points_or_numeric_gates_changed=False,original_native_RGB_failed_candidate_sha256=sha(warm_source),fresh_native_RGB_prior_used=True,original_IR_prior_fits_retained=True,start='retained failed 32-ray hard-coverage candidate; same translation origin and pose bounds',objective_channels='same Huber objective, weights and bounds; containment uses the complete original checkerboard training partition instead of its 32-point subset',RGB_candidate_index=rgb_index,RGB_identity_association_is_inferred_not_human_verified=True,RGB_landmarks_are_all_inferred_and_include_occluded_joints=True,new_hand_or_depth_model_inference=False,surface_samples_reused_unchanged=True,original_source_flags_changed=False,source_camera_calibration_changed=False,old_failed_control_preserved=True,before_RGB=rgb_stats(old_v,old_j),after_RGB=rgb_stats(v,j),heldout_original_IR_image_median_px=old_med,heldout_original_IR_image_p95_px=old_p95,legacy_estimated_joint_depth_median_error_m=legacy_depth,heldout_surface_hits=int(hit.sum()),heldout_surface_rays=len(front),heldout_surface_median_error_m=float(np.median(surface_error[hit])) if hit.any() else None,heldout_surface_p95_error_m=float(np.percentile(surface_error[hit],95)) if hit.any() else None,legacy_source_screen_passed=bool(old_med<=10 and old_p95<=25 and legacy_depth<=.01),dense_source_screen_passed=bool(hit.all() and np.median(surface_error)<=.01 and old_med<=10 and old_p95<=25),pose_root_component_bound_rad=.52,pose_finger_component_bound_rad=.35,translation_optimization_component_bound_m=.25,original_bounds_and_gates_unchanged=True,native_reference_contact_tolerance_m=.001,native_reference_gate_passed=None,reference_accepted=False,training_approved=False,physical_task_accepted=False,solver_success=bool(opt.success),nfev=int(opt.nfev),wall_seconds=time.monotonic()-started,candidate_sha256=sha(O/'source-hand-candidate.npz'))
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
