"""Bounded CPU SE(3) source-hand candidate; preserve uncertainty and failed gates.

Original finite hand shape/pose priors are interpolated in their wrist frame.
New RGB tracks + LR-checked estimated surface depth constrain motion. Finger
articulation is NOT recovered. Held-out joint indices never enter the solver.
This is an upstream source reconstruction experiment, not a <25mm robot morph
repair; no robot baseline/gate is changed and no training is authorized here.
"""
from pathlib import Path
import hashlib,json,time
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,Slerp
S=Path(__file__).resolve().parents[1];O=S/'e037-source-hand-motion-fit-v1';O.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz';I=S/'e037-inferred-hand-image-evidence-v1/image-observations.npz';D=S/'fast-stereo-cpu-full-e037-v1';lock=json.load(open(D/'depth-report.json'))
with np.load(B) as z:b={k:z[k].copy() for k in ['camera_c2w','intrinsics','hand_joints_world','hand_vertices_world','hand_wrist_world','hand_valid','hand_directly_observed','hand_metric_valid','frame_indices','timestamps_s']}
with np.load(I) as z:uv=z['landmarks_px'].copy();image_supported=z['landmark_supported'].copy()
K=b['intrinsics'];report=dict(schema='v2d.source-supported-hand-motion-fit.v1',producer_sha256=sha(Path(__file__)),bundle_sha256=sha(B),source_image_evidence_sha256=sha(I),depth_report_sha256=sha(D/'depth-report.json'),original_source_flags_changed=False,robot_reference_changed=False,robot_morphology_gate_changed=False,all_full_hand_stereo_rejections_retained=True,finger_articulation_recovered=False,training_approved=False,physical_task_accepted=False,depth_is_estimated_not_ground_truth=True,skin_to_joint_depth_prior_m=.005,heldout_joint_ids=[1,4,7,10,13,16,19],gates=dict(minimum_train_points=4,minimum_heldout_points=2,minimum_train_depth_points=3,heldout_median_pixel_error_px=10,heldout_p95_pixel_error_px=25,heldout_median_estimated_depth_error_m=.01,depth_patch_min_pixels=5,depth_patch_mad_m=.005,depth_range_m=[.15,1.4]),rows=[])
save=lambda:(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');save();started=time.monotonic()
# Exact synthetic SE(3) control for transform/projection convention, before data.
def project(p):return np.c_[K[0,0]*p[:,0]/p[:,2]+K[0,2],K[1,1]*p[:,1]/p[:,2]+K[1,2]]
def apply(x,p):return Rotation.from_rotvec(x[:3]).apply(p)+x[3:]
rng=np.random.default_rng(608);control=rng.normal(size=(12,3))*.025;truth=np.array([.2,-.1,.15,.1,.1,.6]);target=apply(truth,control)
fit=least_squares(lambda x:(apply(x,control)-target).ravel(),np.array([0,0,0,0,0,.5]),max_nfev=100)
assert np.max(abs(apply(fit.x,control)-target))<1e-8;report['synthetic_transform_control_max_error_m']=float(np.max(abs(apply(fit.x,control)-target)))
valid=np.isfinite(b['hand_joints_world'][:,1]).all((1,2))&np.isfinite(b['hand_wrist_world'][:,1]).all((1,2));anchor=np.flatnonzero(valid);assert len(anchor)>2
localj=np.zeros((len(anchor),21,3));localv=np.zeros((len(anchor),778,3))
for i,f in enumerate(anchor):
 w=b['hand_wrist_world'][f,1];localj[i]=(b['hand_joints_world'][f,1]-w[:3,3])@w[:3,:3];localv[i]=(b['hand_vertices_world'][f,1]-w[:3,3])@w[:3,:3]
times=b['timestamps_s'];allj=np.stack([np.interp(times,times[anchor],localj[:,k,c]) for k in range(21) for c in range(3)],axis=-1).reshape(151,21,3);allv=np.stack([np.interp(times,times[anchor],localv[:,k,c]) for k in range(778) for c in range(3)],axis=-1).reshape(151,778,3)
worldrot=Slerp(times[anchor],Rotation.from_matrix(b['hand_wrist_world'][anchor,1,:3,:3]))(times).as_matrix();worldt=np.stack([np.interp(times,times[anchor],b['hand_wrist_world'][anchor,1,c,3]) for c in range(3)],axis=-1)
outputj=np.full((151,21,3),np.nan);outputv=np.full((151,778,3),np.nan);outputw=np.full((151,4,4),np.nan);qualified=np.zeros(151,bool);estimated_depth=np.full((151,21),np.nan);hold=np.isin(np.arange(21),report['heldout_joint_ids']);depthfiles={}
for f in range(151):
 if image_supported[f].sum()<6:report['rows'].append(dict(frame=f,status='unknown_insufficient_source_points'));continue
 dp=D/f'{f:06d}-depth.npz';assert sha(dp)==lock['complete'][f]['sha256'];depthfiles[str(dp.relative_to(S))]=sha(dp)
 with np.load(dp) as z:depth=z['depth_m'];lr=z['left_right_consistent'];good=lr&np.isfinite(depth)&(depth>=.15)&(depth<=1.4)
 maskpath=S/f'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/{f:06d}.png';obj=cv2.imread(str(maskpath),0);assert obj is not None;good&=~(cv2.dilate((obj>0).astype(np.uint8),np.ones((9,9),np.uint8))>0)
 for k in np.flatnonzero(image_supported[f]):
  x,y=np.rint(uv[f,k]).astype(int)
  if not(4<=x<1276 and 4<=y<796):continue
  ds=depth[y-4:y+5,x-4:x+5][good[y-4:y+5,x-4:x+5]]
  if len(ds)<5:continue
  med=float(np.median(ds));mad=float(np.median(abs(ds-med)))
  if mad<=.005:estimated_depth[f,k]=med+.005
 train=image_supported[f]&~hold;test=image_supported[f]&hold;td=train&np.isfinite(estimated_depth[f]);hd=test&np.isfinite(estimated_depth[f])
 if train.sum()<4 or test.sum()<2 or td.sum()<3 or hd.sum()<1:report['rows'].append(dict(frame=f,status='unknown_insufficient_disjoint_image_depth_constraints',train_points=int(train.sum()),test_points=int(test.sum()),train_depth_points=int(td.sum()),test_depth_points=int(hd.sum())));continue
 cam=b['camera_c2w'][f];r0=cam[:3,:3].T@worldrot[f];t0=(worldt[f]-cam[:3,3])@cam[:3,:3];prior=np.r_[Rotation.from_matrix(r0).as_rotvec(),t0]
 # Initialize translation from new surface observations, not object motion.
 z=estimated_depth[f,td];measured=np.c_[(uv[f,td,0]-K[0,2])*z/K[0,0],(uv[f,td,1]-K[1,2])*z/K[1,1],z];initial=prior.copy();initial[3:]=np.median(measured-Rotation.from_rotvec(prior[:3]).apply(allj[f,td]),axis=0)
 def residual(x):
  j=apply(x,allj[f]);pixels=(project(j)[train]-uv[f,train])/8;dz=(j[td,2]-estimated_depth[f,td])/.015
  return np.r_[pixels.ravel(),dz,.01*(x[:3]-prior[:3])/.8,.01*(x[3:]-initial[3:])/.5]
 opt=least_squares(residual,initial,loss='huber',f_scale=1,max_nfev=100,bounds=(np.r_[-np.ones(3)*2*np.pi,-2,-2,.15],np.r_[np.ones(3)*2*np.pi,2,2,1.4]))
 j0=apply(prior,allj[f]);j=apply(opt.x,allj[f]);pixels=np.linalg.norm(project(j)-uv[f],axis=-1);oldpixels=np.linalg.norm(project(j0)-uv[f],axis=-1);de=abs(j[hd,2]-estimated_depth[f,hd]);med=float(np.median(pixels[test]));p95=float(np.quantile(pixels[test],.95));depthmed=float(np.median(de));passed=bool(med<=10 and p95<=25 and depthmed<=.01 and np.isfinite(j).all())
 worldj=j@cam[:3,:3].T+cam[:3,3];v=apply(opt.x,allv[f]);worldv=v@cam[:3,:3].T+cam[:3,3];w=np.eye(4);w[:3,:3]=cam[:3,:3]@Rotation.from_rotvec(opt.x[:3]).as_matrix();w[:3,3]=cam[:3,:3]@opt.x[3:]+cam[:3,3]
 outputj[f]=worldj;outputv[f]=worldv;outputw[f]=w;qualified[f]=passed
 report['rows'].append(dict(frame=f,status='source_fit_passed_candidate' if passed else 'source_fit_rejected',train_points=int(train.sum()),test_points=int(test.sum()),train_depth_points=int(td.sum()),test_depth_points=int(hd.sum()),heldout_image_median_px=med,heldout_image_p95_px=p95,heldout_image_before_median_px=float(np.median(oldpixels[test])),heldout_image_before_p95_px=float(np.quantile(oldpixels[test],.95)),heldout_estimated_depth_median_error_m=depthmed,root_correction_from_missing_source_completion_m=float(np.linalg.norm(w[:3,3]-worldt[f])),solver_success=bool(opt.success)))
 if f%20==0:save();print('frame',f,report['rows'][-1],flush=True)
np.savez_compressed(O/'source-hand-candidate.npz',frame_indices=b['frame_indices'],timestamps_s=times,hand_joints_world=outputj,hand_vertices_world=outputv,hand_wrist_world=outputw,source_fit_passed=qualified,landmarks_px=uv,estimated_depth_m=estimated_depth,original_hand_valid=b['hand_valid'],original_hand_directly_observed=b['hand_directly_observed'],original_hand_metric_valid=b['hand_metric_valid'])
report.update(all_frames_complete=True,source_fit_passed_frames=np.flatnonzero(qualified).tolist(),carry_source_fit_passed_frames=np.flatnonzero(qualified[25:100]).__add__(25).tolist(),not_a_complete_source_hand_reference=True,candidate_sha256=sha(O/'source-hand-candidate.npz'),depth_files_sha256=depthfiles,wall_seconds=time.monotonic()-started)
assert sha(B)==report['bundle_sha256'] and sha(I)==report['source_image_evidence_sha256'];report['inputs_unchanged_after']=True;save();print(json.dumps({k:v for k,v in report.items() if k not in ['rows','depth_files_sha256']},indent=2),flush=True)
