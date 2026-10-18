"""Narrow CPU articulated MANO source experiment, same image/depth holdout gates.

The raw finite MANO priors are now used instead of rigidly interpolated shape.
Pose delta bounds match issue6.image_fit (.52rad root/.35rad finger). No global
robot morphology allowance is changed. All original missing flags survive.
"""
from pathlib import Path
import hashlib,json,time,sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from numpy_mano_source_geometry_v2 import NumpyMano
from audit_source_hand_surface_depth_v1 import ray_depth
S=Path(__file__).resolve().parents[1];O=S/'e037-articulated-source-hand-fit-v2';O.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz';P=S/'e037-source-hand-motion-fit-v1';model=S/'models/mano/models/MANO_RIGHT.pkl'
m=NumpyMano(model)
with np.load(B) as z:b={k:z[k].copy() for k in ['camera_c2w','intrinsics','hand_joints_world','hand_vertices_world','hand_right_faces','hand_valid','hand_directly_observed','hand_metric_valid','frame_indices','timestamps_s']}
with np.load(P/'source-hand-candidate.npz') as z:uv=z['landmarks_px'].copy();depth=z['estimated_depth_m'].copy()
K=b['intrinsics'];frames=[25,35,45,50,67,75,89,99];hold=np.isin(np.arange(21),[1,4,7,10,13,16,19]);report=dict(schema='v2d.articulated-source-hand-fit.v1',producer_sha256=sha(Path(__file__)),geometry_producer_sha256=sha(S/'compute/numpy_mano_source_geometry_v2.py'),bundle_sha256=sha(B),input_source_candidate_sha256=sha(P/'source-hand-candidate.npz'),private_model_sha256=sha(model),selected_frames=frames,objective_change='Joint-centre depth residual replaced by nearest hand skin surface for measured visible depth; old joint gate still reported and not overridden',gates=dict(pose_root_delta_bound_rad=.52,pose_finger_delta_bound_rad=.35,heldout_median_image_px=10,heldout_p95_image_px=25,heldout_median_estimated_depth_m=.01),rows=[],original_flags_changed=False,robot_morphology_gate_changed=False,training_approved=False,physical_task_accepted=False)
save=lambda:(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');started=time.monotonic();save()
# Full finite-native closure before optimisation (no CNN inference).
closure=[]
for f in np.flatnonzero(np.isfinite(b['hand_joints_world'][:,1]).all((1,2))):
 record=json.load(open(B.parent/f'source/hands/right/{f:06d}.json'));v,j,_=m.record(record);c=b['camera_c2w'][f];nj=np.einsum('ni,ij->nj',b['hand_joints_world'][f,1]-c[:3,3],c[:3,:3]);nv=np.einsum('ni,ij->nj',b['hand_vertices_world'][f,1]-c[:3,3],c[:3,:3]);closure.append(max(float(abs(nj-j).max()),float(abs(nv-v).max())))
assert max(closure)<1e-6;report['native_geometry_closure']={'finite_source_frames':len(closure),'maximum_error_m':max(closure),'passed':True};save()
outputs=[]
def project(j):return np.c_[K[0,0]*j[:,0]/j[:,2]+K[0,2],K[1,1]*j[:,1]/j[:,2]+K[1,2]]
for f in frames:
 observed=np.isfinite(uv[f]).all(-1);train=observed&~hold;test=observed&hold;td=train&np.isfinite(depth[f]);hd=test&np.isfinite(depth[f]);assert train.sum()>=4 and test.sum()>=2 and td.sum()>=3 and hd.sum()>=1
 record=json.load(open(B.parent/f'source/hands/right/{f:06d}.json'));prior=np.r_[record['mano']['global_orient'],record['mano']['hand_pose']];_,prior_j,_=m.record(record,translation=np.zeros(3));z=depth[f,td];measured=np.c_[(uv[f,td,0]-K[0,2])*z/K[0,0],(uv[f,td,1]-K[1,2])*z/K[1,1],z];initial_t=np.median(measured-prior_j[td],axis=0);surface_points=measured.copy();surface_points[:,2]-=.005;surface_points[:,:2]=np.c_[(uv[f,td,0]-K[0,2])*surface_points[:,2]/K[0,0],(uv[f,td,1]-K[1,2])*surface_points[:,2]/K[1,1]]
 def geometry(x):return m.record(record,pose=prior+x[:48],translation=initial_t+x[48:])
 def residual(x):
  v,j,_=geometry(x);image=(project(j)[train]-uv[f,train])/8;_,nearest=cKDTree(v).query(surface_points);surface=(v[nearest]-surface_points)/.015;return np.r_[image.ravel(),surface.ravel(),np.sqrt(.03)*x[:48]/.15,np.sqrt(.03)*x[48:]/.15]
 bounds=np.r_[np.full(3,.52),np.full(45,.35),np.full(3,.25)];opt=least_squares(residual,np.zeros(51),loss='huber',f_scale=1,bounds=(-bounds,bounds),max_nfev=80,ftol=1e-5,xtol=1e-5)
 v,j,g=geometry(opt.x);errors=np.linalg.norm(project(j)-uv[f],axis=-1);de=abs(j[hd,2]-depth[f,hd]);median=float(np.median(errors[test]));p95=float(np.quantile(errors[test],.95));dm=float(np.median(de));passed=median<=10 and p95<=25 and dm<=.01
 rayids=np.flatnonzero(hd);rays=ray_depth(v,b['hand_right_faces'],uv[f,rayids],K);rayvalid=np.isfinite(rays);surfaceerror=abs(rays-(depth[f,rayids]-.005));c=b['camera_c2w'][f];worldv=np.einsum('ni,ji->nj',v,c[:3,:3])+c[:3,3];worldj=np.einsum('ni,ji->nj',j,c[:3,:3])+c[:3,3]
 outputs.append((worldv,worldj,prior+opt.x[:48],initial_t+opt.x[48:]));row=dict(frame=f,source_fit_passed=bool(passed),heldout_image_median_px=median,heldout_image_p95_px=p95,heldout_estimated_depth_median_error_m=dm,heldout_surface_hits=int(rayvalid.sum()),heldout_surface_rays=len(rays),heldout_surface_median_error_m=float(np.median(surfaceerror[rayvalid])) if rayvalid.any() else None,heldout_surface_p95_error_m=float(np.quantile(surfaceerror[rayvalid],.95)) if rayvalid.any() else None,heldout_source_surface_stage_passed=bool(median<=10 and p95<=25 and rayvalid.all() and np.median(surfaceerror)<=.01 and np.quantile(surfaceerror,.95)<=.025),pose_root_delta_rad=float(np.linalg.norm(opt.x[:3])),maximum_finger_pose_component_delta_rad=float(abs(opt.x[3:48]).max()),solver_function_evaluations=int(opt.nfev),solver_success=bool(opt.success));report['rows'].append(row);save();print(row,flush=True)
np.savez_compressed(O/'source-hand-candidate.npz',frame_indices=np.array(frames),timestamps_s=b['timestamps_s'][frames],vertices_world=np.array([x[0] for x in outputs]),joints_world=np.array([x[1] for x in outputs]),mano_pose=np.array([x[2] for x in outputs]),camera_translation=np.array([x[3] for x in outputs]),original_hand_valid=b['hand_valid'][frames],original_hand_directly_observed=b['hand_directly_observed'][frames],original_hand_metric_valid=b['hand_metric_valid'][frames])
report.update(complete=True,source_fit_passed_frames=[r['frame'] for r in report['rows'] if r['source_fit_passed']],full_reference_qualified=False,wall_seconds=time.monotonic()-started,candidate_sha256=sha(O/'source-hand-candidate.npz'));assert sha(B)==report['bundle_sha256'];save();print('complete',report['source_fit_passed_frames'],flush=True)
