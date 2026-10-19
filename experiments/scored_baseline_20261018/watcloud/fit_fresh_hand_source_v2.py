"""Controlled fresh-prior skin-surface fit; original joint-depth failure stays explicit."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
import trimesh

S=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(S/'compute'))
from numpy_mano_source_geometry_v2 import NumpyMano
from audit_source_hand_surface_depth_v1 import ray_depth

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

O=S/'e037-watcloud-fresh-hand-source-fit-v2'
O.mkdir(exist_ok=True)
P=S/'e037-watcloud-fresh-hand-source-audit-v1'
raw=json.loads((P/'report.json').read_text())
request=json.loads((S/'e037-fresh-source-hand-request-v2/request.json').read_text())
bundle=Path(request['canonical_bundle'])
with np.load(bundle,allow_pickle=False) as z:
    K=z['intrinsics'].copy();clocks=z['frame_indices'].copy();timestamps=z['timestamps_s'].copy();cameras=z['camera_c2w'].copy()
with np.load(S/'e037-source-hand-motion-fit-v1/source-hand-candidate.npz',allow_pickle=False) as z:
    uv=z['landmarks_px'].copy();depth=z['estimated_depth_m'].copy()
model=S/'models/mano/models/MANO_RIGHT.pkl';m=NumpyMano(model)
hold=np.isin(np.arange(21),[1,4,7,10,13,16,19]);gates=raw['thresholds_unchanged']
report=dict(schema='v2d.fresh-source-hand-surface-fit.v2',producer_sha256=sha(Path(__file__)),
    raw_source_audit_sha256=sha(P/'report.json'),
    failed_joint_fit_report_sha256=sha(S/'e037-watcloud-fresh-hand-source-fit-v1/report.json'),
    depth_measurement_model='point-to-triangle visible skin, with heldout front-ray depth; joint screen retained separately',
    prior_failures_overridden=False,canonical_bundle_sha256=sha(bundle),
    private_model_sha256=sha(model),rows=[],gates=gates,source_depth_is_estimated_not_ground_truth=True,
    original_source_flags_changed=False,robot_reference_modified=False,robot_morphology_gates_changed=False,
    new_GPU_work=False,new_training_steps=0,full_source_clock_qualified=False,
    reference_contact_tolerance_m=.001,native_reference_contact_gate_passed=None,
    training_approved=False,physical_task_accepted=False)
save=lambda:(O/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
save();started=time.monotonic();poses=[];translations=[];vertices=[];joints=[];fitted_frames=[]
def project(j):return np.c_[K[0,0]*j[:,0]/j[:,2]+K[0,2],K[1,1]*j[:,1]/j[:,2]+K[1,2]]
for row in request['frames']:
    f=row['frame_idx'];observed=np.asarray(row['keypoints'])[:,2]>0;train=observed&~hold;test=observed&hold;td=train&np.isfinite(depth[f]);hd=test&np.isfinite(depth[f])
    if train.sum()<4 or test.sum()<2 or td.sum()<3 or hd.sum()<1:
        report['rows'].append(dict(frame=f,status='insufficient_independent_source_constraints',train_points=int(train.sum()),heldout_points=int(test.sum()),train_depth_points=int(td.sum()),heldout_depth_points=int(hd.sum()),source_fit_passed=False));save();continue
    record=json.loads((P/'unqualified-handoff/priors/right'/f'{f:06d}.json').read_text())
    prior=np.r_[record['mano']['global_orient'],record['mano']['hand_pose']]
    _,local_j,_=m.record(record,translation=np.zeros(3));z=depth[f,td]
    measured=np.c_[(uv[f,td,0]-K[0,2])*z/K[0,0],(uv[f,td,1]-K[1,2])*z/K[1,1],z]
    initial_t=np.median(measured-local_j[td],axis=0)
    def geometry(x):return m.record(record,pose=prior+x[:48],translation=initial_t+x[48:])
    skin_z=depth[f,td]-.005
    skin_points=np.c_[(uv[f,td,0]-K[0,2])*skin_z/K[0,0],(uv[f,td,1]-K[1,2])*skin_z/K[1,1],skin_z]
    def residual(x):
        v,j,_=geometry(x)
        mesh=trimesh.Trimesh(v,m.faces,process=False)
        closest,_,_=trimesh.proximity.closest_point_naive(mesh,skin_points)
        return np.r_[((project(j)[train]-uv[f,train])/8).ravel(),((closest-skin_points)/.015).ravel(),np.sqrt(.03)*x[:48]/.15,np.sqrt(.03)*x[48:]/.15]
    bounds=np.r_[np.full(3,gates['pose_root_delta_bound_rad']),np.full(45,gates['pose_finger_delta_bound_rad']),np.full(3,.25)]
    opt=least_squares(residual,np.zeros(51),loss='huber',f_scale=1,bounds=(-bounds,bounds),max_nfev=80,ftol=1e-5,xtol=1e-5)
    v,j,_=geometry(opt.x);error=np.linalg.norm(project(j)-uv[f],axis=-1)
    med=float(np.median(error[test]));p95=float(np.quantile(error[test],.95));dz=float(np.median(abs(j[hd,2]-depth[f,hd])))
    passed=med<=gates['heldout_median_image_px'] and p95<=gates['heldout_p95_image_px'] and dz<=gates['heldout_median_estimated_depth_m']
    ray=ray_depth(v,m.faces,uv[f,hd],K);hits=np.isfinite(ray);surf=abs(ray-(depth[f,hd]-.005))
    c=cameras[f];vertices.append(np.einsum('ni,ji->nj',v,c[:3,:3])+c[:3,3]);joints.append(np.einsum('ni,ji->nj',j,c[:3,:3])+c[:3,3]);poses.append(prior+opt.x[:48]);translations.append(initial_t+opt.x[48:]);fitted_frames.append(f)
    visible_surface_screen=bool(hits.all() and med<=gates['heldout_median_image_px'] and p95<=gates['heldout_p95_image_px'] and np.median(surf)<=gates['heldout_median_estimated_depth_m'])
    result=dict(visible_surface_screen_passed=visible_surface_screen,legacy_joint_depth_screen_passed=bool(passed),frame=f,status='source_screen_passed_only' if passed else 'source_screen_rejected',source_fit_passed=bool(passed),heldout_image_median_px=med,heldout_image_p95_px=p95,heldout_estimated_depth_median_error_m=dz,heldout_surface_hits=int(hits.sum()),heldout_surface_rays=len(ray),heldout_surface_median_error_m=float(np.median(surf[hits])) if hits.any() else None,depth_initialization_shift_from_raw_model_m=float(np.linalg.norm(initial_t-record['cam_t'])),maximum_root_pose_component_delta_rad=float(abs(opt.x[:3]).max()),maximum_finger_pose_component_delta_rad=float(abs(opt.x[3:48]).max()),maximum_translation_optimization_component_delta_m=float(abs(opt.x[48:]).max()),solver_function_evaluations=int(opt.nfev),solver_success=bool(opt.success))
    report['rows'].append(result);save();print(json.dumps(result),flush=True)
np.savez_compressed(O/'source-hand-candidate.npz',frame_indices=np.array(fitted_frames),timestamps_s=timestamps[fitted_frames],vertices_world=np.array(vertices),joints_world=np.array(joints),mano_pose=np.array(poses),camera_translation=np.array(translations))
report.update(complete=True,visible_surface_passed_frames=[r['frame'] for r in report['rows'] if r.get('visible_surface_screen_passed')],source_fit_passed_frames=[r['frame'] for r in report['rows'] if r['source_fit_passed']],wall_seconds=time.monotonic()-started,candidate_sha256=sha(O/'source-hand-candidate.npz'))
assert sha(bundle)==report['canonical_bundle_sha256'];save();print(json.dumps(dict(complete=True,passed_frames=report['source_fit_passed_frames'])),flush=True)
