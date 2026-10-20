"""ONE NEW synthetic full-pose contract batch; no episode point or old controls."""
from pathlib import Path
import datetime,hashlib,json,os
import numpy as np
from scipy.spatial.transform import Rotation
from object_full_pose_maps_v1 import *
S=Path(__file__).resolve().parents[1];O=S/'e037-object-full-pose-contracts-v1'

def norm(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):norm(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [norm(v)for v in x]
 return x

def save(name,x):
 p=O/name
 with p.open('w')as f:json.dump(norm(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 p.chmod(0o600)

def check(name,out,passed):
 save(name+'-RETURN.json',dict(actual_RETURN=out,passed=bool(passed),synthetic_ONLY=True,diagnostic_bounds_NOT_original_tolerances=True));assert passed,name

T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[.31,-.27,.19]).as_matrix();T[:3,3]=[.2,-.4,.7]
d=np.array([.003,-.002,.001]);theta=np.array([.17,-.11,.08]);Q=exp_rotation(theta)
C=compose(T,d,theta);v=np.array([.04,.01,-.03]);expected=T[:3,:3]@(Q@v)+T[:3,3]+d;wrong=Q@(T[:3,:3]@v+T[:3,3])+d
check('NEW-own-origin-three-component-order',dict(error=np.max(abs(C[:3,:3]@v+C[:3,3]-expected)),wrong_world_origin_error=np.linalg.norm(expected-wrong),raw_pose=C),np.max(abs(C[:3,:3]@v+C[:3,3]-expected))<1e-15 and np.linalg.norm(expected-wrong)>.01)
G=np.eye(4);G[:3,:3]=Rotation.from_euler('zyx',[-.4,.22,.1]).as_matrix();G[:3,3]=[-.2,.1,.5]
a=G@C;b=compose(G@T,d,theta,G[:3,:3])
check('NEW-camera-native-gauge-composition',dict(composition_A=a,composition_B=b,error=np.max(abs(a-b))),np.max(abs(a-b))<1e-15)
P,Jd,Jr=point_factors(T,np.array([v,[-.01,.03,.02]]),theta,np.eye(3));h=2e-7;fd=[]
for j in range(3):
 e=np.eye(3)[j]*h;plus=compose(T,np.zeros(3),theta+e);minus=compose(T,np.zeros(3),theta-e);fd.append(((np.array([v,[-.01,.03,.02]])@plus[:3,:3].T)-(np.array([v,[-.01,.03,.02]])@minus[:3,:3].T))/(2*h))
FD=np.stack(fd,axis=-1);err=float(np.max(abs(FD-Jr)))
check('NEW-full-rotation-coordinate-Jacobian',dict(actual_FD=FD,analytic_J=Jr,maximum_error=err),err<1e-9)
z=compose(T,np.zeros(3),np.zeros(3));tiny=np.array([0.,0.,1e-300])
check('NEW-zero-cache-and-pure-rotation-dispatch',dict(exact_base=np.array_equal(T,z),zero=source_dispatch(np.zeros(3),np.zeros(3)),pure_rotation=source_dispatch(np.zeros(3),tiny),nonfinite=source_dispatch(np.zeros(3),[0,0,float('inf')])),np.array_equal(T,z)and source_dispatch(np.zeros(3),tiny)=='NEW_actual_training_operators_REQUIRED'and source_dispatch(np.zeros(3),[0,0,float('inf')]).startswith('UNRESOLVED'))
# ORIGINAL world-relative angular convention at nonuniform clocks; it differs
# from subtracting rotation coordinates or using body-relative vectors.
t=np.array([0.,.07,.19,.31]);angles=np.array([[.2,.1,-.1],[.21,.07,-.03],[.23,.02,.05],[.28,-.06,.12]])
R=Rotation.from_rotvec(angles);dt=np.diff(t);omega=(R[1:]*R[:-1].inv()).as_rotvec()/dt[:,None];alpha=np.diff(omega,axis=0)/((dt[:-1]+dt[1:])/2)[:,None]
body=(R[:-1].inv()*R[1:]).as_rotvec()/dt[:,None];bad=np.diff(angles,axis=0)/dt[:,None]
check('NEW-changed-angular-clock-and-p95-authority',dict(actual_world_angular=omega,actual_angular_acceleration=alpha,body_difference=np.max(abs(omega-body)),coordinate_difference=np.max(abs(omega-bad)),actual_p95=float(np.quantile(np.linalg.norm(omega,axis=1),.95)),all_six_caps_must_be_checked=True),np.max(abs(omega-body))>.01 and np.max(abs(omega-bad))>.01 and np.isfinite(alpha).all())
pi=relative_rotation_status(exp_rotation([np.pi,0,0]),np.eye(3));small=relative_rotation_status(exp_rotation([.2,.1,.03]),np.eye(3))
check('NEW-principal-log-pi-and-chart-uncertainty',dict(pi=pi,regular=small,Exp_valid_at_pi=np.isfinite(exp_rotation([np.pi,0,0])).all(),theta_to_rotation_NOT_observation=True),pi['status']=='principal_log_pi_branch'and small['status']=='finite_principal_log')
# Full source vertex depth, not center depth; no raster/NN replay.
source_vertices=np.array([[0.,0.,.002],[.02,0.,.04]]);cam=np.eye(4);cam[:3,3]=[0,0,.03];changed=compose(cam,np.zeros(3),[0.,np.pi,0.]);points=source_vertices@changed[:3,:3].T+changed[:3,3]
check('NEW-rotation-all-vertex-nearplane',dict(camera_vertices=points,center_depth=changed[2,3],all_vertices_above_original_nearplane=bool(np.all(points[:,2]>.01)),unknown_not_dropped=True),changed[2,3]>.01 and not np.all(points[:,2]>.01) and np.isfinite(points).all())
# Mesh points are already in BODY coordinates. Applying geom-local placement
# again would double-transform them. This checks a new rotating body chain.
geom=np.eye(4);geom[:3,:3]=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix();geom[:3,3]=[.02,-.01,.005]
raw=np.array([[.01,.02,-.01],[-.02,.01,.03]]);bodyV=raw@geom[:3,:3].T+geom[:3,3];direct=bodyV@C[:3,:3].T+C[:3,3];chain=C@geom;other=raw@chain[:3,:3].T+chain[:3,3]
check('NEW-compiled-body-geom-rotation-order',dict(body_vertices=bodyV,direct=direct,independent_chain=other,error=np.max(abs(direct-other))),np.max(abs(direct-other))<1e-15)
# An old translation-only separation/omission can disappear under rotation.
rod=np.array([[-1.,0,0],[1.,0,0]]);fixed=np.array([[0.,.9,0]]);old=float(np.min(np.linalg.norm(rod-fixed,axis=1)));rotated=rod@exp_rotation([0,0,np.pi/2]).T;new=float(np.min(np.linalg.norm(rotated-fixed,axis=1)))
check('NEW-rotation-invalidates-old-omissions',dict(old_endpoint_distance=old,new_endpoint_distance=new,old_fixed_axis_and_omissions_must_not_be_inherited=True),old>1 and new<.11)
names=['hand','hand','object','object','support'];ct=[1,2,1,0,1];ca=[1,2,1,0,1];pairs=eligible_pairs(names,ct,ca,'object');expected_pairs=np.array([[0,2],[4,2]],np.int32)
check('NEW-full-mask-incidence-without-broadphase',dict(pairs=pairs,all_eligible_retained=True,zero_collision_mask_object_stays_declared=True),np.array_equal(pairs,expected_pairs))
kw=dict(sealed_id='new-sealed',return_id='new-sealed',frames=list(range(151)),expected_frames=list(range(151)),training_passes=[True]*151,motion_passes=[True]*6,raw_finite=True,source_queries_correct=True)
yes=full_pose_acceptance(**kw);unknown=full_pose_acceptance(**{**kw,'training_passes':[True]*150+[None]});angularfail=full_pose_acceptance(**{**kw,'motion_passes':[True,False,True,True,True,True]});wrongid=full_pose_acceptance(**{**kw,'return_id':'other'})
check('NEW-full-pose-actual-return-acceptance',dict(pass_only_limited_scope=yes,unknown=unknown,angular_failure=angularfail,identity_failure=wrongid),yes['accepted_by_training_motion_construction']and not any(x['accepted_by_training_motion_construction']for x in [unknown,angularfail,wrongid])and not yes['whole_source_native_self_swept_motor_grasp_qualified'])
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in O.glob('*RETURN.json')};save('report.json',dict(schema='v2d.NEW-full-object-pose-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_contracts=len(files),files=files,no_episode_pose_source_raster_NN_FK_native_solver_physics_GPU=True,original_F_arrays_authority_parameters_gates_UNCHANGED=True))
print(json.dumps(dict(NEW_contracts=len(files),success=True)))
