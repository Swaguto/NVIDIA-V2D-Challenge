"""NEW full-object-pose factors. No episode scoring, solving, FK or native calls.

Each clock has three world translations and three ORIGINAL-object-frame rotation
coordinates. R_new = R_F Exp(theta); t_new=t_F+delta. All three coordinates are
free; this is unrelated to the one saved, unqualified robot wrist mode.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
 x,y,z=np.asarray(v,float);return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def exp_rotation(theta):
 a=np.asarray(theta,float)
 if a.shape!=(3,)or not np.isfinite(a).all():raise ValueError('finite three-coordinate object rotation required')
 if np.array_equal(a,np.zeros(3)):return np.eye(3)
 return Rotation.from_rotvec(a).as_matrix()


def right_jacobian(theta):
 a=np.asarray(theta,float);q=float(a@a);S=skew(a)
 if q<1e-8:
  A=.5-q/24+q*q/720-q*q*q/40320
  B=1/6-q/120+q*q/5040-q*q*q/362880
 else:
  t=np.sqrt(q);A=(1-np.cos(t))/q;B=(t-np.sin(t))/(q*t)
 return np.eye(3)-A*S+B*(S@S)


def compose(base,delta,theta,translation_linear=None):
 T=np.asarray(base,float);d=np.asarray(delta,float);a=np.asarray(theta,float)
 if T.shape!=(4,4)or not np.isfinite(T).all()or d.shape!=(3,)or not np.isfinite(d).all():raise ValueError('finite base and world translation required')
 Q=exp_rotation(a);L=np.eye(3)if translation_linear is None else np.asarray(translation_linear,float)
 if L.shape!=(3,3)or not np.isfinite(L).all():raise ValueError('finite inherited translation gauge required')
 # Exact cached base at zero; no baseline operator rerun or radius repair.
 if np.array_equal(d,np.zeros(3))and np.array_equal(a,np.zeros(3)):return T.copy()
 out=T.copy();out[:3,:3]=T[:3,:3]@Q;out[:3,3]=T[:3,3]+L@d;return out


def point_factors(base,vertices,theta,translation_linear):
 """Current-position/Jacobian map, not a new source or native measurement."""
 T=np.asarray(base,float);V=np.asarray(vertices,float);Q=exp_rotation(theta);R=T[:3,:3]@Q;L=np.asarray(translation_linear,float)
 positions=V@R.T+T[:3,3]
 Jr=right_jacobian(theta)
 rotation_J=np.stack([-R@skew(v)@Jr for v in V])
 return positions,np.broadcast_to(L,(len(V),3,3)).copy(),rotation_J


def source_dispatch(delta,theta):
 d=np.asarray(delta,float);a=np.asarray(theta,float)
 if d.shape!=(3,)or a.shape!=(3,)or not np.isfinite(d).all()or not np.isfinite(a).all():return 'UNRESOLVED_nonfinite_or_shape'
 return 'CACHED_F_BY_HASH'if np.array_equal(d,np.zeros(3))and np.array_equal(a,np.zeros(3))else'NEW_actual_training_operators_REQUIRED'


def relative_rotation_status(Rnext,Rprev):
 A=np.asarray(Rnext,float)@np.asarray(Rprev,float).T
 if not np.isfinite(A).all():return dict(status='UNRESOLVED_nonfinite_relative_rotation')
 angle=float(np.linalg.norm(Rotation.from_matrix(A).as_rotvec()))
 return dict(status='principal_log_pi_branch'if abs(angle-np.pi)<=32*np.finfo(float).eps else'finite_principal_log',angle_rad=angle,pi_branch_NOT_a_global_smooth_chart=True)


def eligible_pairs(body_names,contype,conaffinity,object_body):
 """Cross-body compiled collision-mask incidence, no broadphase omissions."""
 rows=[]
 for a,name in enumerate(body_names):
  if name==object_body:continue
  for b,other in enumerate(body_names):
   if other!=object_body:continue
   if (int(contype[a])&int(conaffinity[b]))or(int(contype[b])&int(conaffinity[a])):rows.append((a,b))
 return np.asarray(rows,dtype=np.int32).reshape(-1,2)


def full_pose_acceptance(*,sealed_id,return_id,frames,expected_frames,training_passes,motion_passes,raw_finite,source_queries_correct):
 """Complete actual-return hard filter; a pass is NOT physical qualification."""
 checks=dict(sealed_pose_identity=bool(sealed_id and sealed_id==return_id),all151_clock_identity=list(frames)==list(expected_frames),all_actual_training=bool(len(training_passes)==len(expected_frames)and all(x is True for x in training_passes)),all_six_actual_motion=bool(len(motion_passes)==6 and all(x is True for x in motion_passes)),finite_raw_pose=bool(raw_finite),actual_nonzero_rotation_dispatch=bool(source_queries_correct))
 return dict(checks=checks,accepted_by_training_motion_construction=all(checks.values()),whole_source_native_self_swept_motor_grasp_qualified=False)
