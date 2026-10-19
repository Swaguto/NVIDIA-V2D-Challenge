"""Bounded explicit inferred wrist repair; never changes observation authority."""
from pathlib import Path
import datetime, hashlib, json, sys, time
from types import SimpleNamespace

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0] = [str(R/'scripts/track3'), str(R/'scripts/dex3_training')]
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from temporal_pose_path import motion_summary
from v2d_dex3_flashchord.kinematics import URDFKinematics

O = S/'e037-motion-constrained-wrist-v3'
O.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
P = S/'e037-joint-object-native-validation-v1/frozen'
m = json.loads((P/'manifest.json').read_text())
for name, h in m['files'].items():
    assert sha(P/name) == h
with np.load(P/'targets.npz') as z:
    targets = {k:z[k].copy() for k in z.files}
with np.load(P/'aligned-source-reference.npz') as z:
    source = {k:z[k].copy() for k in z.files}
proposal_path = S/'e037-object-relative-command-proposal-v2/inferred-robot-commands.npz'
with np.load(proposal_path) as z:
    desired = z['robot_qpos'][:,1].copy()
control = targets['robot_qpos'][:,1].copy()
t = targets['timestamps_s']; n = len(t)
dt = np.diff(t); mid_dt = (dt[:-1]+dt[1:])/2
free = np.arange(25,100)
assert not source['hand_directly_observed'][free,1].any()
assert not source['hand_metric_valid'][free,1].any()
fixed = np.ones(n,bool); fixed[free] = False

def transforms(q):
    a = np.tile(np.eye(4),(len(q),1,1))
    a[:,:3,:3] = Rotation.from_quat(q[:,3:7]).as_matrix()
    a[:,:3,3] = q[:,:3]
    return a

caps = motion_summary(transforms(source['robot_qpos'][:,1]), t)
D = np.diff(np.eye(n),axis=0)/dt[:,None]
A = np.diff(D,axis=0)/mid_dt[:,None]
changing_edge = np.any(D[:,free] != 0,axis=1)
changing_acc = np.any(A[:,free] != 0,axis=1)
velocity_caps = np.full(n-1,caps['maximum_linear_speed_m_s'])
velocity_caps[changing_edge] = caps['p95_linear_speed_m_s']
angular_caps = np.full(n-1,caps['maximum_angular_speed_rad_s'])
angular_caps[changing_edge] = caps['p95_angular_speed_rad_s']
# A tighter bound only on modified edges guarantees the unchanged global p95.
margin = 1-1e-7
velocity_caps[changing_edge] *= margin
angular_caps[changing_edge] *= margin
acc_cap = caps['maximum_linear_acceleration_m_s2']*margin
alpha_cap = caps['maximum_angular_acceleration_rad_s2']*margin
weights = np.full(len(free),.1)
weights[(free>=36)&(free<70)] = 1
start = time.monotonic()

# Certify why the old [36,70) restriction cannot satisfy the original caps.
old_fixed_acc = np.all(A[:,np.arange(36,70)] == 0,axis=1)
v0 = np.einsum('ij,jk->ik',D,control[:,:3],optimize=False)
a0 = np.einsum('ij,jk->ik',A,control[:,:3],optimize=False)
fixed_violations = np.flatnonzero(old_fixed_acc & (np.linalg.norm(a0,axis=1)>caps['maximum_linear_acceleration_m_s2']+1e-9))

def position(x):
    p = control[:,:3].copy(); p[free] = x.reshape(-1,3)
    return p

def p_objective(x):
    delta = x.reshape(-1,3)-desired[free,:3]
    return .5*float(np.sum(weights[:,None]*delta**2))

def p_gradient(x):
    return (weights[:,None]*(x.reshape(-1,3)-desired[free,:3])).ravel()

def p_constraints(x):
    p = position(x)
    v = np.einsum('ij,jk->ik',D,p,optimize=False)
    a = np.einsum('ij,jk->ik',A,p,optimize=False)
    return np.r_[1-np.sum(v*v,axis=1)/velocity_caps**2,
                 1-np.sum(a*a,axis=1)/acc_cap**2]

def p_jacobian(x):
    p = position(x)
    v = np.einsum('ij,jk->ik',D,p,optimize=False)
    a = np.einsum('ij,jk->ik',A,p,optimize=False)
    return np.concatenate([
        (-2*D[:,free,None]*v[:,None,:]/velocity_caps[:,None,None]**2).reshape(n-1,-1),
        (-2*A[:,free,None]*a[:,None,:]/acc_cap**2).reshape(n-2,-1)],axis=0)

with np.load(S/'e037-motion-constrained-wrist-v2/inferred-robot-commands.npz') as z:
    convex_initializer = z['robot_qpos'][:,1].copy()
position_fit = minimize(p_objective, convex_initializer[free,:3].ravel(), jac=p_gradient,
    method='SLSQP', bounds=list(zip((control[free,:3]-.3).ravel(),(control[free,:3]+.3).ravel())),
    constraints=[dict(type='ineq',fun=p_constraints,jac=p_jacobian)],
    options=dict(maxiter=300,ftol=1e-12,disp=False))

# Reuse the completed angular solve; do not rerun it.
with np.load(S/'e037-motion-constrained-wrist-v1/inferred-robot-commands.npz') as z:
    reused_angular_commands=z['robot_qpos'][:,1].copy()
angular_proof=json.loads((S/'e037-motion-constrained-wrist-v1/report.json').read_text())
assert angular_proof['rotation_solver']['success']
rotation_fit=SimpleNamespace(success=True,message='Reused completed v1 angular solve',nit=0,fun=angular_proof['rotation_solver']['objective'])
jac_errors=[angular_proof['analytic_angular_jacobian_max_error']]
q = control.copy();q[:,:3]=position(position_fit.x)
q[free,3:7]=reused_angular_commands[free,3:7]
assert np.array_equal(q[fixed],control[fixed])
assert np.array_equal(q[:,7:],control[:,7:])
result = targets['robot_qpos'].copy();result[:,1]=q
motion = motion_summary(transforms(q),t)
over = [k for k,v in motion.items() if v>caps[k]+1e-9]

# Source image comparison uses only positive-confidence inferred evidence.
stage = np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
kin=URDFKinematics(P/'robot/urdfs/g1/dex3_right.urdf')
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz') as z:K=z['intrinsics'].copy()
image_rows=[];image_hashes={}
for frame in [40,45,50,55,67]:
    f=S/f'e037-watcloud-mask-refinement-handoff-v1/image-observations/right/{frame:06d}.json'
    d=json.loads(f.read_text());points=np.asarray(d['keypoints']);ids=np.array([4,8,12]);select=points[ids,2]>0
    assert not d['metric_valid'] and not d['directly_observed']
    errors=[]
    for commands in [control,desired,q]:
        poses=kin.poses(commands[frame,:7],dict(zip(m['joint_names']['right'],commands[frame,7:],strict=True)))
        tips=np.stack([poses['right_'+digit+'_tip'][:3,3] for digit in ['thumb','index','middle']])[select]
        world=np.einsum('ni,ij->nj',tips,stage,optimize=False);camera=source['camera_c2w'][frame]
        C=np.einsum('ni,ij->nj',world-camera[:3,3],camera[:3,:3],optimize=False)
        assert (C[:,2]>.01).all()
        uv=np.c_[K[0,0]*C[:,0]/C[:,2]+K[0,2],K[1,1]*C[:,1]/C[:,2]+K[1,2]]
        errors.append(np.linalg.norm(uv-points[ids[select],:2],axis=1))
    image_rows.append(dict(frame=frame,inferred_landmark_ids=ids[select].tolist(),control_errors_px=errors[0].tolist(),unconstrained_errors_px=errors[1].tolist(),constrained_errors_px=errors[2].tolist(),control_median_px=float(np.median(errors[0])),constrained_median_px=float(np.median(errors[2])),all_points_nonregressing_vs_control=bool(np.all(errors[2]<=errors[0]+1e-6)),direct=False,metric=False))
    image_hashes[str(f.relative_to(S))]=sha(f)
changed=np.any(q!=control,axis=1)
np.savez_compressed(O/'inferred-robot-commands.npz',robot_qpos=result,inferred_command_mask=changed,frame_indices=targets['frame_indices'],timestamps_s=t,
    original_source_valid=source['hand_valid'],original_source_directly_observed=source['hand_directly_observed'],original_source_metric_valid=source['hand_metric_valid'])
solver=lambda fit:dict(success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),objective=float(fit.fun))
report=dict(schema='v2d.motion-constrained-inferred-wrist.v3',convex_initializer_sha256=sha(S/'e037-motion-constrained-wrist-v2/inferred-robot-commands.npz'),previous_failed_or_inaccurate_solver_controls_retained=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(P/'manifest.json'),proposal_sha256=sha(proposal_path),source_image_sha256=image_hashes,
    original_caps=caps,old_gap_immutable_acceleration_violations=fixed_violations.tolist(),mutable_inferred_interval_half_open=[25,100],source_valid_direct_and_metric_flags_unchanged=True,directly_observed_and_metric_root_commands_unchanged=True,fingers_and_left_hand_unchanged=True,
    position_solver=solver(position_fit),rotation_solver=solver(rotation_fit),analytic_angular_jacobian_max_error=max(jac_errors),motion=motion,over_original_caps=over,motion_passed=not over,
    image_rows=image_rows,inferred_image_checks_passed=all(r['all_points_nonregressing_vs_control'] for r in image_rows),maximum_translation_change_m=float(np.max(np.linalg.norm(q[:,:3]-control[:,:3],axis=1))),maximum_rotation_change_rad=float(np.max((Rotation.from_quat(q[:,3:7])*Rotation.from_quat(control[:,3:7]).inv()).magnitude())),
    native_contact_certificate_invalidated=True,native_and_swept_checks_run=False,source_human_metric_recovery_certified=False,physical_task_accepted=False,training_approved=False,new_GPU_work=False,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
for name,h in m['files'].items():assert sha(P/name)==h
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
