"""Convex translation projection with verified Euclidean norm cuts.

Retain v1's failed SLSQP position control and reuse its completed angular fit.
Cuts are outer approximations; only the final exact norm checks can qualify it.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(R/'scripts/track3'),str(R/'scripts/dex3_training')]
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation
from temporal_pose_path import motion_summary
from v2d_dex3_flashchord.kinematics import URDFKinematics
O=S/'e037-motion-constrained-wrist-v2';O.mkdir(exist_ok=True)
P=S/'e037-joint-object-native-validation-v1/frozen'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.monotonic()
m=json.loads((P/'manifest.json').read_text())
for name,h in m['files'].items():assert sha(P/name)==h
with np.load(P/'targets.npz') as z:targets={k:z[k].copy() for k in z.files}
with np.load(P/'aligned-source-reference.npz') as z:source={k:z[k].copy() for k in z.files}
desired_path=S/'e037-object-relative-command-proposal-v2/inferred-robot-commands.npz'
with np.load(desired_path) as z:desired=z['robot_qpos'][:,1].copy()
previous_path=S/'e037-motion-constrained-wrist-v1/inferred-robot-commands.npz'
with np.load(previous_path) as z:previous=z['robot_qpos'][:,1].copy()
proof=json.loads((S/'e037-motion-constrained-wrist-v1/report.json').read_text())
assert proof['rotation_solver']['success'] and not proof['position_solver']['success']
control=targets['robot_qpos'][:,1];t=targets['timestamps_s'];n=len(t);free=np.arange(25,100)
fixed=np.ones(n,bool);fixed[free]=False
assert not source['hand_directly_observed'][free,1].any() and not source['hand_metric_valid'][free,1].any()
dt=np.diff(t);D=np.diff(np.eye(n),axis=0)/dt[:,None];A=np.diff(D,axis=0)/((dt[:-1]+dt[1:])/2)[:,None]
caps=proof['original_caps'];changed_edge=np.any(D[:,free]!=0,axis=1)
velocity_caps=np.full(n-1,caps['maximum_linear_speed_m_s']);velocity_caps[changed_edge]=caps['p95_linear_speed_m_s']*(1-1e-7)
acc_caps=np.full(n-2,caps['maximum_linear_acceleration_m_s2']*(1-1e-7))
weights=np.full(len(free),.1);weights[(free>=36)&(free<70)]=1
H=sparse.diags(np.repeat(weights,3),format='csc');linear=-np.repeat(weights,3)*desired[free,:3].ravel()
zero=control[:,:3].copy();zero[free]=0
operators=[D,A];limits=[velocity_caps,acc_caps]
constants=[np.einsum('ij,jk->ik',M,zero,optimize=False) for M in operators]
matrices=[M[:,free] for M in operators]
rows=[];upper=[]
def add_cut(group,index,direction):
    M=matrices[group];norm=float(np.linalg.norm(direction));assert norm>0
    normal=direction/norm
    rows.append(sparse.csr_matrix((M[index,:,None]*normal[None,:]).reshape(1,-1)))
    upper.append(limits[group][index]-float(np.dot(normal,constants[group][index])))
for group in range(2):
    v=np.einsum('ij,jk->ik',operators[group],desired[:,:3],optimize=False)
    for i in range(len(v)):
        for axis in np.r_[np.eye(3),-np.eye(3)]:add_cut(group,i,axis)
        if np.linalg.norm(v[i])>1e-10:add_cut(group,i,v[i])
bounds=sparse.eye(len(free)*3,format='csc')
lower_box=(control[free,:3]-.3).ravel();upper_box=(control[free,:3]+.3).ravel()
history=[];x=control[free,:3].ravel();converged=False
for iteration in range(60):
    mat=sparse.vstack([sparse.vstack(rows),bounds],format='csc')
    lower=np.r_[np.full(len(rows),-np.inf),lower_box];u=np.r_[upper,upper_box]
    solver=osqp.OSQP();solver.setup(P=H,q=linear,A=mat,l=lower,u=u,verbose=False,eps_abs=1e-9,eps_rel=1e-9,max_iter=30000,polishing=True)
    solver.warm_start(x=x);solution=solver.solve(raise_error=False)
    if solution.x is None:raise RuntimeError(solution.info.status)
    x=solution.x.copy();position=control[:,:3].copy();position[free]=x.reshape(-1,3)
    violations=[];counts=[]
    for group in range(2):
        values=np.einsum('ij,jk->ik',operators[group],position,optimize=False)
        excess=np.linalg.norm(values,axis=1)-limits[group]
        bad=np.flatnonzero(excess>2e-8);counts.append(len(bad));violations.append(float(excess.max()))
        for i in bad:add_cut(group,int(i),values[i])
    history.append(dict(iteration=iteration,osqp_status=solution.info.status,qp_iterations=int(solution.info.iter),primal_residual=float(solution.info.prim_res),dual_residual=float(solution.info.dual_res),maximum_excess_speed_and_acceleration=violations,new_cuts=counts))
    print(json.dumps(history[-1]),flush=True)
    if solution.info.status=='solved' and not any(counts):converged=True;break
q=control.copy();q[free,:3]=position[free];q[free,3:7]=previous[free,3:7]
assert np.array_equal(q[fixed],control[fixed]);assert np.array_equal(q[:,7:],control[:,7:])
commands=targets['robot_qpos'].copy();commands[:,1]=q
transforms=np.tile(np.eye(4),(n,1,1));transforms[:,:3,:3]=Rotation.from_quat(q[:,3:7]).as_matrix();transforms[:,:3,3]=q[:,:3]
motion=motion_summary(transforms,t);over=[k for k,v in motion.items() if v>caps[k]+1e-9]
kin=URDFKinematics(P/'robot/urdfs/g1/dex3_right.urdf');stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz') as z:K=z['intrinsics'].copy()
image_rows=[];hashes={}
for frame in [40,45,50,55,67]:
    path=S/f'e037-watcloud-mask-refinement-handoff-v1/image-observations/right/{frame:06d}.json';d=json.loads(path.read_text());points=np.asarray(d['keypoints']);ids=np.array([4,8,12]);select=points[ids,2]>0
    assert not d['directly_observed'] and not d['metric_valid'];errors=[]
    for a in [control,desired,q]:
        poses=kin.poses(a[frame,:7],dict(zip(m['joint_names']['right'],a[frame,7:],strict=True)))
        tips=np.stack([poses['right_'+digit+'_tip'][:3,3] for digit in ['thumb','index','middle']])[select]
        world=np.einsum('ni,ij->nj',tips,stage,optimize=False);camera=source['camera_c2w'][frame]
        C=np.einsum('ni,ij->nj',world-camera[:3,3],camera[:3,:3],optimize=False);assert (C[:,2]>.01).all()
        uv=np.c_[K[0,0]*C[:,0]/C[:,2]+K[0,2],K[1,1]*C[:,1]/C[:,2]+K[1,2]]
        errors.append(np.linalg.norm(uv-points[ids[select],:2],axis=1))
    image_rows.append(dict(frame=frame,landmark_ids=ids[select].tolist(),control_errors_px=errors[0].tolist(),raw_object_relative_errors_px=errors[1].tolist(),constrained_errors_px=errors[2].tolist(),control_median_px=float(np.median(errors[0])),constrained_median_px=float(np.median(errors[2])),all_points_nonregressing_vs_control=bool(np.all(errors[2]<=errors[0]+1e-6)),direct=False,metric=False))
    hashes[str(path.relative_to(S))]=sha(path)
np.savez_compressed(O/'inferred-robot-commands.npz',robot_qpos=commands,inferred_command_mask=np.any(q!=control,axis=1),frame_indices=targets['frame_indices'],timestamps_s=t,original_source_valid=source['hand_valid'],original_source_directly_observed=source['hand_directly_observed'],original_source_metric_valid=source['hand_metric_valid'])
report=dict(schema='v2d.motion-constrained-inferred-wrist.v2',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(P/'manifest.json'),raw_object_relative_commands_sha256=sha(desired_path),reused_angular_fit_commands_sha256=sha(previous_path),retained_failed_control_report_sha256=sha(S/'e037-motion-constrained-wrist-v1/report.json'),source_image_sha256=hashes,
    solver='OSQP outer approximation with final exact Euclidean checks',osqp_version=osqp.__version__,translation_solver_converged=converged,solver_history=history,original_caps=caps,motion=motion,over_original_caps=over,motion_passed=not over,
    modified_interval_half_open=[25,100],all_source_validity_flags_unchanged=True,direct_and_metric_commands_unchanged=True,left_hand_and_fingers_unchanged=True,image_rows=image_rows,inferred_image_checks_passed=all(r['all_points_nonregressing_vs_control'] for r in image_rows),maximum_translation_change_m=float(np.linalg.norm(q[:,:3]-control[:,:3],axis=1).max()),native_contact_certificate_invalidated=True,native_and_swept_checks_run=False,training_approved=False,physical_task_accepted=False,new_GPU_work=False,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
for name,h in m['files'].items():assert sha(P/name)==h
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
