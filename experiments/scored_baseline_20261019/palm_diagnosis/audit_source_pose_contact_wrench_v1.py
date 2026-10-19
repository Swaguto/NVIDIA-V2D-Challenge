"""Optimistic native-contact load feasibility without invented force budgets.

An outer friction box contains the native elliptical/pyramidal friction cone.
Infeasibility rejects that frozen contact set in this static model. Feasibility
does not prove actual force production, motor feasibility, dynamics or success.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
import mujoco,numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from audit_hand_reference_contacts import canonical_id
O=S/'e037-source-pose-contact-wrench-v1';O.mkdir(exist_ok=True);P=S/'e037-motion-constrained-wrist-native-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.monotonic()
def solve(contacts,com,load):
    if not contacts:return dict(feasible=False,reason='No native hand/object contacts',contacts=0)
    count=sum(c['dim'] for c in contacts);W=np.zeros((6,count));normal_ids=[];inequalities=[];offset=0;variable_bounds=[]
    for c in contacts:
        dim=c['dim'];assert dim in [1,3,4,6]
        basis=np.asarray(c['frame']).reshape(3,3);point=np.asarray(c['point']);mu=np.asarray(c['friction'])
        normal_ids.append(offset)
        nf=min(dim,3);W[:3,offset:offset+nf]=basis[:nf].T
        W[3:,offset:offset+nf]=np.cross(point-com,basis[:nf]).T
        if dim>=4:W[3:,offset+3]=basis[0]
        if dim==6:W[3:,offset+4:offset+6]=basis[1:].T
        variable_bounds.append((0,None))
        for component in range(1,dim):
            variable_bounds.append((None,None));coefficient=mu[component-1]
            for sign in [-1.,1.]:
                row=np.zeros(count);row[offset+component]=sign;row[offset]=-coefficient;inequalities.append(row)
        offset+=dim
    objective=np.zeros(count);objective[normal_ids]=1
    scale=np.r_[np.ones(3),np.full(3,10.)]
    fit=linprog(objective,A_ub=np.asarray(inequalities) if inequalities else None,b_ub=np.zeros(len(inequalities)) if inequalities else None,A_eq=W*scale[:,None],b_eq=np.asarray(load)*scale,bounds=variable_bounds,method='highs')
    if fit.status==2:return dict(feasible=False,reason=fit.message,contacts=len(contacts),LP_status=int(fit.status))
    if not fit.success:raise RuntimeError(fit.message)
    residual=np.einsum('ij,j->i',W,fit.x,optimize=False)-load
    assert np.max(abs(residual))<1e-6
    return dict(feasible=True,reason=fit.message,contacts=len(contacts),LP_status=int(fit.status),minimum_total_normal_force_in_outer_model_N=float(fit.fun),normal_forces_in_outer_model_N=fit.x[normal_ids].tolist(),maximum_wrench_residual=float(abs(residual).max()),forces_are_solved_not_measured=True)
def contact(point,normal,mu=1.):
    n=np.asarray(normal,float);helper=np.eye(3)[np.argmin(abs(n))];t=np.cross(n,helper);t/=np.linalg.norm(t);b=np.cross(n,t)
    return dict(point=point,frame=np.stack([n,t,b]).tolist(),dim=3,friction=[mu,mu,0,0,0])
opposed=[contact([.1,0,0],[-1,0,0]),contact([-.1,0,0],[1,0,0])]
controls=dict(opposed_pair_supports_upward_load=solve(opposed,np.zeros(3),[0,0,1,0,0,0]),single_side_cannot_balance_pure_vertical_load=solve([opposed[0]],np.zeros(3),[0,0,1,0,0,0]),opposed_point_pair_cannot_make_axial_torque=solve(opposed,np.zeros(3),[0,0,0,1,0,0]))
assert controls['opposed_pair_supports_upward_load']['feasible']
assert not controls['single_side_cannot_balance_pure_vertical_load']['feasible']
assert not controls['opposed_point_pair_cannot_make_axial_torque']['feasible']
manifest=json.loads((P/'frozen/manifest.json').read_text())
for name,h in manifest['files'].items():assert sha(P/'frozen'/name)==h
with np.load(P/'frozen/targets.npz') as z:commands=z['robot_qpos'].copy();poses=z['object_poses'][:,0].copy();times=z['timestamps_s'].copy()
model_path=P/'native-model.mjb';model=mujoco.MjModel.from_binary_path(str(model_path));data=mujoco.MjData(model)
bodies=[model.body(i).name for i in range(model.nbody)];joints=[model.joint(i).name for i in range(model.njnt)];bid=canonical_id(bodies,'wooden_spoon_object');oa=int(model.jnt_qposadr[model.body_jntadr[bid]])
binding={}
for side in ['left','right']:
    binding[side]=([int(model.jnt_qposadr[canonical_id(joints,f'{side}_wrist_{a}')]) for a in ['x','y','z','rotvec']],[int(model.jnt_qposadr[canonical_id(joints,n)]) for n in manifest['joint_names'][side]])
mass=float(model.body_mass[bid]);gravity=np.asarray(model.opt.gravity).copy();frame=50
Robj=Rotation.from_quat(poses[:,3:]);coms=poses[:,:3]+Robj.apply(model.body_ipos[bid]);dt0=times[frame]-times[frame-1];dt1=times[frame+1]-times[frame];md=.5*(dt0+dt1)
acc=((coms[frame+1]-coms[frame])/dt1-(coms[frame]-coms[frame-1])/dt0)/md
w0=(Robj[frame]*Robj[frame-1].inv()).as_rotvec()/dt0;w1=(Robj[frame+1]*Robj[frame].inv()).as_rotvec()/dt1;omega=.5*(w0+w1);alpha=(w1-w0)/md
reports=[];hashes={}
for label in ['raw_RGB','multiview_fit_unqualified','hard_skin_fit_failed']:
    path=S/'e037-source-pose-family-opposed-grasp-v2'/label/'single-knot-finger-command.npz';hashes[str(path.relative_to(S))]=sha(path)
    with np.load(path) as z:q=z['right_command'].copy()
    data.qpos[:]=model.qpos0;data.qpos[oa:oa+3]=poses[frame,:3];data.qpos[oa+3:oa+7]=poses[frame,[6,3,4,5]]
    for si,side in enumerate(['left','right']):
        command=q if side=='right' else commands[frame,si];w,f=binding[side]
        data.qpos[w[:3]]=command[:3];data.qpos[w[3]:w[3]+4]=command[[6,3,4,5]];data.qpos[f]=command[7:]
    mujoco.mj_fwdPosition(model,data);contacts=[];excluded=0
    for c in data.contact[:data.ncon]:
        a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
        if bid not in [a,b]:excluded+=1;continue
        other=b if a==bid else a
        if not bodies[other].startswith('dex3_right_'):excluded+=1;continue
        basis=np.asarray(c.frame).reshape(3,3).copy()*(1 if b==bid else -1)
        contacts.append(dict(point=np.asarray(c.pos).tolist(),frame=basis.tolist(),dim=int(c.dim),friction=np.asarray(c.friction).tolist(),native_bodies=[bodies[a],bodies[b]],signed_gap_m=float(c.dist),normal_is_hand_into_object=True))
    np.testing.assert_allclose(data.xipos[bid],coms[frame],atol=1e-10,rtol=0)
    principal=np.asarray(data.ximat[bid]).reshape(3,3);inertia=np.einsum('ij,j,kj->ik',principal,model.body_inertia[bid],principal,optimize=False)
    torque=np.einsum('ij,j->i',inertia,alpha,optimize=False)+np.cross(omega,np.einsum('ij,j->i',inertia,omega,optimize=False))
    static=np.r_[-mass*gravity,np.zeros(3)];motion=np.r_[mass*(acc-gravity),torque]
    reports.append(dict(source_label=label,source_pose_and_selected_best_intermediate_remain_unqualified=True,contacts=contacts,excluded_support_and_other_contact_count=excluded,gravity_load_wrench_world=static.tolist(),estimated_reference_motion_load_wrench_world=motion.tolist(),gravity_outer_friction_feasibility=solve(contacts,coms[frame],static),estimated_motion_outer_friction_feasibility=solve(contacts,coms[frame],motion)))
report=dict(schema='v2d.native-source-pose-optimistic-wrench-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),native_model_sha256=sha(model_path),input_manifest_sha256=sha(P/'frozen/manifest.json'),candidate_sha256=hashes,frame=frame,analytical_controls=controls,all_analytical_controls_passed=True,nominal_native_object_mass_kg=mass,nominal_native_object_inertia_principal_kg_m2=np.asarray(model.body_inertia[bid]).tolist(),gravity_world_m_s2=gravity.tolist(),reference_COM_acceleration_m_s2=acc.tolist(),reference_angular_velocity_world_rad_s=omega.tolist(),reference_angular_acceleration_world_rad_s2=alpha.tolist(),native_model_cone=int(model.opt.cone),rows=reports,friction_model='Outer independent component box using actual per-contact dim and friction, including native torsional/rolling coefficients where present',normal_force_budgets_not_invented=True,normal_force_or_motor_caps_not_applied=True,measured_contact_forces_not_available=True,load_pose_mass_and_friction_not_measured_source_truth=True,source_and_robot_contact_labels_not_modified=True,outer_box_feasible_does_not_certify_exact_friction_or_motor_or_grasp_feasibility=True,outer_box_infeasible_applies_only_to_frozen_native_contact_set_and_nominal_load=True,whole_trajectory_or_new_source_fit_or_physics_steps=False,new_GPU_work=False,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
for name,h in manifest['files'].items():assert sha(P/'frozen'/name)==h
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
