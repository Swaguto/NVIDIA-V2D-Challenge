"""NEW coupled surface/force/effort pose seed under unchanged source residual boxes.

Only the saved multiview positive control passes the new exploratory skin
association prerequisite. It is NOT accepted source truth. Other source cases
remain rejected. One SLSQP solve; no old source/pose/gain job repeated.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'))
import mujoco,numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from trimesh.triangles import closest_point
from v2d_dex3_flashchord.kinematics import URDFKinematics
from v2d_dex3_flashchord.contact_load_feasibility import ContactGeometry,contact_effort_map,outer_actuated_contact_load_feasibility
from audit_hand_reference_contacts import canonical_id
O=S/'e037-skin-bound-actuated-pose-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan_path=S/'compute/skin-bound-actuated-pose-CPU-plan-v1.json';plan=json.loads(plan_path.read_text());elig_path=S/'e037-skin-seed-independent-closure-v1/report.json';elig=json.loads(elig_path.read_text());case_index=1;assert elig['rows'][case_index]['all_three_cached_sites_source_skin_eligible'];assert not elig['rows'][0]['all_three_cached_sites_source_skin_eligible'] and not elig['rows'][2]['all_three_cached_sites_source_skin_eligible']
control_path=S/'e037-source-patch-wrench-sites-v1/report.json';control=json.loads(control_path.read_text());seed=control['rows'][case_index]['proposed_site_seed'];meta_path=S/'compute/palm-wrench-library-compatibility-inputs-v1.json';meta=json.loads(meta_path.read_text());loads_report=json.loads(Path(meta['source_report']).read_text());loads=[np.asarray(loads_report['rows'][0][key]) for key in ['gravity_load_wrench_world','estimated_reference_motion_load_wrench_world']];load_names=['gravity','estimated_motion'];com=np.asarray(meta['object_COM_native_world_m'])
P=S/'e037-motion-constrained-wrist-native-v1';f=P/'frozen';manifest=json.loads((f/'manifest.json').read_text());urdf=f/'robot/urdfs/g1/dex3_right.urdf';kin=URDFKinematics(urdf);xml=ET.parse(urdf).getroot();joint_nodes={j.attrib['name']:j for j in xml.findall('joint')};joint_names=manifest['joint_names']['right'];digits=['thumb','index','middle'];bodies=[f'right_hand_{d}_{2 if d=="thumb" else 1}_link' for d in digits]
parent_joint={j.find('child').attrib['link']:j for j in joint_nodes.values()};ancestors={}
for link in [l.attrib['name'] for l in xml.findall('link')]:
 chain=[];v=link
 while v in parent_joint:
  j=parent_joint[v];chain.append(j.attrib['name']);v=j.find('parent').attrib['link']
 ancestors[link]=chain
inertias=[]
for link in xml.findall('link'):
 inertia=link.find('inertial')
 if inertia is None:continue
 mass=float(inertia.find('mass').attrib['value']);origin=inertia.find('origin');xyz=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3);inertias.append((link.attrib['name'],mass,xyz))
mp=P/'native-model.mjb';assert sha(mp)==meta['native_model_sha256'];model=mujoco.MjModel.from_binary_path(str(mp));native_joints=[model.joint(i).name for i in range(model.njnt)];native_bodies=[model.body(i).name for i in range(model.nbody)];joint_ids=np.array([canonical_id(native_joints,n) for n in joint_names]);dofs=model.jnt_dofadr[joint_ids];caps=model.jnt_actfrcrange[joint_ids].copy();stiction=model.dof_frictionloss[dofs].copy();assert model.jnt_actfrclimited[joint_ids].all() and model.jnt_actgravcomp[joint_ids].all();assert np.all(model.jnt_stiffness[joint_ids]==0);np.testing.assert_allclose(caps,[-.76,.76]*np.ones((7,1)),atol=2e-8,rtol=0);effort_bounds=caps+np.c_[-stiction,stiction];gravity=np.asarray(model.opt.gravity)
with np.load(f/'targets.npz') as z:commands=z['robot_qpos'][50].copy();object_pose=z['object_poses'][50,0].copy()
with np.load(S/'e037-skin-associated-wrench-seed-v1/seed-geometry.npz') as z:geometry={k:z[k].copy() for k in z.files}
root_center=np.asarray(seed['inferred_wrist_pose_seed_xyzw']);old_fingers=np.asarray(seed['original_fingers_rad_not_solved']);sites=np.stack([seed['sites'][d]['point'] for d in digits]);object_normals=np.stack([seed['sites'][d]['normal'] for d in digits]);friction=np.asarray(control['native_outer_box_friction_hypothesis']);basis=[]
for n in object_normals:
 n=-n;t=np.cross(n,np.eye(3)[np.argmin(abs(n))]);t/=np.linalg.norm(t);basis.append(np.stack([n,t,np.cross(n,t)]))
basis=np.asarray(basis);W=np.zeros((6,9))
for c in range(3):W[:3,c*3:c*3+3]=basis[c].T;W[3:,c*3:c*3+3]=np.cross(sites[c]-com,basis[c]).T
poses0=kin.poses(root_center,dict(zip(joint_names,old_fingers,strict=True)));robot_tri=[];robot_normal=[];bary0=[];face_ids=[]
for c,(digit,body) in enumerate(zip(digits,bodies,strict=True)):
 tris=geometry[f'robot_{digit}_triangles_body'];normal=geometry[f'robot_{digit}_outward_normals_body'];p=poses0[body];world=np.einsum('nvi,ji->nvj',tris,p[:3,:3],optimize=False)+p[:3,3];nw=np.einsum('ni,ji->nj',normal,p[:3,:3],optimize=False);eligible=np.flatnonzero(np.einsum('ni,i->n',nw,object_normals[c],optimize=False)<=-.5);assert len(eligible)
 nearest=closest_point(world[eligible],np.broadcast_to(sites[c],(len(eligible),3)));chosen=int(np.argmin(np.linalg.norm(nearest-sites[c],axis=1)));k=int(eligible[chosen]);local_point=np.einsum('i,ij->j',nearest[chosen]-p[:3,3],p[:3,:3],optimize=False);coords=np.linalg.lstsq((tris[k,1:]-tris[k,0]).T,local_point-tris[k,0],rcond=None)[0];coords=np.maximum(0,coords);coords/=max(1,coords.sum());robot_tri.append(tris[k]);robot_normal.append(normal[k]);bary0.extend(coords);face_ids.append(k)
robot_tri=np.asarray(robot_tri);robot_normal=np.asarray(robot_normal)
initial_forces=np.array([seed['outer_model_load_results'][name]['solved_contact_components'] for name in load_names]);assert initial_forces.shape==(2,3,3)
for i in range(2):np.testing.assert_allclose(np.einsum('ij,j->i',W,initial_forces[i].ravel(),optimize=False),loads[i],atol=1e-7,rtol=0)
x0=np.r_[np.zeros(6),old_fingers,bary0,initial_forces.ravel()];bounds=[(-1.,1.)]*6+[kin.limits[n] for n in joint_names]+[(0.,1.)]*6+[(0.,None) if k%3==0 else (None,None) for k in range(18)];lower=np.array([-np.inf if a is None else a for a,b in bounds]);upper=np.array([np.inf if b is None else b for a,b in bounds]);assert ((x0>=lower)&(x0<=upper)).all()
cache_x=None;cache_value=None;calls=0;iteration_records=[]
def measure(x,enforce_wall=True):
 global cache_x,cache_value,calls
 x=np.asarray(x);assert np.isfinite(x).all() and (x>=lower-1e-10).all() and (x<=upper+1e-10).all(),'NLP evaluated outside declared box'
 if enforce_wall and time.monotonic()-start>plan['wall_cap_seconds']:raise TimeoutError('New coupled motor/contact solve wall cap')
 if cache_x is not None and np.array_equal(x,cache_x):return cache_value
 calls+=1;q=np.r_[root_center[:3]+.008*x[:3],(Rotation.from_rotvec(.08*x[3:6])*Rotation.from_quat(root_center[3:])).as_quat(),x[6:13]];p=kin.poses(q[:7],dict(zip(joint_names,q[7:],strict=True)));axes=[];origins=[]
 for name in joint_names:
  j=joint_nodes[name];parent=p[j.find('parent').attrib['link']];o=j.find('origin');xyz=np.fromstring(o.attrib.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(o.attrib.get('rpy','0 0 0'),sep=' ');r=Rotation.from_euler('xyz',rpy).as_matrix();axis=np.fromstring(j.find('axis').attrib['xyz'],sep=' ');axes.append(np.einsum('ij,jk,k->i',parent[:3,:3],r,axis,optimize=False));origins.append(np.einsum('ij,j->i',parent[:3,:3],xyz,optimize=False)+parent[:3,3])
 axes=np.asarray(axes);origins=np.asarray(origins);points=[];normals=[];J=[];free=np.zeros(7)
 for c,body in enumerate(bodies):
  a,b=x[13+c*2:15+c*2];local=(1-a-b)*robot_tri[c,0]+a*robot_tri[c,1]+b*robot_tri[c,2];point=np.einsum('ij,j->i',p[body][:3,:3],local,optimize=False)+p[body][:3,3];points.append(point);normals.append(np.einsum('ij,j->i',p[body][:3,:3],robot_normal[c],optimize=False));jac=np.zeros((3,7))
  for j,name in enumerate(joint_names):
   if name in ancestors[body]:jac[:,j]=np.cross(axes[j],point-origins[j])
  J.append(jac)
 for body,mass,local in inertias:
  center=np.einsum('ij,j->i',p[body][:3,:3],local,optimize=False)+p[body][:3,3]
  for j,name in enumerate(joint_names):
   if name in ancestors[body]:free[j]-=np.dot(np.cross(axes[j],center-origins[j]),mass*gravity)
 points=np.asarray(points);normals=np.asarray(normals);J=np.asarray(J);force=x[19:].reshape(2,3,3);effort=[]
 for k in range(2):effort.append(free+sum(np.einsum('ij,jk,k->i',J[c].T,basis[c].T,force[k,c],optimize=False) for c in range(3)))
 cache_x=x.copy();cache_value=dict(q=q,poses=p,points=points,normals=normals,jac=J,free=free,effort=np.asarray(effort),forces=force);return cache_value
def eq(x):
 m=measure(x);return np.r_[(m['points']-sites).ravel()/.01,*[(np.einsum('ij,j->i',W,m['forces'][k].ravel(),optimize=False)-loads[k])*np.r_[np.ones(3),np.full(3,10.)] for k in range(2)]]
def inequalities(x):
 m=measure(x);force=m['forces'];out=[1-x[13:19].reshape(3,2).sum(1),-.5-np.einsum('ni,ni->n',m['normals'],object_normals,optimize=False)]
 for k in range(2):
  out.extend([(effort_bounds[:,1]-m['effort'][k])/.76,(m['effort'][k]-effort_bounds[:,0])/.76])
  for c in range(3):out.extend([(friction[:2]*force[k,c,0]-force[k,c,1:])/10,(friction[:2]*force[k,c,0]+force[k,c,1:])/10])
 return np.concatenate(out)
def objective(x):
 m=measure(x);return float(m['forces'][:,:,0].sum()/100+1e-4*np.sum(x[:6]**2)+1e-5*np.sum((x[6:13]-old_fingers)**2))
def callback(x):
 m=measure(x);iteration_records.append(dict(iteration=len(iteration_records)+1,objective=objective(x),max_eq=float(abs(eq(x)).max()),min_inequality=float(inequalities(x).min())))
initial=measure(x0);initial_points=initial['points'].copy();error=None;fit=None
try:fit=minimize(objective,x0,method='SLSQP',bounds=bounds,constraints=[{'type':'eq','fun':eq},{'type':'ineq','fun':inequalities}],callback=callback,options=dict(maxiter=plan['iteration_cap'],ftol=1e-9))
except Exception as e:error=repr(e)
report=dict(schema='v2d.skin-bound-coupled-actuated-pose.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),inputs_sha256={str(p):sha(p) for p in [control_path,elig_path,mp,f/'manifest.json',f/'targets.npz',urdf,meta_path]},source_label='multiview_fit_unqualified',source_pose_not_selected_as_truth=True,raw_and_failed_skin_cases_remain_ineligible=True,source_and_heldout_flags_unchanged=True,old_native_certificates_not_reused=True,one_new_coupled_surface_force_and_effort_solve=True,root_translation_component_bound_m=.008,root_rotation_component_bound_rad=.08,actual_native_effort_bounds_Nm=caps.tolist(),actual_native_frictionloss_Nm=stiction.tolist(),gravity_compensation_included_inside_effort_limits=True,URDF_effort_ratings_not_substituted_for_runtime_caps=True,selected_robot_distal_triangle_ids=face_ids,initial_site_distances_m=np.linalg.norm(initial_points-sites,axis=1).tolist(),evaluations=calls,iteration_records=iteration_records,numerical_error=error,solver_success=bool(fit.success) if fit is not None else False,solver_message=str(fit.message) if fit is not None else error,source_reference_qualified=False,full_native_1mm_motion_swept_motor_grasp_gates_still_required=True,required_full_native_reference_contact_gate_m=.001,new_GPU_work=False,new_PPO_steps=0,training_approved=False,physical_task_accepted=False)
if fit is not None:
 x=fit.x;m=measure(x,enforce_wall=False);report.update(returned_candidate_only_not_best_intermediate=True,maximum_equality_residual=float(abs(eq(x)).max()),minimum_inequality_residual=float(inequalities(x).min()),actual_site_distances_m=np.linalg.norm(m['points']-sites,axis=1).tolist(),conditional_potential_site_required_joint_effort_Nm=m['effort'].tolist(),analytic_free_gravity_effort_Nm=m['free'].tolist(),candidate_in_declared_bounds=bool(((x>=lower-1e-10)&(x<=upper+1e-10)).all()),surface_force_effort_constraints_passed=bool(abs(eq(x)).max()<1e-6 and inequalities(x).min()>-1e-6))
 np.savez_compressed(O/'single-knot-motor-seed.npz',frame_index=50,right_command=m['q'],original_right_command=commands[1],optimization_returned_x=x,actual_hand_points=m['points'],object_sites=sites,analytic_point_jacobians=m['jac'])
 data=mujoco.MjData(model);data.qpos[:]=model.qpos0;ob=canonical_id(native_bodies,'wooden_spoon_object');oa=int(model.jnt_qposadr[model.body_jntadr[ob]]);data.qpos[oa:oa+3]=object_pose[:3];data.qpos[oa+3:oa+7]=object_pose[[6,3,4,5]]
 for si,side in enumerate(['left','right']):
  q=m['q'] if side=='right' else commands[si];w=[canonical_id(native_joints,f'{side}_wrist_{a}') for a in ['x','y','z','rotvec']];data.qpos[model.jnt_qposadr[w[:3]]]=q[:3];a=int(model.jnt_qposadr[w[3]]);data.qpos[a:a+4]=q[[6,3,4,5]];ids=[canonical_id(native_joints,n) for n in manifest['joint_names'][side]];data.qpos[model.jnt_qposadr[ids]]=q[7:]
 mujoco.mj_fwdPosition(model,data);mujoco.mj_fwdVelocity(model,data);free=data.qfrc_bias[dofs]-data.qfrc_passive[dofs];free_error=float(abs(free-m['free']).max());assert free_error<2e-6,free_error;native_jac_error=0.
 for c,body in enumerate(bodies):
  b=canonical_id(native_bodies,body);jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jac(model,data,jp,jr,m['points'][c],b);native_jac_error=max(native_jac_error,float(abs(jp[:,dofs]-m['jac'][c]).max()))
 assert native_jac_error<2e-6,native_jac_error
 contacts=[];contact_geometries=[];effort_maps=[];all_hand_contacts=[]
 for c in data.contact[:data.ncon]:
  a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2]);na,nb=native_bodies[a],native_bodies[b]
  if na.startswith('dex3_right_') or nb.startswith('dex3_right_'):all_hand_contacts.append(dict(bodies=[na,nb],signed_gap_m=float(c.dist)))
  if ob not in [a,b]:continue
  other=b if a==ob else a
  if not native_bodies[other].startswith('dex3_right_'):continue
  geometry=ContactGeometry.from_mujoco_geometry(point_world_m=np.asarray(c.pos),frame_geom1_to_geom2_world=np.asarray(c.frame).reshape(3,3),dimension=int(c.dim),friction=np.asarray(c.friction),object_is_geom2=bool(b==ob));jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jac(model,data,jp,jr,np.asarray(c.pos),other);contact_geometries.append(geometry);effort_maps.append(contact_effort_map(geometry,jp[:,dofs],jr[:,dofs]));contacts.append(dict(native_bodies=[na,nb],point=np.asarray(c.pos).tolist(),frame=geometry.frame_into_object_world.tolist(),dim=int(c.dim),friction=np.asarray(c.friction).tolist(),signed_gap_m=float(c.dist)))
 native_load_checks={name:outer_actuated_contact_load_feasibility(contact_geometries,com,load,effort_maps=effort_maps,free_effort_Nm=free,effort_bounds_Nm=effort_bounds) for name,load in zip(load_names,loads,strict=True)}
 report.update(new_candidate_native_position_velocity_only_no_steps=True,independent_native_gravity_effort_closure_max_Nm=free_error,independent_native_point_jacobian_closure_max_m_per_rad=native_jac_error,new_actual_native_contacts=contacts,all_new_right_hand_native_contacts=all_hand_contacts,maximum_new_single_knot_penetration_m=max([max(0,-c['signed_gap_m']) for c in all_hand_contacts]+[0]),actual_native_contact_and_effort_load_checks=native_load_checks,passive_joint_stop_and_self_contact_load_assistance_not_certified=True)
report['wall_seconds']=time.monotonic()-start;(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
