"""NEW dynamic full-finger surface-correspondence solves for declared patch modes.

One bounded geometry solve per mode. No fixed robot triangle, missing middle
contact target, source/MCP recentering, native recompile or physics step. Actual
native manifolds and effort/load checks remain separate from potential patches.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'))
import mujoco,numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from trimesh.triangles import closest_point
from audit_hand_reference_contacts import canonical_id
from v2d_dex3_flashchord.contact_patches import ObjectContactPatch,RobotContactMode
from v2d_dex3_flashchord.contact_load_feasibility import ContactGeometry,contact_effort_map,outer_actuated_contact_load_feasibility
from v2d_dex3_flashchord.kinematics import URDFKinematics
O=S/'e037-native-contact-patch-modes-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();plan_path=S/'compute/native-contact-patch-modes-CPU-plan-v1.json';plan=json.loads(plan_path.read_text());bank_path=S/'e037-object-local-contact-patches-v1/report.json';bank=json.loads(bank_path.read_text());patch_path=S/'e037-object-local-contact-patches-v1/object-local-patches.npz';assert sha(patch_path)==bank['patch_file_sha256'];P=S/'e037-motion-constrained-wrist-native-v1';f=P/'frozen';manifest=json.loads((f/'manifest.json').read_text());mp=P/'native-model.mjb';assert sha(mp)==bank['geometry_sha256'];model=mujoco.MjModel.from_binary_path(str(mp));data=mujoco.MjData(model);bn=[model.body(i).name for i in range(model.nbody)];jn=[model.joint(i).name for i in range(model.njnt)];joint_names=manifest['joint_names']['right'];finger_joints=np.array([canonical_id(jn,n) for n in joint_names]);finger_dofs=model.jnt_dofadr[finger_joints];native_caps=model.jnt_actfrcrange[finger_joints].copy();stiction=model.dof_frictionloss[finger_dofs];effort_bounds=native_caps+np.c_[-stiction,stiction];assert model.jnt_actfrclimited[finger_joints].all() and model.jnt_actgravcomp[finger_joints].all();np.testing.assert_allclose(native_caps,[-.76,.76]*np.ones((7,1)),atol=2e-8,rtol=0)
with np.load(f/'targets.npz') as z:commands=z['robot_qpos'][50].copy();obj=z['object_poses'][50,0].copy()
with np.load(patch_path) as z:
 T=z['object_to_world'].copy();patches={key:ObjectContactPatch(key,bank['object_id'],bank['geometry_sha256'],z[key+'_triangles'],z[key+'_normals']) for key in bank['patches']}
kin=URDFKinematics(f/'robot/urdfs/g1/dex3_right.urdf');wrist={s:[canonical_id(jn,f'{s}_wrist_{a}') for a in ['x','y','z','rotvec']] for s in ['left','right']};ob=canonical_id(bn,'wooden_spoon_object');oa=int(model.jnt_qposadr[model.body_jntadr[ob]]);root_q=commands[1,:7].copy();base=data.qpos.copy();base[:]=model.qpos0;base[oa:oa+3]=obj[:3];base[oa+3:oa+7]=obj[[6,3,4,5]]
for si,side in enumerate(['left','right']):
 q=commands[si];w=wrist[side];base[model.jnt_qposadr[w[:3]]]=q[:3];a=int(model.jnt_qposadr[w[3]]);base[a:a+4]=q[[6,3,4,5]];ids=[canonical_id(jn,n) for n in manifest['joint_names'][side]];base[model.jnt_qposadr[ids]]=q[7:]
right_geoms=[g for g,b in enumerate(model.geom_bodyid) if bn[b].startswith('dex3_right_')];meshes={}
for g in right_geoms:
 assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH;mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);v=np.asarray(model.mesh_vert[a:a+n],float);h=ConvexHull(v);meshes[g]=(v[h.simplices],h.equations[:,:3])
def digit_of(body):
 return next((d for d in ['thumb','index','middle'] if '_hand_'+d+'_' in body),None)
def command(x,active):
 q=commands[1].copy();q[:3]=root_q[:3]+.008*x[:3];q[3:7]=(Rotation.from_rotvec(.08*x[3:6])*Rotation.from_quat(root_q[3:7])).as_quat();q[7+active]=x[6:];return q
def set_state(q):
 data.qpos[:]=base;w=wrist['right'];data.qpos[model.jnt_qposadr[w[:3]]]=q[:3];a=int(model.jnt_qposadr[w[3]]);data.qpos[a:a+4]=q[[6,3,4,5]];data.qpos[model.jnt_qposadr[finger_joints]]=q[7:];data.qvel[:]=0;mujoco.mj_fwdPosition(model,data)
def left_jac(v):
 theta=np.linalg.norm(v);x,y,z=v;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
 return np.eye(3)+(.5 if theta<1e-8 else (1-np.cos(theta))/theta**2)*K+(1/6 if theta<1e-8 else (theta-np.sin(theta))/theta**3)*np.einsum('ij,jk->ik',K,K,optimize=False)
def point_jac(point,body,q,x,active):
 if not bn[body].startswith('dex3_right_'):return np.zeros((3,len(x)))
 jp=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jp,None,point,body);lever=point-q[:3];angular=.08*np.cross(left_jac(.08*x[3:6]).T,lever).T
 return np.c_[.008*np.eye(3),angular,jp[:,finger_dofs[active]]]
def pairs_for_mode(mode):
 pairs=[]
 for digit,key in mode.digit_patches:
  best=None;patch=patches[key]
  for g in right_geoms:
   body=int(model.geom_bodyid[g])
   if digit_of(bn[body])!=digit:continue
   triangles,normals=meshes[g];G=data.geom_xmat[g].reshape(3,3);world=np.einsum('nvi,ji->nvj',triangles,G,optimize=False)+data.geom_xpos[g];nw=np.einsum('ni,ji->nj',normals,G,optimize=False);query=data.geom_xpos[g].copy()
   for _ in range(2):
    hit=patch.closest_world(query[None],T);object_point=hit.points[0];normal=hit.outward_normals[0];eligible=np.flatnonzero(np.einsum('ni,i->n',nw,normal,optimize=False)<=-.5)
    if not len(eligible):break
    nearest=closest_point(world[eligible],np.broadcast_to(object_point,(len(eligible),3)));j=int(np.argmin(np.linalg.norm(nearest-object_point,axis=1)));hand_point=nearest[j];query=hand_point
   else:
    error=object_point+plan['target_signed_gap_m']*normal-hand_point;cost=float(np.dot(error,error));candidate=dict(digit=digit,patch=key,geom=g,body=body,hand_point=hand_point,object_point=object_point,normal=normal,error=error,triangle=int(eligible[j]))
    if best is None or cost<best[0]:best=(cost,candidate)
  if best is None:raise RuntimeError('No opposed full-finger surface for '+digit)
  pairs.append(best[1])
 return pairs
def evaluate(x,active,mode,with_jac=True):
 q=command(x,active);set_state(q);pairs=pairs_for_mode(mode);residual=[];matrix=[]
 for p in pairs:
  residual.extend(p['error']/.01)
  if with_jac:matrix.extend(point_jac(p['hand_point'],p['body'],q,x,active)/.01)
 max_pen=0.;contacts=[]
 for c in data.contact[:data.ncon]:
  a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
  if not (bn[a].startswith('dex3_right_') or bn[b].startswith('dex3_right_')):continue
  max_pen=max(max_pen,max(0,-float(c.dist)));contacts.append((a,b,float(c.dist)))
  if c.dist<-.0005:
   normal=np.asarray(c.frame[:3]);residual.append(10*(-.0005-float(c.dist))/.01)
   if with_jac:
    pa=np.asarray(c.pos)-.5*c.dist*normal;pb=np.asarray(c.pos)+.5*c.dist*normal;ja=point_jac(pa,a,q,x,active);jb=point_jac(pb,b,q,x,active);matrix.append(10*np.einsum('i,ij->j',normal,jb-ja,optimize=False)/.01)
 residual=np.asarray(residual);return dict(q=q,pairs=pairs,residual=residual,A=np.asarray(matrix) if with_jac else None,merit=float(np.dot(residual,residual)),max_penetration_m=max_pen,contacts=contacts)
meta_path=S/'compute/palm-wrench-library-compatibility-inputs-v1.json';meta=json.loads(meta_path.read_text());lr=json.loads(Path(meta['source_report']).read_text());loads={k:lr['rows'][0][field] for k,field in [('gravity','gravity_load_wrench_world'),('estimated_motion','estimated_reference_motion_load_wrench_world')]};com=np.asarray(meta['object_COM_native_world_m']);mode_reports=[];jac_controls=[]
for proposal in bank['contact_mode_hypotheses']:
 if proposal['status']!='hypothesis_ready_for_new_geometry_solve':mode_reports.append(proposal);continue
 mode=RobotContactMode(proposal['mode_id'],'right',bank['object_id'],bank['geometry_sha256'],proposal['digit_patches']);mode.validate_patch_bindings(patches);active=np.array([i for i,n in enumerate(joint_names) if any('_hand_'+d+'_' in n for d in mode.required_digits)]);x=np.r_[np.zeros(6),commands[1,7+active]];lower=np.r_[-np.ones(6),[kin.limits[joint_names[i]][0] for i in active]];upper=np.r_[np.ones(6),[kin.limits[joint_names[i]][1] for i in active]];assert ((x>=lower)&(x<=upper)).all();step_cap=np.r_[np.full(6,.25),np.full(len(active),.15)];history=[];reason='iteration_cap';mode_start=time.monotonic();m=evaluate(x,active,mode)
 # New root-rotation Jacobian control at a concrete surface point, not an old replay.
 p=m['pairs'][0];body=p['body'];point=p['hand_point'].copy();local=np.einsum('i,ij->j',point-data.xpos[body],data.xmat[body].reshape(3,3),optimize=False);J=point_jac(point,body,m['q'],x,active);errors=[]
 for axis in range(3):
  points=[]
  for sign in [-1,1]:
   y=x.copy();y[3+axis]+=sign*1e-5;set_state(command(y,active));points.append(np.einsum('ij,j->i',data.xmat[body].reshape(3,3),local,optimize=False)+data.xpos[body])
  fd=(points[1]-points[0])/2e-5;errors.append(float(abs(fd-J[:,3+axis]).max()))
 assert max(errors)<2e-6,errors;jac_controls.append(dict(mode=mode.mode_id,root_rotation_point_jacobian_max_error_m_per_normalized_unit=max(errors)));m=evaluate(x,active,mode)
 for iteration in range(plan['iteration_cap']):
  if time.monotonic()-start>220 or time.monotonic()-mode_start>plan['mode_wall_cap_seconds']:reason='wall_cap';break
  errors=[float(np.linalg.norm(p['error'])) for p in m['pairs']]
  if max(errors)<=plan['surface_target_error_m'] and m['max_penetration_m']<=.001:reason='geometric_target_reached';break
  A=np.r_[m['A'],1e-3*np.eye(len(x))];rhs=np.r_[m['residual'],np.zeros(len(x))];lb=np.maximum(lower-x,-step_cap);ub=np.minimum(upper-x,step_cap);free=ub-lb>1e-12
  fit=lsq_linear(A[:,free],rhs,bounds=(lb[free],ub[free]),method='bvls',tol=1e-10,max_iter=100)
  if not fit.success:reason='linear_subproblem_unresolved: '+fit.message;break
  delta=np.zeros(len(x));delta[free]=fit.x;accepted=None
  for scale in [1.,.5,.25,.125]:
   y=x+scale*delta;assert ((y>=lower-1e-10)&(y<=upper+1e-10)).all();trial=evaluate(y,active,mode,with_jac=False)
   if trial['merit']<m['merit']-1e-8 and trial['max_penetration_m']<=.001:accepted=(y,trial);break
  history.append(dict(iteration=iteration+1,merit=m['merit'],target_errors_m=errors,maximum_penetration_m=m['max_penetration_m'],step_accepted=accepted is not None))
  if accepted is None:reason='no_acceptable_bounded_geometry_step';break
  x=accepted[0];m=evaluate(x,active,mode)
 m=evaluate(x,active,mode);mujoco.mj_fwdVelocity(model,data);contacts=[];geometries=[];maps=[];full_maps=[];actual_digits=set();unexpected=[];root_dofs=list(model.jnt_dofadr[wrist['right'][:3]])+list(range(int(model.jnt_dofadr[wrist['right'][3]]),int(model.jnt_dofadr[wrist['right'][3]])+3));free_effort=data.qfrc_bias[finger_dofs]-data.qfrc_passive[finger_dofs];root_free=data.qfrc_bias[root_dofs]-data.qfrc_passive[root_dofs]
 for c in data.contact[:data.ncon]:
  a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
  if ob not in [a,b]:continue
  other=b if a==ob else a
  if not bn[other].startswith('dex3_right_'):continue
  digit=digit_of(bn[other]);binding=dict(mode.digit_patches).get(digit);distance=float(patches[binding].closest_world(np.asarray(c.pos)[None],T).distances_m[0]) if binding else None;matched=binding is not None and distance<=.001
  if matched:actual_digits.add(digit)
  else:unexpected.append(len(contacts))
  geometry=ContactGeometry.from_mujoco_geometry(point_world_m=np.asarray(c.pos),frame_geom1_to_geom2_world=np.asarray(c.frame).reshape(3,3),dimension=int(c.dim),friction=np.asarray(c.friction),object_is_geom2=bool(b==ob));jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jac(model,data,jp,jr,np.asarray(c.pos),other);geometries.append(geometry);maps.append(contact_effort_map(geometry,jp[:,finger_dofs],jr[:,finger_dofs]));full_maps.append(contact_effort_map(geometry,jp[:,root_dofs],jr[:,root_dofs]));contacts.append(dict(native_bodies=[bn[a],bn[b]],point=np.asarray(c.pos).tolist(),frame=geometry.frame_into_object_world.tolist(),dimension=int(c.dim),friction=np.asarray(c.friction).tolist(),signed_gap_m=float(c.dist),actual_digit=digit,assigned_patch=binding,distance_to_assigned_patch_m=distance,mode_match=matched,included_in_load_check=True))
 checks={name:outer_actuated_contact_load_feasibility(geometries,com,wrench,effort_maps=maps,free_effort_Nm=free_effort,effort_bounds_Nm=effort_bounds) for name,wrench in loads.items()};root_checks={}
 for name,result in checks.items():
  if result['feasible_in_outer_model']:
   effort=root_free.copy()
   for matrix,components in zip(full_maps,result['solved_contact_components'],strict=True):effort+=np.einsum('ij,j->i',matrix,np.asarray(components),optimize=False)
   root_checks[name]=dict(chosen_translation_effort_N=effort[:3].tolist(),chosen_orientation_effort_Nm=effort[3:].tolist(),within_unchanged_60_component_cap=bool((abs(effort)<=60).all()),single_solved_allocation_only_not_dynamic_certificate=True)
 image_path=S/'e037-watcloud-mask-refinement-handoff-v1/image-observations/right/000050.json';image=json.loads(image_path.read_text());kp=np.asarray(image['keypoints']);ids=np.array([4,8,12]);selected=kp[ids,2]>0
 with np.load(f/'aligned-source-reference.npz') as z:camera=z['camera_c2w'][50].copy()
 stage=np.asarray(json.loads((f/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
 with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz') as z:K=z['intrinsics'].copy()
 image_errors=[]
 for q in [commands[1],m['q']]:
  poses=kin.poses(q[:7],dict(zip(joint_names,q[7:],strict=True)));points=np.stack([poses[f'right_{d}_tip'][:3,3] for d in ['thumb','index','middle']]);world=np.einsum('ni,ij->nj',points,stage,optimize=False);C=np.einsum('ni,ij->nj',world-camera[:3,3],camera[:3,:3],optimize=False);assert (C[:,2]>.01).all();uv=np.c_[K[0,0]*C[:,0]/C[:,2]+K[0,2],K[1,1]*C[:,1]/C[:,2]+K[1,2]];image_errors.append(np.linalg.norm(uv[selected]-kp[ids[selected],:2],axis=1).tolist())
 normals=[np.asarray(c['frame'])[0] for c in contacts];opposition=min([float(np.dot(normals[i],normals[j])) for i,a in enumerate(contacts) for j,b in enumerate(contacts) if a['actual_digit']=='thumb' and b['actual_digit'] in set(mode.required_digits)-{'thumb'}],default=None);presence=set(mode.required_digits)<=actual_digits;loads_ok=all(r['feasible_in_outer_model'] for r in checks.values());source_ok=bool(np.all(np.asarray(image_errors[1])<=np.asarray(image_errors[0])+1e-6));native_ok=m['max_penetration_m']<=.001;static_gate=bool(presence and not unexpected and opposition is not None and opposition<-.5 and loads_ok and native_ok and source_ok and all(r['within_unchanged_60_component_cap'] for r in root_checks.values()))
 folder=O/mode.mode_id;folder.mkdir();np.savez_compressed(folder/'single-knot-patch-command.npz',frame_index=50,right_command=m['q'],original_right_command=commands[1],returned_geometry_iterate=x);mode_report=dict(mode_id=mode.mode_id,required_digits=mode.required_digits,digit_patches=mode.digit_patches,reason=reason,iterations=len(history),history=history,final_merit=m['merit'],final_surface_target_errors_m=[float(np.linalg.norm(p['error'])) for p in m['pairs']],maximum_single_knot_penetration_m=m['max_penetration_m'],native_contacts=contacts,required_actual_digits_present=presence,unmatched_actual_contact_indices=unexpected,minimum_thumb_other_native_normal_dot=opposition,actual_contact_effort_load_checks=checks,chosen_allocation_root_effort_checks=root_checks,inferred_image_tip_ids=ids[selected].tolist(),inferred_image_errors_before_after_px=image_errors,inferred_image_nonregression=source_ok,single_knot_static_and_source_prerequisites_passed=static_gate,geometry_stop_not_stationarity_or_physical_success=True,source_metric_hand_and_full_trajectory_not_qualified=True,training_approved=False,physical_task_accepted=False,mode_wall_seconds=time.monotonic()-mode_start);(folder/'report.json').write_text(json.dumps(mode_report,indent=2)+'\n');mode_reports.append(mode_report);print(json.dumps({k:mode_report[k] for k in ['mode_id','reason','iterations','final_surface_target_errors_m','maximum_single_knot_penetration_m','required_actual_digits_present','minimum_thumb_other_native_normal_dot','inferred_image_nonregression','single_knot_static_and_source_prerequisites_passed']},indent=2),flush=True)
report=dict(schema='v2d.native-object-contact-patch-mode-study.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),inputs_sha256={str(p):sha(p) for p in [bank_path,patch_path,mp,f/'manifest.json',f/'targets.npz',meta_path]},root_anchor='original wristv3, no source/MCP recenter',root_translation_component_bounds_m=.008,root_rotation_component_bounds_rad=.08,root_rotation_jacobian_controls=jac_controls,mode_reports=mode_reports,source_flags_and_31of32_heldout_failure_unchanged=True,native_model_friction_condim_mass_inertia_effort_unchanged=True,all_native_object_contacts_retained_and_included_in_load_checks=True,mode_is_hypothesis_not_observed_source_contact=True,required_full_native_reference_contact_gate_m=.001,whole_source_motion_native_swept_actual_motor_grasp_still_required=True,new_native_compile_or_physics_steps=False,new_GPU_work=False,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(completed_modes=len(mode_reports),wall_seconds=report['wall_seconds']),indent=2),flush=True)
