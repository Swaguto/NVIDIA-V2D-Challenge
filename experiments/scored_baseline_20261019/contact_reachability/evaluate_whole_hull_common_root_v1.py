"""NEW fixed-facet common-root LP and separate actual native effort audit."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.monotonic();O=S/'e037-whole-hull-common-root-native-v1';plan_path=S/'compute/whole-hull-common-root-native-CPU-plan-v1.json';plan=json.loads(plan_path.read_text())
from v2d_dex3_flashchord.contact_translation import common_translation_contact_seed
from v2d_dex3_flashchord.native_mesh import compiled_mesh_vertices
setup=S/'compute/solve_compiled_thumb_index_mode_v2.py';old=json.loads((S/'e037-compiled-thumb-index-mode-v2/report.json').read_text());assert sha(setup)==old['producer_sha256'];prefix=setup.read_text().split("for proposal in bank['contact_mode_hypotheses']:")[0];assert 'for iteration in range' not in prefix;ns={'__file__':str(setup)};exec(compile(prefix,str(setup),'exec'),ns)
np=ns['np'];mj=ns['mujoco'];model=ns['model'];data=ns['data'];bn=ns['bn'];ob=ns['ob'];kin=ns['kin'];q0=ns['commands'][1].copy();names=ns['joint_names'];atlas_path=S/'e037-compiled-contact-reachability-atlas-v1/report.json';atlas=json.loads(atlas_path.read_text());queue_path=S/'e037-compiled-contact-reachability-atlas-v1/fixed-facet-pair-queue.json';assert sha(queue_path)==atlas['queue_sha256'];assert all(sha(Path(p))==h for p,h in atlas['inputs_sha256'].items());queue=json.loads(queue_path.read_text());assert len(queue)==atlas['queued_facet_pair_trials']<=64;finger_dofs=ns['finger_dofs'];caps=ns['effort_bounds'];T=ns['T'];patches=ns['patches'];loads=ns['loads'];com=ns['com'];wrist=ns['wrist'];root_dofs=list(model.jnt_dofadr[wrist['right'][:3]])+list(range(int(model.jnt_dofadr[wrist['right'][3]]),int(model.jnt_dofadr[wrist['right'][3]])+3))
image_path=S/'e037-watcloud-mask-refinement-handoff-v1/image-observations/right/000050.json';image=json.loads(image_path.read_text());kp=np.asarray(image['keypoints']);ids=np.array([4,8,12]);selected=kp[ids,2]>0
with np.load(ns['f']/'aligned-source-reference.npz') as z:camera=z['camera_c2w'][50].copy()
stage=np.asarray(json.loads((ns['f']/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz') as z:K=z['intrinsics'].copy()
def image_errors(q):
 poses=kin.poses(q[:7],dict(zip(names,q[7:],strict=True)));p=np.stack([poses[f'right_{d}_tip'][:3,3] for d in ['thumb','index','middle']]);world=np.einsum('ni,ij->nj',p,stage,optimize=False);C=np.einsum('ni,ij->nj',world-camera[:3,3],camera[:3,:3],optimize=False)
 if not (C[:,2]>.01).all():return None
 uv=np.c_[K[0,0]*C[:,0]/C[:,2]+K[0,2],K[1,1]*C[:,1]/C[:,2]+K[1,2]];return np.linalg.norm(uv[selected]-kp[ids[selected],:2],axis=1).tolist()
before=image_errors(q0);rows=[];commands=[]
object_geoms=np.flatnonzero(model.geom_bodyid==ob)
def world_vertices(g):
 mid=int(model.geom_dataid[g]);v=compiled_mesh_vertices(model,mid,mujoco_version=mj.__version__)
 return np.einsum('ni,ji->nj',v,data.geom_xmat[g].reshape(3,3),optimize=False)+data.geom_xpos[g]
def whole_hull_planes(facets):
 q=q0.copy()
 for r in facets:q[7+np.array(r['active_joint_indices'])]=r['active_joint_values_rad']
 ns['set_state'](q);A=[];B=[];records=[];hulls={g:world_vertices(g) for g in ns['right_geoms']+object_geoms.tolist()};self_pen=max([max(0,-float(c.dist)) for c in data.contact[:data.ncon] if bn[int(model.geom_bodyid[c.geom1])].startswith('dex3_right_') and bn[int(model.geom_bodyid[c.geom2])].startswith('dex3_right_')],default=0.)
 for g in ns['right_geoms']:
  for og in object_geoms:
   og=int(og);segment=np.zeros(6);gap=float(mj.mj_geomDistance(model,data,g,og,1.,segment))
   if gap>np.sqrt(3)*.008+plan['whole_hull_support_penetration_cap_m']:
    records.append(dict(hand_geom=g,object_geom=og,signed_gap_at_root_center_m=gap,certificate='native-distance Lipschitz bound exceeds full translation box'));continue
   n=segment[3:]-segment[:3]
   if abs(gap)>1e-12:n/=gap
   if np.linalg.norm(n)<1e-12:n=data.geom_xpos[og]-data.geom_xpos[g]
   if np.linalg.norm(n)<1e-12:n=np.array([1.,0,0])
   n/=np.linalg.norm(n);support=float(np.einsum('ni,i->n',hulls[og],n,optimize=False).min()-np.einsum('ni,i->n',hulls[g],n,optimize=False).max());bound=support+plan['whole_hull_support_penetration_cap_m'];A.append(n);B.append(bound);records.append(dict(hand_geom=g,object_geom=og,signed_gap_at_root_center_m=gap,certificate='fixed compiled-support separating plane',normal_hand_to_object_world=n.tolist(),support_gap_m=support,translation_upper_m=bound))
 return np.array(A).reshape(-1,3),np.asarray(B),records,self_pen

for index,proposal in enumerate(queue):
 if time.monotonic()-start>220:raise TimeoutError('Shared-root candidate cap')
 a,b=proposal['digits'];facets=[a,b];plane_A,plane_b,plane_records,fixed_self_penetration=whole_hull_planes(facets);seed=common_translation_contact_seed([r['hand_triangle_world_m'] for r in facets],[r['hand_outward_normal_world'] for r in facets],[r['object_triangle_world_m'] for r in facets],[r['object_outward_normal_world'] for r in facets],np.array([[-.008,.008]]*3),signed_gap_m=plan['target_signed_gap_m'],translation_halfspaces=(plane_A,plane_b));row=dict(whole_hull_planes=plane_records,fixed_self_penetration_m=fixed_self_penetration,candidate_id=index,mode=proposal['mode'],queue_rank_score=proposal['rank_score'],atlas_digit_samples=[r['grid_sample'] for r in facets],hypothesized_patches={r['digit']:r['patch'] for r in facets},geometric_LP=seed,physical_task_accepted=False,training_approved=False)
 if not seed['geometrically_feasible']:rows.append(row);continue
 if fixed_self_penetration>.001:row['rejected_fixed_configuration_self_penetration']=True;rows.append(row);continue
 q=q0.copy();q[:3]+=np.asarray(seed['translation_m'])
 for r in facets:q[7+np.array(r['active_joint_indices'])]=r['active_joint_values_rad']
 assert (abs(q[:3]-q0[:3])<=.008+1e-8).all();assert np.allclose(q[3:7],q0[3:7]);assert all(kin.limits[n][0]-1e-9<=v<=kin.limits[n][1]+1e-9 for n,v in zip(names,q[7:],strict=True));ns['set_state'](q);mj.mj_fwdVelocity(model,data);native=[];geometries=[];maps=[];root_maps=[];actual_digits=set();unmatched=[];all_contacts=[]
 for c in data.contact[:data.ncon]:
  i,j=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2]);all_contacts.append(dict(bodies=[bn[i],bn[j]],signed_gap_m=float(c.dist)))
  if ob not in [i,j]:continue
  other=j if i==ob else i
  if not bn[other].startswith('dex3_right_'):continue
  digit=ns['digit_of'](bn[other]);binding=row['hypothesized_patches'].get(digit);distance=float(patches[binding].closest_world(np.asarray(c.pos)[None],T).distances_m[0]) if binding else None;match=binding is not None and distance<=.001
  if match:actual_digits.add(digit)
  else:unmatched.append(len(native))
  geometry=ns['ContactGeometry'].from_mujoco_geometry(point_world_m=np.asarray(c.pos),frame_geom1_to_geom2_world=np.asarray(c.frame).reshape(3,3),dimension=int(c.dim),friction=np.asarray(c.friction),object_is_geom2=bool(j==ob));jp=np.zeros((3,model.nv));jr=jp.copy();mj.mj_jac(model,data,jp,jr,np.asarray(c.pos),other);geometries.append(geometry);maps.append(ns['contact_effort_map'](geometry,jp[:,finger_dofs],jr[:,finger_dofs]));root_maps.append(ns['contact_effort_map'](geometry,jp[:,root_dofs],jr[:,root_dofs]));native.append(dict(native_geom_ids=[int(c.geom1),int(c.geom2)],native_bodies=[bn[i],bn[j]],point_world_m=np.asarray(c.pos).tolist(),frame_into_object_world=geometry.frame_into_object_world.tolist(),dimension=int(c.dim),friction=np.asarray(c.friction).tolist(),signed_gap_m=float(c.dist),digit=digit,patch=binding,distance_to_patch_m=distance,mode_match=match,included_in_effort_load_check=True))
 free=data.qfrc_bias[finger_dofs]-data.qfrc_passive[finger_dofs];root_free=data.qfrc_bias[root_dofs]-data.qfrc_passive[root_dofs];checks={name:ns['outer_actuated_contact_load_feasibility'](geometries,com,wrench,effort_maps=maps,free_effort_Nm=free,effort_bounds_Nm=caps) for name,wrench in loads.items()};root_checks={}
 for name,result in checks.items():
  if not result['feasible_in_outer_model']:continue
  effort=root_free.copy()
  for matrix,components in zip(root_maps,result['solved_contact_components'],strict=True):effort+=np.einsum('ij,j->i',matrix,np.asarray(components),optimize=False)
  root_checks[name]=dict(chosen_translation_effort_N=effort[:3].tolist(),chosen_orientation_effort_Nm=effort[3:].tolist(),within_original_60_component_cap=bool((abs(effort)<=60).all()),chosen_allocation_only_not_global_motor_or_dynamics_certificate=True)
 after=image_errors(q);source=after is not None and bool(np.all(np.asarray(after)<=np.asarray(before)+1e-6));max_pen=max([max(0,-c['signed_gap_m']) for c in all_contacts],default=0.);presence=set(row['hypothesized_patches'])<=actual_digits;opposition=min([float(np.dot(np.asarray(x['frame_into_object_world'])[0],np.asarray(y['frame_into_object_world'])[0])) for x in native for y in native if x['digit']=='thumb' and y['digit'] in set(row['hypothesized_patches'])-{'thumb'}],default=None);native_ok=max_pen<=.001;load_ok=all(r['feasible_in_outer_model'] for r in checks.values());static_ok=bool(presence and not unmatched and opposition is not None and opposition<-.5 and native_ok and load_ok and source and all(r['within_original_60_component_cap'] for r in root_checks.values()))
 row.update(command_index=len(commands),root_translation_m=(q[:3]-q0[:3]).tolist(),all_actual_native_contacts=all_contacts,actual_right_hand_object_contacts=native,required_native_digits_present=presence,unmatched_actual_contact_indices=unmatched,minimum_thumb_other_native_normal_dot=opposition,maximum_single_knot_penetration_m=max_pen,full_single_knot_1mm_native_gate_passed=native_ok,actual_effort_load_checks=checks,chosen_root_allocation_checks=root_checks,inferred_image_ids=ids[selected].tolist(),inferred_image_errors_before_after_px=[before,after],inferred_image_nonregression=source,single_knot_static_prerequisites_passed=static_ok,full_source_motion_native_swept_motor_grasp_certificates_not_established=True);commands.append(q);rows.append(row)
np.savez_compressed(O/'geometric-seed-commands.npz',right_commands=np.array(commands).reshape(-1,len(q0)),original_right_command=q0,frame_index=50)
summary=dict(queued_facet_pairs=len(queue),geometrically_feasible=sum(r['geometric_LP']['geometrically_feasible'] for r in rows),actual_thumb_contact_candidates=sum(any(c['digit']=='thumb' for c in r.get('actual_right_hand_object_contacts',[])) for r in rows),required_native_digits_present=sum(r.get('required_native_digits_present',False) for r in rows),native_single_knot_gate_passed=sum(r.get('full_single_knot_1mm_native_gate_passed',False) for r in rows),both_actual_loads_outer_feasible=sum(all(c['feasible_in_outer_model'] for c in r['actual_effort_load_checks'].values()) for r in rows if 'actual_effort_load_checks' in r),inferred_image_nonregression=sum(r.get('inferred_image_nonregression',False) for r in rows),single_knot_prerequisites_passed=sum(r.get('single_knot_static_prerequisites_passed',False) for r in rows))
report=dict(schema='v2d.whole-hull-common-root-facet-native-validation.v1',all_fixed_facet_pairs_reused_without_new_queue_or_local_sweep=True,whole_hull_separation_planes_are_conservative_not_global=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),inputs_sha256={str(p):sha(p) for p in [setup,atlas_path,queue_path,ns['mp'],ns['f']/'targets.npz',image_path,R/'scripts/dex3_training/v2d_dex3_flashchord/contact_translation.py']},summary=summary,rows=rows,command_file_sha256=sha(O/'geometric-seed-commands.npz'),finite_atlas_and_top_queue_not_global_reachability=True,source_flags_and_31of32_heldout_failure_unchanged=True,all_native_contacts_retained=True,all_actual_right_object_contacts_in_load_checks=True,no_force_budget_fabricated=True,model_morphology_mass_friction_effort_unchanged=True,no_prior_optimizer_or_source_fit_rerun=True,no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
