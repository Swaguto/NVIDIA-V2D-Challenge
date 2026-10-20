"""ONE NEW 906-variable full-pose factor construction, no assigned episode pose.

Load pinned MJB STATIC metadata only: no MjData, FK, fwdPosition, native gap or
physics calls. Reuse SAVED compiled body vertices/frames. No broadphase omission.
"""
from pathlib import Path
import datetime,hashlib,json,os
import numpy as np,mujoco,trimesh
from scipy import sparse
from scipy.spatial.transform import Rotation
from object_full_pose_angular_v1 import angular_clock_maps
S=Path(__file__).resolve().parents[1];C=S/'compute';O=S/'e037-object-full-pose-factors-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def cv(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return{str(k):cv(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return[cv(v)for v in x]
 return x

def save(n,x):
 p=O/n
 with p.open('w')as f:json.dump(cv(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 p.chmod(0o600);return dict(path=str(p),sha256=sha(p))

def arrays(n,**x):
 p=O/n
 with p.open('wb')as f:np.savez_compressed(f,**x);f.flush();os.fsync(f.fileno())
 p.chmod(0o600);return dict(path=str(p),sha256=sha(p))

def journal(kind,**x):
 with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('a')as f:json.dump(cv(dict(kind=kind,**x)),f,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

plan=read(C/'object-full-pose-factors-CPU-plan-v1.json');journal('BEFORE_NEW_full_pose_construction',plan=plan)
# Relevant immutable chains, not old producer/audit execution.
oldreceipts=['private-integrity-object-authority-v1.json','private-integrity-object-visibility-conic-v1.json','private-integrity-training-source-v1.json','private-integrity-hard-source-filter-v1.json','private-integrity-object-dynamic-v1.json']
registry={}
for name in oldreceipts:
 for k,v in read(C/name)['files'].items():registry[str(S/k)]=v['sha256']if isinstance(v,dict)else v
frozen=S/'e037-thumb-ray-independent-validation-v1/frozen';manifest=read(frozen/'manifest.json');modelpath=S/'e037-motion-constrained-wrist-native-v1/native-model.mjb';gpath=S/'e037-object-visibility-support-model-v1/NEW-compiled-support-geometry.npz';bpath=S/'e037-object-visibility-support-model-v3/NEW-pure-kinematic-body-frame-returns.json';dpath=S/'e037-object-dynamic-runtime-v1/dynamic-source-data.npz';stagepath=S/'e037-object-source-authority-v1/NEW-source-world-native-frame-return.json';stats_path=S/'e037-training-source-constraints-v1/all151-NEW-training-source-statistics.json';meshpath=frozen/manifest['objects'][0]['mesh']
provenance={str(p):sha(p)for p in [frozen/'manifest.json',frozen/'targets.npz',modelpath,gpath,bpath,dpath,stagepath,stats_path,meshpath,C/'object-authority-checkpoint-v1.json',C/'hard-source-filter-checkpoint-v1.json',*[C/n for n in oldreceipts]]}
save('NEW-declaration-BEFORE.json',dict(rotation_order='R_world_f=R_world_F_f Exp(theta_f); t_world_f=t_world_F_f+delta_f',rotation_coordinates='ALL THREE original-object-frame components at ALL151 clocks',variables=[dict(clock=i,world_translation_columns=list(range(6*i,6*i+3)),object_rotation_columns=list(range(6*i+3,6*i+6)))for i in range(151)],translation_or_rotation_halfwidths_NOT_selected=True,no_episode_pose_assignment=True,no_solver_or_method_selected=True,old49071omissions_and_all_fixed_axis_affine_certificates_INVALID_FOR_THIS_FAMILY=True,all_original_source_motion_native_self_swept_motor_grasp_gates_required=True,provenance=provenance))
assert provenance[str(frozen/'manifest.json')]=='f438adf3da846ec1c76c31c7b6d6c12b46bea76004aa7745cfada504378da606'
assert provenance[str(modelpath)]=='755fc51729d4c4fc03d04108d39bec2b249074351863f096f9aa498c9ac28566'
for p in [gpath,bpath,stagepath,stats_path,dpath]:assert provenance[str(p)]==registry[str(p)]
assert provenance[str(frozen/'targets.npz')]==manifest['files']['targets.npz']
model=mujoco.MjModel.from_binary_path(str(modelpath))
body_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(b))for b in model.geom_bodyid]
static=dict(mujoco_version=mujoco.__version__,ngeom=int(model.ngeom),nbody=int(model.nbody),explicit_pairs=int(model.npair),body_exclusions=int(model.nexclude),geoms=[dict(geom_id=g,name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g),body_id=int(model.geom_bodyid[g]),body=body_names[g],type=int(model.geom_type[g]),contype=int(model.geom_contype[g]),conaffinity=int(model.geom_conaffinity[g]),geom_pos=model.geom_pos[g],geom_quat_wxyz=model.geom_quat[g])for g in range(model.ngeom)],no_MjData_FK_forward_position_distance_or_physics=True)
save('NEW-static-MJB-geometry-metadata-RETURN.json',static);assert mujoco.__version__=='3.8.1'
with np.load(gpath)as z:geom_cache={k:z[k].copy()for k in z.files}
with np.load(dpath)as z:
 world=z['original_world_from_object'];cam=z['original_camera_from_object'];Rcw=z['camera_from_world_linear'];times=z['timestamps_s'];indices=z['frame_indices'];K=z['K'];faces=z['faces']
frames=read(bpath)['poses'];object_name='wooden_spoon_object';object_geoms=[g for g,b in enumerate(body_names)if b==object_name];fixed_geoms=[g for g,b in enumerate(body_names)if b!=object_name and (b=='support'or b.startswith('dex3_'))];other_geoms=[g for g in range(model.ngeom)if g not in object_geoms+fixed_geoms]
unknown=[g for g in range(model.ngeom)if f'g{g}_vertices_body'not in geom_cache or any(body_names[g]not in frames[str(i)]for i in range(151))]
save('NEW-compiled-cache-coverage-RETURN.json',dict(object_geoms=object_geoms,fixed_hand_support_geoms=fixed_geoms,other_geoms=other_geoms,missing_cached_geometry_or_frame=unknown,vertices_already_in_BODY_coordinates_NOT_retransformed_by_geom=True,source='SAVED MuJoCo3.8.1 native_mesh.compiled_mesh_vertices returns BY HASH'))
assert not unknown and object_geoms and fixed_geoms
# Retain every cross pair in hand/object/support scope, including zero-mask
# pairs. Eligibility is declared separately, never used as a broadphase drop.
pairs=[];elig=[]
for i in range(151):
 for a in fixed_geoms:
  for b in object_geoms:
   pairs.append((i,a,b));elig.append(bool((int(model.geom_contype[a])&int(model.geom_conaffinity[b]))or(int(model.geom_contype[b])&int(model.geom_conaffinity[a]))))
P=np.array(pairs,np.int32);eligible=np.array(elig,bool)
paired=arrays('NEW-ALL-cross-pair-incidence-RETURN.npz',frame_fixed_object=P,collision_mask_eligible=eligible,old_broadphase_omissions_reused=np.array(False));journal('AFTER_NEW_complete_pair_incidence_RETURN',actual_RETURN=paired)
# A free support body is bound directly to ORIGINAL MJB qpos0, not world-fixed
# defaults or guessed URDF placement. No new FK state is evaluated.
support_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'support');j0=int(model.body_jntadr[support_id]);qadr=int(model.jnt_qposadr[j0]);q0=np.array(model.qpos0[qadr:qadr+7]);support=np.array(frames['0']['support']);R0=Rotation.from_quat(q0[[4,5,6,3]]).as_matrix();support_error=max(float(np.max(abs(support[:3,3]-q0[:3]))),float(np.max(abs(support[:3,:3]-R0))))
save('NEW-support-qpos0-binding-RETURN.json',dict(body_id=support_id,joint_id=j0,joint_type=int(model.jnt_type[j0]),qpos0_wxyz=q0,cached_support_pose=support,component_difference=support_error,diagnostic_only_NOT_new_native_certificate=True));assert int(model.jnt_type[j0])==int(mujoco.mjtJoint.mjJNT_FREE)and support_error<1e-14
base_body=np.array([[frames[str(i)][body_names[g]]for g in range(model.ngeom)]for i in range(151)])
source_V=np.asarray(trimesh.load(meshpath,force='mesh',process=False).vertices,float)
# NEW rotational derivative coefficients for all visual vertices. These are
# SEARCH factors only; cache aliases and actual changed-pose queries stay separate.
Sx=np.zeros((len(source_V),3,3));Sx[:,0,1]=-source_V[:,2];Sx[:,0,2]=source_V[:,1];Sx[:,1,0]=source_V[:,2];Sx[:,1,2]=-source_V[:,0];Sx[:,2,0]=-source_V[:,1];Sx[:,2,1]=source_V[:,0]
sourceJ=-np.einsum('fij,vjk->fvik',cam[:,:3,:3],Sx,optimize=False)
source_ret=arrays('NEW-source-rotation-factors-RETURN.npz',visual_vertices=source_V,camera_rotation_J_at_zero=sourceJ,original_camera_from_object=cam,camera_from_world_translation_linear=Rcw,original_world_from_object=world,original_timestamps_s=times,original_frame_indices=indices,K=K,faces=faces)
journal('AFTER_NEW_source_rotation_factors_RETURN',actual_RETURN=source_ret)
# Native compiled geometry map is body_R_F Exp(theta) V_body + body_t_F +
# stage_rotation delta. Object geom normals rotate with the same body map.
stage=np.array(read(stagepath)['stage_rotation']);nativeJ={}
for g in object_geoms:
 V=geom_cache[f'g{g}_vertices_body'];Sv=np.zeros((len(V),3,3));Sv[:,0,1]=-V[:,2];Sv[:,0,2]=V[:,1];Sv[:,1,0]=V[:,2];Sv[:,1,2]=-V[:,0];Sv[:,2,0]=-V[:,1];Sv[:,2,1]=V[:,0];nativeJ[f'g{g}_rotation_J_at_zero']=-np.einsum('fij,vjk->fvik',base_body[:,g,:3,:3],Sv,optimize=False)
native_ret=arrays('NEW-native-body-rotation-factors-RETURN.npz',cached_body_poses=base_body,stage_world_translation_linear=stage,**geom_cache,**nativeJ);journal('AFTER_NEW_native_rotation_factors_RETURN',actual_RETURN=native_ret)
angular=angular_clock_maps(world[:,:3,:3],times)
sp=sparse.vstack([angular['angular_velocity_J'],angular['angular_acceleration_J']],format='csr');sparse.save_npz(O/'NEW-original-angular-clock-Jacobian-RETURN.npz',sp);(O/'NEW-original-angular-clock-Jacobian-RETURN.npz').chmod(0o600)
angularret=arrays('NEW-angular-clock-base-factors-RETURN.npz',principal_relative_rotvec=angular['principal_relative_rotvec'],angular_velocity=angular['angular_velocity'],angular_acceleration=angular['angular_acceleration'],clock_blocks=angular['clock_blocks'],pi_branch_unknown=angular['pi_branch_unknown']);journal('AFTER_NEW_original_angular_clock_factors_RETURN',actual_RETURN=angularret,J_sha256=sha(O/'NEW-original-angular-clock-Jacobian-RETURN.npz'))
# Source gradient SEARCH branches use only saved baseline NN/rank returns.
# The full NN/ranks and raster MUST change through actual unmodified tests.
stats=read(stats_path);grad=[];central_unknown=[];point_paths=[]
for f,st in enumerate(stats):
 mp=Path(st['median_data']['path'])if 'median_data'in st else None
 if mp is None:
  # Actual saved dataset has an explicit path below; do not regenerate a return.
  keys=[k for k in st if 'median'in k.lower()];raise KeyError(('saved_median_RETURN_path',keys))
 assert sha(mp)==st['median_data']['sha256']
 with np.load(mp)as z:
  y=z['point_local'];ids=z['nearest_vertex_indices'];dist=z['nearest_vertex_distances'];central=z['central_point_indices'];weights=z['central_weights'];g=np.zeros(3);zero=False
  for i,w in zip(central,weights,strict=True):
   if dist[i]==0:zero=True;continue
   u=(y[i]-source_V[ids[i]])/dist[i];g+=w*np.cross(u,y[i])
 grad.append(g);central_unknown.append(zero);point_paths.append(dict(frame=f,saved_median_RETURN=str(mp),sha256=sha(mp)))
gradret=arrays('NEW-training-median-rotation-SEARCH-gradient-RETURN.npz',rotation_SEARCH_gradient_at_F=np.array(grad),zero_central_unknown=np.array(central_unknown));journal('AFTER_NEW_saved_rank_branch_rotation_SEARCH_gradient_RETURN',actual_RETURN=gradret)
authority=read(C/'object-authority-checkpoint-v1.json');oldmotion=S/'e037-object-dynamic-runtime-v1/dynamic-motion-linear-map.npz'
save('NEW-full-pose-factor-semantics-RETURN.json',dict(variables=906,clocks=151,translation_variables=453,rotation_variables=453,source_pose='camera_R_F Exp(theta), camera_t_F + Rcw delta',world_pose='world_R_F Exp(theta), world_t_F + delta',native_pose='cached_native_body_R_F Exp(theta), cached_native_body_t_F + stage delta',all_same_ORIGINAL_object_frame_coordinates=True,zero_F_uses_saved_cache_NO_new_baseline_query=True,NEW_nonzero_pose_actual_source_requires_sealed_proposal=True,source_camera_and_native_base_gauge_old_closures_reused_BY_HASH_NOT_reshown=True,old_cache_alias_numerical_and_source_uncertainties_retained=True,visual_vertices=len(source_V),compiled_vertices_are_DISTINCT_native_support=True,all_cross_pairs=len(P),mask_eligible_pairs=int(eligible.sum()),broadphase_omissions=0,old49071omission_certificates_NOT_inherited=True,old_fixed_axes_and_affine_object_support_NOT_inherited=True,original_linear_motion_maps_BY_HASH=dict(path=str(oldmotion),sha256=sha(oldmotion)),NEW_angular_function='Log(Rnew[i+1] Rnew[i]^T)/actual_dt; midpoint-difference angular acceleration; ACTUAL p95',NEW_angular_search_J_shape=sp.shape,NEW_angular_search_J_nonzeros=sp.nnz,pi_branch_unknown_clocks=np.flatnonzero(angular['pi_branch_unknown']),all_original_caps_COPIED=authority['original_object_motion_caps'],object_motion_comparison=1e-9,original_fullpixel_source_POST_only_comparison=1e-10,robot_tolerances_UNCHANGED=True,training_NN_rank_raster_actual_operators_NOT_replaced_by_search_J=True,training_point_sources_BY_HASH=point_paths,original_fullpixel_heldout_failures_NOT_inputs=True,source_accuracy_native_self_swept_motor_grasp_NOT_certified=True,no_solver_method_parameter_box_pose_evaluation_selected=True))
report=dict(schema='v2d.NEW-full-object-pose-factors.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_variables=906,NEW_continuous_rotation_components=453,source_rotation_J_coefficients=sourceJ.size,compiled_geoms=int(model.ngeom),object_geoms=len(object_geoms),fixed_hand_support_geoms=len(fixed_geoms),all_cross_pair_instances=len(P),mask_eligible_pair_instances=int(eligible.sum()),broadphase_omissions=0,new_angular_J_coefficients=int(sp.nnz),pi_branch_unknown=int(np.count_nonzero(angular['pi_branch_unknown'])),source_rotational_SEARCH_gradients=151,NEW_pose_assignments=0,NEW_solver_calls=0,NEW_episode_source_NN_raster_native_FK_physics_GPU=0,all_original_F_data_authority_gates_UNCHANGED=True)
save('report.json',report);journal('AFTER_NEW_full_pose_construction_RETURN',actual_RETURN=report);print(json.dumps(report))
