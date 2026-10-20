"""ONE NEW training-ray/analytic-support/full-clock object SOCP construction.

No old source/native audit is replayed. Compiled geometry is read, and two
independent pure kinematic frame bindings are checked without native positioning.
Every support plane is NEW construction evidence, never a reconstructed old gap.
"""
from pathlib import Path
from fractions import Fraction as Q
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path[:0]=[str(R/'scripts/dex3_training')]
import cv2,mujoco,numpy as np,trimesh
from scipy import sparse
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.native_mesh import compiled_mesh_vertices
from v2d_dex3_flashchord.kinematics import URDFKinematics
from object_visibility_support_v1 import CONFIG,first_cell_for_ray,support_axis_plane,p95_speed_envelope,sparse_record
from thumb_conic_geometry_v1 import plain,journal
O=S/'e037-object-visibility-support-model-v1';F=S/'e037-thumb-ray-independent-validation-v1/frozen';start=time.monotonic();cv2.setNumThreads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
manifest=read(F/'manifest.json');assert sha(F/'manifest.json')=='f438adf3da846ec1c76c31c7b6d6c12b46bea76004aa7745cfada504378da606';native_path=S/'e037-motion-constrained-wrist-native-v1/native-model.mjb';assert sha(native_path)=='755fc51729d4c4fc03d04108d39bec2b249074351863f096f9aa498c9ac28566';assert mujoco.__version__=='3.8.1'
with np.load(F/'targets.npz')as z:qF=z['robot_qpos'];native_obj=z['object_poses'];times=z['timestamps_s'];frames=z['frame_indices']
with np.load(S/'e037-joint-pose-translation-repair-v1/unqualified-combined-object-proposal.npz')as z:world=z['world_from_object'];camera_obj=z['camera_from_object'];assert np.array_equal(times,z['timestamps_s'])
with np.load(S/'plane-aligned-pose-temporal-path-v4/inferred-object-trajectory.npz')as z:cameras=z['candidate_camera_c2w'];assert np.array_equal(times,z['timestamps_s'])
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz')as z:K=z['intrinsics']
stage=np.asarray(read(S/'plane-aligned-articulated-native-validation-v2/frozen/command-proposal-report.json')['stage_rotation_from_inferred_support_normal']);camera_inv=np.linalg.inv(cameras);authority=read(S/'e037-object-source-authority-v1/report.json');caps=authority['original_object_motion_caps_COPIED'];maps=read(S/'e037-object-translation-binding-v1/all299-UNQUALIFIED-object-motion-affine-maps.json');n=len(times)*3
with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 emit(dict(kind='BEFORE_ONE_NEW_object_visibility_support_model',configuration=CONFIG,old14controls_reused_BY_HASH=sha(S/'e037-object-translation-contract-v1/report.json'),NEW11controls_sha256=sha(S/'e037-object-visibility-support-contract-v1/report.json'),original_authority_sha256=sha(S/'e037-object-source-authority-v1/report.json'),all299motion_maps_reused_BY_HASH=True,NO_solver_or_old_source_native_audit_called=True))
 mesh_path=F/manifest['objects'][0]['mesh'];mesh=trimesh.load(mesh_path,force='mesh',process=False);vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces);assert len(faces)==12000
 model=mujoco.MjModel.from_binary_path(str(native_path));compiled={};localnorm={};geombodies={};gids=[]
 for g in range(model.ngeom):
  if not(model.geom_contype[g]or model.geom_conaffinity[g]):continue
  body=model.body(int(model.geom_bodyid[g])).name
  if not(body.startswith('dex3_')or body in ['wooden_spoon_object','support']):continue
  assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
  raw=compiled_mesh_vertices(model,int(model.geom_dataid[g]),mujoco_version=mujoco.__version__);G=Rotation.from_quat(model.geom_quat[g][[1,2,3,0]]).as_matrix();vv=np.einsum('ni,ji->nj',raw,G,optimize=False)+model.geom_pos[g];h=ConvexHull(vv);compiled[g]=vv;localnorm[g]=h.equations[:,:3];geombodies[g]=body;gids.append(g)
 arrays={}
 for g in gids:arrays[f'g{g}_vertices_body']=compiled[g];arrays[f'g{g}_normals_body']=localnorm[g]
 np.savez_compressed(O/'NEW-compiled-support-geometry.npz',**arrays)
 emit(dict(kind='AFTER_NEW_compiled_support_constructor_RETURN',pinned_MJB_sha256=sha(native_path),compiled_reader_sha256=sha(R/'scripts/dex3_training/v2d_dex3_flashchord/native_mesh.py'),compiled_geometry_npz_sha256=sha(O/'NEW-compiled-support-geometry.npz'),geoms=[dict(geom=g,body=geombodies[g],vertices=len(compiled[g]),normals=len(localnorm[g]),contype=int(model.geom_contype[g]),conaffinity=int(model.geom_conaffinity[g]))for g in gids],no_mesh_or_physical_parameter_mutation=True,no_native_position_or_gap_measurement=True))
 kin={s:URDFKinematics(F/f'robot/urdfs/g1/dex3_{s}.urdf')for s in ['left','right']};bodyposes={};max_fk=0.;fk_count=0
 for f in range(len(times)):
  bp={}
  for si,side in enumerate(['left','right']):
   qs=dict(zip(manifest['joint_names'][side],qF[f,si,7:],strict=True));poses=kin[side].poses(qF[f,si,:7],qs);prefix=f'dex3_{side}_';palm=prefix+kin[side].root_name;bid=model.body(palm).id;computed={bid:poses[kin[side].root_name]};pending=[b for b in range(model.nbody)if model.body(b).name.startswith(prefix)and b!=bid]
   while pending:
    rest=[]
    for b in pending:
     parent=int(model.body_parentid[b])
     if parent not in computed:rest.append(b);continue
     B=np.eye(4);B[:3,:3]=Rotation.from_quat(model.body_quat[b][[1,2,3,0]]).as_matrix();B[:3,3]=model.body_pos[b]
     for j in range(int(model.body_jntadr[b]),int(model.body_jntadr[b])+int(model.body_jntnum[b])):
      assert model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE;name=model.joint(j).name;name=name.removeprefix(prefix);assert name in qs;H=np.eye(4);H[:3,:3]=Rotation.from_rotvec(model.jnt_axis[j]*qs[name]).as_matrix();H[:3,3]=model.jnt_pos[j]-H[:3,:3]@model.jnt_pos[j];B=B@H
     computed[b]=computed[parent]@B;actual=poses[model.body(b).name.removeprefix(prefix)];err=float(abs(computed[b]-actual).max());max_fk=max(max_fk,err);fk_count+=1;assert err<1e-10
    assert len(rest)<len(pending);pending=rest
   for b,T in computed.items():bp[model.body(b).name]=T
  ot=np.eye(4);ot[:3,:3]=Rotation.from_quat(native_obj[f,0,3:]).as_matrix();ot[:3,3]=native_obj[f,0,:3];bp['wooden_spoon_object']=ot
  sb=model.body('support').id;assert model.body_parentid[sb]==0 and model.body_jntnum[sb]==0;st=np.eye(4);st[:3,:3]=Rotation.from_quat(model.body_quat[sb][[1,2,3,0]]).as_matrix();st[:3,3]=model.body_pos[sb];bp['support']=st;bodyposes[f]=bp
 save('NEW-pure-kinematic-body-frame-returns.json',dict(maximum_URDF_vs_compiled_kinematic_component_error=max_fk,independent_body_frame_checks=fk_count,poses=bodyposes,no_mj_fwdPosition_or_native_gap_calls=True))
 P=np.eye(n)*2*CONFIG['translation_damping']/(CONFIG['world_translation_construction_halfwidth_m']**2);q=np.zeros(n);linear=[];rhs=[];origins=[];ray_records=[];geometry_records=[];missing=[]
 def row(terms,b,origin):linear.append(terms);rhs.append(float(b));origins.append(origin)
 for j in range(n):
  row({j:1.},CONFIG['world_translation_construction_halfwidth_m'],dict(kind='algorithm_world_translation_box_upper',column=j,NOT_original_object_box=True));row({j:-1.},CONFIG['world_translation_construction_halfwidth_m'],dict(kind='algorithm_world_translation_box_lower',column=j,NOT_original_object_box=True))
 objectgeoms=[g for g in gids if geombodies[g]=='wooden_spoon_object'];fixedgeoms=[g for g in gids if g not in objectgeoms];radius=np.sqrt(3)*CONFIG['world_translation_construction_halfwidth_m'];omitted=0
 for f in range(len(times)):
  assert time.monotonic()-start<210,'NEWpreflight internal wall bound'
  emit(dict(kind='BEFORE_NEW_frame_training_and_analytic_support_constructor',frame=f,no_pose_assigned=True));dp=S/f'fast-stereo-cpu-full-e037-v1/{f:06d}-depth.npz';mp=S/f'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/{f:06d}.png';ar=read(S/f'e037-object-source-authority-v1/frame-{f:03d}-source-authority.json');assert sha(dp)==ar['depth_sha256']and sha(mp)==ar['mask_sha256']
  with np.load(dp)as z:depth=z['depth_m'];valid=z['left_right_consistent']
  mask=cv2.imread(str(mp),0)>127;yy,xx=np.indices(mask.shape);eligible=valid&mask&np.isfinite(depth)&(depth>.1)&(depth<2.5)&((xx+yy)%2==0);yx=np.argwhere(eligible);assert len(yx)>=CONFIG['training_rays_per_clock'];indices=np.linspace(0,len(yx)-1,CONFIG['training_rays_per_clock']).astype(int);pixels=yx[indices][:,::-1];observations=depth[pixels[:,1],pixels[:,0]]
  partition=dict(frame=f,depth_sha256=sha(dp),mask_sha256=sha(mp),selection='24 deterministic raster-order linspace samples from ORIGINAL eligible EVEN pixels; fixed BEFORE ray evaluation',eligible_even_count=len(yx),indices=indices,pixels=pixels,observed_depth_m=observations,heldout_values_passed_to_constructor=False,source_arrays_and_flags_NOT_changed=True);save(f'frame-{f:03d}-DECLARED-training-partition.json',partition);emit(dict(kind='BEFORE_NEW_training_ray_cell_evaluation',proposal=partition))
  vC=np.einsum('nj,ij->ni',vertices,camera_obj[f,:3,:3],optimize=False)+camera_obj[f,:3,3];tri=vC[faces];Rcw=camera_inv[f,:3,:3]
  for k,(uv,obs)in enumerate(zip(pixels,observations,strict=True)):
   ray=np.array([(uv[0]-K[0,2])/K[0,0],(uv[1]-K[1,2])/K[1,1],1.]);ret=first_cell_for_ray(tri,ray);record=dict(frame=f,ray_index=k,pixel=uv,observation_depth_m=float(obs),camera_from_world_linear=Rcw,local_cell_RETURN=ret);ray_records.append(record);emit(dict(kind='AFTER_NEW_training_ray_cell_RETURN',returned=record))
   if ret['status'].startswith('MISSING'):missing.append(dict(frame=f,ray_index=k,record=record));continue
   U=np.asarray(ret['ud'])@Rcw;V=np.asarray(ret['vd'])@Rcw;Z=np.asarray(ret['zd'])@Rcw;D=Z/CONFIG['source_depth_conditioning_m'];r0=(ret['z0']-obs)/CONFIG['source_depth_conditioning_m'];ids=np.arange(3*f,3*f+3);P[np.ix_(ids,ids)]+=2*np.outer(D,D);q[ids]+=2*r0*D
   for vals,b,kind in [(-U,ret['u0'],'u_nonnegative'),(-V,ret['v0'],'v_nonnegative'),(U+V,1-ret['u0']-ret['v0'],'u_plus_v_at_most_one'),(-Z,ret['z0']-CONFIG['visibility_cell_positive_depth_m'],'positive_ray_plane_depth')]:row({3*f+j:float(vals[j])for j in range(3)},b,dict(kind='ONE_training_face_cell_'+kind,frame=f,ray_index=k,face=ret['face'],NOT_full_visibility_or_original_source_gate=True))
  for constant,threshold,kind in [(camera_obj[f,2,3],.05+1e-9,'tracker_center'),(float(vC[:,2].min()),.01+1e-9,'all_source_vertices')]:row({3*f+j:float(-Rcw[2,j])for j in range(3)},constant-threshold,dict(kind='strict_depth_sufficient_construction_'+kind,frame=f,construction_inward_margin_m=1e-9,NOT_candidate_tolerance=True))
  verts={};norms={}
  for g in gids:
   T=bodyposes[f][geombodies[g]];verts[g]=compiled[g]@T[:3,:3].T+T[:3,3];norms[g]=localnorm[g]@T[:3,:3].T
  for og in objectgeoms:
   ov=verts[og];omin,omax=ov.min(0),ov.max(0)
   for hg in fixedgeoms:
    if not((int(model.geom_contype[hg])&int(model.geom_conaffinity[og]))or(int(model.geom_contype[og])&int(model.geom_conaffinity[hg]))):continue
    hv=verts[hg];hmin,hmax=hv.min(0),hv.max(0);aabb_gap=np.maximum(np.maximum(omin-hmax,hmin-omax),0);dist=float(np.linalg.norm(aabb_gap))
    if dist>radius+.001+CONFIG['packing_construction_guard_m']:omitted+=1;continue
    axes=np.r_[norms[hg],-norms[hg],norms[og],-norms[og]];plane=support_axis_plane(hv,ov,axes);normal_world=np.asarray(plane['normal'])@stage;rec=dict(frame=f,hand_or_support_geom=hg,object_geom=og,fixed_body=geombodies[hg],NEW_axis_plane_RETURN=plane,coefficient_world=normal_world,broadphase_radius_m=radius,original_native_gate_m=.001,old_gap_normals_NOT_reconstructed=True);geometry_records.append(rec);row({3*f+j:float(-normal_world[j])for j in range(3)},-plane['minimum_dot_native_translation'],dict(kind='NEW_sufficient_fixed_axis_support_plane',record=len(geometry_records)-1,frame=f))
  emit(dict(kind='AFTER_NEW_frame_training_and_support_constructor',frame=f,training_rays=CONFIG['training_rays_per_clock'],model_source_heldout_values_unused=True,support_records_so_far=len(geometry_records),broadphase_omissions_so_far=omitted,no_native_contact_measurements=True))
 save('all-NEW-training-ray-cell-RETURNS.json',ray_records);save('all-NEW-support-plane-RETURNS.json',geometry_records);save('all-missing-NEW-training-ray-RETURNS.json',missing)
 current_v=np.diff(world[:,:3,3],axis=0)/np.diff(times)[:,None];speeds=np.linalg.norm(current_v,axis=1);envelope=p95_speed_envelope(speeds,caps['p95_linear_speed_m_s'],caps['maximum_linear_speed_m_s']);save('NEW-sufficient-p95-speed-envelope-RETURN.json',envelope)
 rr=[];cc=[];vv=[]
 for i,r in enumerate(linear):
  for j,v in r.items():
   if v:rr.append(i);cc.append(j);vv.append(v)
 A=sparse.csr_matrix((vv,(rr,cc)),shape=(len(rhs),n));b=np.asarray(rhs);nonneg=len(rhs);socorigins=[];socrows=[];socrhs=[]
 for mi,m in enumerate(maps):
  rows=[];const=[]
  for sr in m['scalar_maps']:
   rowterms={int(t['column']):-float(Q(t['coefficient']['numerator'],t['coefficient']['denominator']))for t in sr['terms']};rows.append(rowterms);co=sr['constant'];const.append(float(Q(co['numerator'],co['denominator'])))
  cap=float(envelope['per_edge_caps'][m['index']])if m['family']=='velocity'else caps['maximum_linear_acceleration_m_s2'];base=len(socrhs);socrhs.extend([cap,*const]);socrows.append({});socrows.extend(rows);socorigins.append(dict(kind='NEW_original_object_motion_SOC',saved_map_index=mi,source_family=m['family'],source_index=m['index'],cap=cap,canonical_start=nonneg+base,dimension=4,sufficient_p95_speed_envelope_NOT_original_gate=True if m['family']=='velocity'else False))
 r=[];c=[];v=[]
 for i,terms in enumerate(socrows):
  for j,a in terms.items():r.append(i);c.append(j);v.append(a)
 A=sparse.vstack([A,sparse.csr_matrix((v,(r,c)),shape=(len(socrhs),n))],format='csc');b=np.r_[b,socrhs];P=sparse.csc_matrix(np.triu(P));sparse.save_npz(O/'canonical-A.npz',A);sparse.save_npz(O/'canonical-P.npz',P);np.savez_compressed(O/'canonical-vectors.npz',q=q,b=b);save('canonical-row-origins.json',dict(nonnegative=origins,SOC=socorigins));save('canonical-shapes-and-config.json',dict(columns=n,rows=A.shape[0],nonnegative_rows=nonneg,SOC4=len(socorigins),config=CONFIG,pinned_native_model_sha256=sha(native_path),starting_F_targets_sha256=sha(F/'targets.npz'),object_source_world_arrays_sha256=sha(S/'e037-joint-pose-translation-repair-v1/unqualified-combined-object-proposal.npz'),all299maps_sha256=sha(S/'e037-object-translation-binding-v1/all299-UNQUALIFIED-object-motion-affine-maps.json'),all_source_depth_mask_hashes_sha256=sha(S/'e037-object-source-authority-v1/cached-source-hashes.json')))
 summary=dict(schema='v2d.NEW-training-visibility-support-object-model.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),columns=n,rows=A.shape[0],nonnegative_rows=nonneg,SOC4=len(socorigins),NEW_training_partitions=len(times),NEW_training_rays=len(ray_records),baseline_missing_training_rays=len(missing),NEW_analytic_support_planes=len(geometry_records),uniform_box_broadphase_omissions=omitted,compiled_collision_geoms=len(gids),independent_pure_URDF_compiled_body_checks=fk_count,maximum_body_frame_component_error=max_fk,full_original_source_gate_NOT_evaluated_or_optimized=True,original_p95_preserved_by_declared_sufficient_envelope_only_NOT_gate_change=True,source_cell_does_NOT_prove_first_surface_occlusion_or_mask_gate=True,no_native_position_gap_physics_or_solver_calls=True,no_old_native_gap_normal_reconstructed=True,no_object_pose_assigned_or_qualified=True,all_old_source_masks_depth_flags_geometry_parameters_F_UNCHANGED=True,wall_seconds=time.monotonic()-start);emit(dict(kind='AFTER_ONE_NEW_object_visibility_support_model_RETURN',returned=summary));save('report.json',summary);print(json.dumps(summary))
