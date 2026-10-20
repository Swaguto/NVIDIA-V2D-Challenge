"""NEW all-axis dynamic-source dataset from saved immutable records, no FK/ray replay."""
from pathlib import Path
import datetime,hashlib,json,os,time
import numpy as np,trimesh
from scipy import sparse
from object_dynamic_visibility_v1 import CONFIG
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-object-dynamic-runtime-v1';M=S/'e037-object-visibility-support-model-v3';F=S/'e037-thumb-ray-independent-validation-v1/frozen';start=time.monotonic();read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 emit(dict(kind='BEFORE_ONE_NEW_dynamic_source_union_support_dataset',configuration=CONFIG,old28controls_AND_baseline_queries_NOT_repeated=True,NEW10controls_sha256=sha(S/'e037-object-dynamic-contract-v1/report.json'),old_fixed_face_SOCP_failure_and_certificate_preserved=True))
 planes=read(M/'all-NEW-support-plane-RETURNS.json');poses=read(M/'NEW-pure-kinematic-body-frame-returns.json')['poses'];rays=read(M/'all-NEW-training-ray-cell-RETURNS.json');assert len(planes)==17520 and len(rays)==3624
 old_geom=S/'e037-object-visibility-support-model-v1/NEW-compiled-support-geometry.npz';local={};norms={}
 with np.load(old_geom)as z:
  for key in z.files:
   g=int(key.split('_')[0][1:])
   if key.endswith('vertices_body'):local[g]=z[key].copy()
   elif key.endswith('normals_body'):norms[g]=z[key].copy()
 frames={};body_to_geom={}
 # Body identities come from the saved source/geometry producer RETURN by hash.
 oldjournal=S/'e037-object-visibility-support-model-v1/normalized-BEFORE-AFTER-journal.jsonl';metadata=None
 for line in oldjournal.open():
  r=json.loads(line)
  if r['kind']=='AFTER_NEW_compiled_support_constructor_RETURN':metadata=r;break
 assert metadata and sha(old_geom)==metadata['compiled_geometry_npz_sha256'];geom_body={r['geom']:r['body']for r in metadata['geoms']}
 stage=np.asarray(read(S/'plane-aligned-articulated-native-validation-v2/frozen/command-proposal-report.json')['stage_rotation_from_inferred_support_normal']);N=sum(len(r['NEW_axis_plane_RETURN']['all_axis_gaps'])for r in planes);axes=np.lib.format.open_memmap(O/'ALL-retained-axis-world-coefficients.npy',mode='w+',dtype=np.float64,shape=(N,3));g0=np.lib.format.open_memmap(O/'ALL-retained-axis-saved-gaps.npy',mode='w+',dtype=np.float64,shape=(N,));ptr=[0];pairframes=[];pairids=[];offset=0;maxerror=0.;selectederror=0.;axischecks=0;framecache=None;oldf=None
 for i,r in enumerate(planes):
  f=r['frame'];hg=r['hand_or_support_geom'];og=r['object_geom'];emit(dict(kind='BEFORE_NEW_all_face_direction_union_binding',pair=i,frame=f,geoms=[hg,og],source_selected_plane_record=i,old_saved_gaps_REUSED_BY_HASH=True,no_native_gap_or_FK_call=True))
  if oldf!=f:
   framecache={}
   for g in local:
    T=np.asarray(poses[str(f)][geom_body[g]]);v=np.column_stack([local[g][:,0]*T[j,0]+local[g][:,1]*T[j,1]+local[g][:,2]*T[j,2]+T[j,3]for j in range(3)]);n=np.einsum('ni,ji->nj',norms[g],T[:3,:3],optimize=False);framecache[g]=(v,n)
   oldf=f
  H,Hn=framecache[hg];Ov,On=framecache[og];a=np.r_[Hn,-Hn,On,-On];a=a/np.linalg.norm(a,axis=1)[:,None];world=np.einsum('ni,ij->nj',a,stage,optimize=False);saved=np.asarray(r['NEW_axis_plane_RETURN']['all_axis_gaps']);assert len(a)==len(saved);hmax=np.einsum('ni,ki->nk',H,a,optimize=False).max(0);omin=np.einsum('ni,ki->nk',Ov,a,optimize=False).min(0);actual=omin-hmax;err=float(abs(saved-actual).max());sel=r['NEW_axis_plane_RETURN']['selected_axis'];se=float(abs(world[sel]-np.asarray(r['coefficient_world'])).max());axes[offset:offset+len(a)]=world;g0[offset:offset+len(a)]=saved;axes.flush();g0.flush();ret=dict(pair=i,frame=f,geoms=[hg,og],axis_start=offset,axis_end=offset+len(a),actual_axis_block_sha256=hashlib.sha256(world.tobytes()).hexdigest(),actual_saved_gap_block_sha256=hashlib.sha256(saved.tobytes()).hexdigest(),NEW_independent_all_direction_support_error_m=err,prior_selected_direction_error=se,all_directions_retained_no_anchor_selection=True,arrays_durable_before_assertions=True);emit(dict(kind='AFTER_NEW_all_face_direction_union_binding_RETURN',actual_RETURN=ret));assert err<1e-12 and se<1e-12;maxerror=max(maxerror,err);selectederror=max(selectederror,se);axischecks+=len(a);offset+=len(a);ptr.append(offset);pairframes.append(f);pairids.append([hg,og])
 np.savez_compressed(O/'dynamic-axis-group-bindings.npz',ptr=np.asarray(ptr,np.int64),pair_frames=np.asarray(pairframes,np.int16),pair_geoms=np.asarray(pairids,np.int32));assert offset==N
 # Same declared original EVEN samples and baseline returns, copied without rays.
 pixels=[];observations=[];baseface=[];basez=[];basezd=[];Rcws=[]
 for f in range(151):
  partition=read(M/f'frame-{f:03d}-DECLARED-training-partition.json');pixels.append(partition['pixels']);observations.append(partition['observed_depth_m']);fr=[];zr=[];dz=[]
  for r in rays[24*f:24*(f+1)]:
   c=r['local_cell_RETURN'];missing=c['status'].startswith('MISSING');fr.append(-1 if missing else c['face']);zr.append(float('inf')if missing else c['z0']);dz.append([0.,0.,0.]if missing else c['zd'])
  baseface.append(fr);basez.append(zr);basezd.append(dz);Rcws.append(rays[24*f]['camera_from_world_linear']);emit(dict(kind='AFTER_training_and_baseline_front_RETURNS_REUSED_BY_HASH',frame=f,partition_sha256=sha(M/f'frame-{f:03d}-DECLARED-training-partition.json'),baseline_rays_recomputed=0,missing_zero_derivative_placeholder_NOT_depth_or_hit=True))
 with np.load(S/'e037-joint-pose-translation-repair-v1/unqualified-combined-object-proposal.npz')as z:cam=z['camera_from_object'];times=z['timestamps_s'];worldposes=z['world_from_object'];frame_axis=z['frame_indices']
 with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz')as z:K=z['intrinsics']
 manifest=read(F/'manifest.json');mesh=trimesh.load(F/manifest['objects'][0]['mesh'],force='mesh',process=False);verts=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces);edges=np.unique(np.sort(np.r_[faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]],axis=1),axis=0);C0=np.stack([np.column_stack([verts[:,0]*T[j,0]+verts[:,1]*T[j,1]+verts[:,2]*T[j,2]+T[j,3]for j in range(3)])for T in cam]);np.savez_compressed(O/'dynamic-source-data.npz',C0=C0,faces=faces,edges=edges,K=K,pixels=np.asarray(pixels),observations=np.asarray(observations,float),baseline_faces=np.asarray(baseface,np.int32),baseline_depth=np.asarray(basez),baseline_depth_derivative=np.asarray(basezd),camera_from_world_linear=np.asarray(Rcws),timestamps_s=times,frame_indices=frame_axis,original_world_from_object=worldposes,original_camera_from_object=cam)
 A=sparse.load_npz(M/'canonical-A.npz').tocsr();meta=read(M/'canonical-shapes-and-config.json')
 with np.load(M/'canonical-vectors.npz')as z:b=z['b']
 indices=np.array([meta['nonnegative_rows']+4*k+j for k in range(299)for j in [1,2,3]]);D=-A[indices];constant=b[indices];sparse.save_npz(O/'dynamic-motion-linear-map.npz',D);np.save(O/'dynamic-motion-constants.npy',constant)
 authority=read(S/'e037-object-source-authority-v1/report.json');save('runtime-config-and-original-authority.json',dict(config=CONFIG,original_object_caps=authority['original_object_motion_caps_COPIED'],NEW_axis_count=N,pair_count=len(planes),source_partition_report_sha256=sha(M/'all-NEW-training-ray-cell-RETURNS.json'),prior299motion_map_sha256=sha(S/'e037-object-translation-binding-v1/all299-UNQUALIFIED-object-motion-affine-maps.json'),old_sufficient_speed_envelope_NOT_used=True,all_actual_original_motion_gates_still_required=True,compiled_mesh_geometry_sha256=sha(old_geom),actual_native_model_sha256=meta['pinned_native_model_sha256'],all49071broadphase_omissions_reused_BY_HASH=sha(S/'e037-object-visibility-support-binding-v3/report.json'),body_frame_bindings_reused_BY_HASH=sha(M/'NEW-pure-kinematic-body-frame-returns.json'),no_FK_or_baseline_ray_replay=True,source_base_coordinates_are_NEW_functional_inputs_NOT_reconstructed_old_missing_intermediates=True))
 out=dict(schema='v2d.NEW-object-dynamic-runtime-dataset.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),columns=453,pairs=len(planes),ALL_retained_face_axes=N,NEW_independent_axis_support_bindings=axischecks,maximum_all_direction_gap_error_m=maxerror,maximum_selected_prefix_direction_error=selectederror,training_partitions_copied=151,baseline_ray_returns_REUSED_BY_HASH=3624,missing_ray_returns_retained=1662,projected_full_geometry_edges=len(edges),original299motion_maps_reused=True,old_p95_envelope_NOT_substituted=True,no_native_position_FK_baseline_ray_solver_or_pose_calls=True,old_actual_vectors_F_source_geometry_parameters_and_gates_UNCHANGED=True,wall_seconds=time.monotonic()-start);save('report.json',out);emit(dict(kind='AFTER_ONE_NEW_dynamic_runtime_dataset_RETURN',actual_RETURN=out));print(json.dumps(out))
