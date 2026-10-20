"""Independent NEW training-only quantizer/count/point/NN/rank/gradient bindings."""
from pathlib import Path
import datetime,hashlib,json,math,os,time
import cv2,numpy as np,trimesh
from source_vertex_bvh_v1 import SourceVertexBVH
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-training-source-independent-v1';T=S/'e037-training-source-constraints-v1';D=S/'e037-object-dynamic-runtime-v1';F=S/'e037-thumb-ray-independent-validation-v1/frozen';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();cv2.setNumThreads(1)
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
def arrays(n,**kw):
 p=O/n;np.savez_compressed(p,**kw);p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 emit(dict(kind='BEFORE_NEW_independent_training_source_bindings',NEW12controls_reused_BY_HASH=sha(S/'e037-training-source-contract-v1/report.json'),old_scorer_renderer_source_fit_or_solver_NOT_replayed=True))
 # Four NEW independent spatial-certificate controls, no episode data.
 vv=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[1.,1.,1.],[0.,0.,0.]]);bvh=SourceVertexBVH(vv,leaf_size=2);pp=np.array([[.2,.1,0],[0,0,0],[5,4,3.]],float);dist=np.sqrt(np.einsum('pvi,pvi->pv',pp[:,None]-vv[None],pp[:,None]-vv[None],optimize=False));brute=np.sort(dist,axis=1)[:,:2];emit(dict(kind='BEFORE_NEW_BVH_nearest_pair_control',vertices=vv,points=pp,second_radius=brute[:,1]));ret=bvh.nearest_pair_certificate(pp,brute[:,1]);arrays('NEW-BVH-control-return.npz',actual=ret['first_two_distances'],brute=brute);emit(dict(kind='AFTER_NEW_BVH_nearest_pair_control_RETURN',actual_RETURN={k:v for k,v in ret.items()if k!='visited_nodes'}));assert np.max(abs(ret['first_two_distances']-brute))<1e-14
 cover=np.concatenate(bvh.leaves);emit(dict(kind='AFTER_NEW_BVH_cover_control_RETURN',covered_ids=cover,all_original_vertex_ids=np.arange(len(vv))));assert np.array_equal(np.sort(cover),np.arange(len(vv)))
 emit(dict(kind='AFTER_NEW_duplicate_zero_neighbor_control_RETURN',zero_nearest_pair=ret['first_two_distances'][1],source_zero_derivative_still_UNRESOLVED=True));assert np.array_equal(ret['first_two_distances'][1],[0,0])
 lo=np.array([0.,0.,0.]);hi=np.array([1.,1.,1.]);p=np.array([2.,3.,4.]);lower=np.linalg.norm(np.maximum(np.maximum(lo-p,p-hi),0));actual=np.linalg.norm(vv-p,axis=1);emit(dict(kind='AFTER_NEW_AABB_point_lower_bound_control_RETURN',lower=lower,vertex_distances=actual));assert (actual>=lower).all()
 manifest=read(F/'manifest.json');mesh=trimesh.load(F/manifest['objects'][0]['mesh'],force='mesh',process=False);V=np.asarray(mesh.vertices);index=SourceVertexBVH(V,leaf_size=32)
 with np.load(D/'dynamic-source-data.npz')as z:C0=z['C0'];K=z['K'];cam=z['original_camera_from_object'];RC=z['camera_from_world_linear']
 records=read(T/'all151-NEW-training-source-statistics.json');assert len(records)==151;total_points=0;uv_values=0;counts_checked=0;maxprojection=0.;maxpoint=0.;maxnn=0.;maxgrad=0.;spatial_pairs=0;tie_frames=[]
 for r in records:
  assert time.monotonic()-start<210,'NEWindependent training binding internal cap'
  f=r['frame'];emit(dict(kind='BEFORE_NEW_training_frame_independent_binding',frame=f,source_statistic_record_sha256=sha(T/f'frame-{f:03d}-NEW-training-source-binding.json')))
  with np.load(r['partition']['path'])as z:pixels=z['pixels'];indices=z['indices'];trainmask=z['training_mask'];domain=z['domain'];depth=z['selected_estimated_depth_m']
  yy,xx=np.indices(domain.shape);expected_domain=(xx+yy)%2==0;assert np.array_equal(domain,expected_domain)and not trainmask[~domain].any();assert np.all(pixels.sum(1)%2==0)and len(pixels)==r['training_depth_points'];eligible=r['eligible_EVEN_source_pixels'];expected=np.linspace(0,eligible-1,4096,dtype=int)if eligible>4096 else np.arange(eligible);assert np.array_equal(indices,expected)
  # Bind EVEN source labels/validity without evaluating any original fullpixel
  # source metric. Decoding cached data does not make odd labels training inputs.
  mask=cv2.imread(str(S/f'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/{f:06d}.png'),0)>127;assert np.array_equal(trainmask[domain],mask[domain])
  with np.load(S/f'fast-stereo-cpu-full-e037-v1/{f:06d}-depth.npz')as z:originaldepth=z['depth_m'];valid=z['left_right_consistent']
  flat=np.flatnonzero(valid&trainmask&np.isfinite(originaldepth)&(originaldepth>.1)&(originaldepth<2.5));chosen=flat[expected];independent_pixels=np.column_stack([chosen%domain.shape[1],chosen//domain.shape[1]]);assert np.array_equal(independent_pixels,pixels);assert np.array_equal(depth,originaldepth[pixels[:,1],pixels[:,0]])
  with np.load(r['raster']['path'])as z:P=z['predicted_training_mask'];uv=z['uv_float'];integer=z['uv_integer']
  independent_int=np.array([[min(20000,max(-20000,math.trunc(float(x))))for x in row]for row in uv],np.int32);assert np.array_equal(integer,independent_int);uv_values+=uv.size;homogeneous=np.column_stack([(K[0,0]*C0[f,:,0]+K[0,2]*C0[f,:,2])/C0[f,:,2],(K[1,1]*C0[f,:,1]+K[1,2]*C0[f,:,2])/C0[f,:,2]]);projection=float(abs(homogeneous-uv).max());assert projection<1e-9;maxprojection=max(maxprojection,projection)
  assert not P[~domain].any();I=int((P&trainmask).sum());U=int(((P|trainmask)&domain).sum());assert I==r['training_intersection']and U==r['training_union']and r['training_IoU']==I/max(1,U);counts_checked+=2
  with np.load(r['point_data']['path'])as z:points=z['points_C_m']
  reproj=np.column_stack([K[0,0]*points[:,0]/points[:,2]+K[0,2],K[1,1]*points[:,1]/points[:,2]+K[1,2]]);pe=float(abs(reproj-pixels).max());maxpoint=max(maxpoint,pe);assert pe<1e-9 and np.array_equal(points[:,2],depth.astype(float))
  with np.load(r['median_data']['path'])as z:local=z['point_local'];ids=z['nearest_vertex_indices'];first=z['nearest_vertex_distances'];second=z['second_vertex_distances'];central=z['central_point_indices'];centraldist=z['central_distances'];gradient=z['world_translation_local_gradient'];B=z['point_world_translation_linear']
  # Independent scalar-expanded body transform and BVH radius certificate.
  independent_local=np.column_stack([sum((points[:,j]-cam[f,j,3])*cam[f,j,k]for j in range(3))for k in range(3)]);assert abs(independent_local-local).max()<1e-12
  emit(dict(kind='BEFORE_NEW_independent_source_visual_BVH_radius_certificate',frame=f,point_count=len(local),source_visual_vertex_count=len(V),NOT_native_collision_mesh=True));independent=index.nearest_pair_certificate(local,second);nearest_return=arrays(f'frame-{f:03d}-independent-nearest-pair-RETURN.npz',first_two_distances=independent['first_two_distances'],original_query_first=first,original_query_second=second);emit(dict(kind='AFTER_NEW_independent_source_visual_BVH_RETURN',frame=f,actual_RETURN={k:v for k,v in independent.items()if k!='first_two_distances'},array_RETURN=nearest_return));assert independent['all_points_have_two_candidates'];error=float(abs(independent['first_two_distances']-np.c_[first,second]).max());assert error<1e-12;maxnn=max(maxnn,error);spatial_pairs+=independent['leaf_point_vertex_pairs'];direct=np.sqrt(np.einsum('ni,ni->n',local-V[ids],local-V[ids],optimize=False));assert abs(direct-first).max()<1e-12
  order=np.argsort(independent['first_two_distances'][:,0],kind='stable');n=len(local);rank=[(n-1)//2,n//2];true_median=float(np.mean(independent['first_two_distances'][order[rank],0]));assert abs(true_median-r['training_mesh_vertex_median_m'])<1e-12;assert np.array_equal(centraldist,first[central]);expected_gradient=np.zeros(3)
  for i in central:
   if first[i]>0:expected_gradient+=.5*(local[i]-V[ids[i]])@B/first[i]
  ge=float(abs(expected_gradient-gradient).max());maxgrad=max(maxgrad,ge);assert ge<1e-12;quality=float((I/max(1,U))*np.exp(-r['training_mesh_vertex_median_m']/.02));assert abs(quality-r['training_quality'])<1e-15;total_points+=len(points)
  if r['nearest_vertex_tie_count']or r['central_rank_tie']:tie_frames.append(f)
  result=dict(frame=f,training_pixel_point_count=len(points),projection_homogeneous_identity_error_px=projection,point_backprojection_error_px=pe,independent_NN_pair_error_m=error,independent_median= true_median,median_gradient_identity_error=ge,all_odd_labels_excluded=True,original_fullpixel_or_heldout_gate_NOT_evaluated=True,NN_and_rank_ties_retained_NOT_mandatory_observations=True);save(f'frame-{f:03d}-NEW-independent-training-binding.json',result);emit(dict(kind='AFTER_NEW_training_frame_independent_binding_RETURN',actual_RETURN=result))
 out=dict(schema='v2d.NEW-training-source-independent-bindings.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_BVH_analytic_controls=4,NEW_training_frames=151,training_points_bound=total_points,integer_quantizer_coordinate_bindings=uv_values,training_intersection_union_bindings=counts_checked,independent_source_visual_BVH_nearest_pairs=total_points,independent_leaf_point_vertex_pairs=spatial_pairs,maximum_source_projection_identity_error_px=maxprojection,maximum_point_backprojection_error_px=maxpoint,maximum_NN_pair_identity_error_m=maxnn,maximum_median_gradient_identity_error=maxgrad,NN_or_rank_tie_frames=tie_frames,all_original_inputs_and_candidate_gates_UNCHANGED=True,only_NEW_training_statistic_bound_NOT_pose_or_feasibility=True,no_old_original_scorer_cohort_renderer_source_fit_native_FK_solver_or_failed_point_replay=True,no_source_accuracy_or_trajectory_qualified=True,wall_seconds=time.monotonic()-start);save('report.json',out);emit(dict(kind='AFTER_NEW_training_source_independent_binding_suite',actual_RETURN=out));print(json.dumps(out))
