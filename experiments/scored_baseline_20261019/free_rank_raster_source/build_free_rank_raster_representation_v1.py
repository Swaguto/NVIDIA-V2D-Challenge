"""ONE factored free-rank/nearest-union/raster formulation, not a solver.

Only saved training data and original source definitions enter. No cached F
raster/nearest metric, rejected vector, source scorer or physical model is run.
"""
from pathlib import Path
from fractions import Fraction as Q
import datetime,hashlib,json,os
import numpy as np,trimesh
from exact_training_source_logic_v1 import free_rank_formula

S=Path(__file__).resolve().parents[1];O=S/'e037-free-rank-raster-representation-v1'
T=S/'e037-training-source-constraints-v1';D=S/'e037-object-dynamic-runtime-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pack(v):
 if isinstance(v,Q):return dict(numerator=str(v.numerator),denominator=str(v.denominator))
 if isinstance(v,np.ndarray):return pack(v.tolist())
 if isinstance(v,np.generic):return pack(v.item())
 if isinstance(v,dict):return {str(k):pack(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [pack(x)for x in v]
 return v
def save(n,v):
 p=O/n;p.write_text(json.dumps(pack(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def pointer(p):return dict(path=str(p),sha256=sha(p))
def q(x):return Q(float(x))

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):
  log.write(json.dumps(pack(v),allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
 declaration=dict(schema='v2d.NEW-free-rank-raster-declaration.v1',
  single_construction='factored logical source constraints; NO search, solver, pose proposal or actual source evaluation',
  inputs='saved151EVEN training arrays and ORIGINAL source visual geometry/camera only',
  all_nearest_vertices_and_rank_subsets_FREE=True,
  all_integer_projected_vertex_cells_FREE_with_original_clip_truncate_range=True,
  no_initial_F_prefix_axis_face_or_rejected_vector_used=True,
  object_rotations_FIXED_F_for_THIS_formulation_only=True,
  world_translations_continuous_UNASSIGNED=True,
  no_new_constructor_box_radius_threshold_or_acceptance_tolerance=True,
  original_fullpixel_heldout_POSTchecks_and_all6objectcaps_STILL_REQUIRED=True,
  stronger_training_nonregression_NOT_original_source_family_necessity=True,
  exact_IDEAL_saved_binary_input_polynomials_NOT_full_binary_operator_or_source_truth=True,
  KD_norm_roundoff_and_raster_binary_equivalence_boundary_obligations_UNRESOLVED=True)
 save('ONE-free-rank-raster-construction-BEFORE.json',declaration);emit(dict(kind='BEFORE_NEW_formulation',declaration=declaration))
 # Template syntax is a portable logical representation, with named input arrays.
 # It does not allocate billions of per-point/per-vertex polynomial copies.
 templates=dict(
  point_local=dict(expression='l[p,k]=sum_i((P[p,i]-t[i])*R[i,k])+sum_j(B[k,j]*delta[j])',B='-transpose(R)*Rcw',arithmetic='exact rational SAVED binary coefficients in IDEAL branch'),
  nearest_event=dict(expression='Z[p,r] <=> OR_{v=0..Vcount-1}(sum_k(l[p,k]-V[v,k])^2 <= r^2)',radius='nonnegative FREE threshold a or b; all original visual vertices',numerical_branch='ORIGINAL KD/norm output is retained as an explicit opaque binary64 operator, not assumed equal to IDEAL polynomial'),
  median_even=dict(variables=['a>=0','b>=a'],constraints=['sum_p Z[p,a]>=n/2','sum_p Z[p,b]>=n/2+1','a+b<=2*m0'],actual_operator_constraint='fl(fl(a+b)/2)<=m0 with ordered binary64 thresholds for ORIGINAL numerical-distance branch'),
  median_odd=dict(variables=['a>=0'],constraints=['sum_p Z[p,a]>=floor(n/2)+1','a<=m0']),
  camera=dict(expression='C[v,i]=sum_k(R[i,k]*V[v,k])+t[i]+sum_j(Rcw[i,j]*delta[j])',strict_depth='ALL C[v,2]>.01m; center>.05m remains tracker POSTcondition'),
  projection=dict(Nu='fx*C[v,0]+cx*C[v,2]',Nv='fy*C[v,1]+cy*C[v,2]',Z='C[v,2]',integer_q_range=[-20000,20000],
   free_quantizer_cells=[dict(condition='0<q<20000',lower='q*Z<=N',upper='N<(q+1)*Z'),
    dict(condition='-20000<q<0',lower='(q-1)*Z<N',upper='N<=q*Z'),
    dict(condition='q==0',lower='-Z<N',upper='N<Z'),
    dict(condition='q==20000',lower='20000*Z<=N',upper=None),
    dict(condition='q==-20000',lower=None,upper='N<=-20000*Z')],
   binary_error='Original projection arithmetic or justified outward interval required; any crossed quantizer boundary UNRESOLVED, no new pixel tolerance'),
  raster=dict(expression='P[y,x]=OR_{face in ALL_ORIGINAL_visual_faces} LINE8_SHIFT0_TRIANGLE(q[face],x,y)',
   triangle_program='exact_training_source_logic_v1.triangle_line8_integer',
   library_reference='OpenCV4.11.0 drawing.cpp+imgproc.hpp; tag matching/synthetic contracts do not prove full installed binary equivalence',
   occlusion='FULL triangle OR, no front-surface or depth-buffer approximation',domain='EXISTING saved EVEN domain, no partition/mask edits'),
  IoU=dict(I='sum_{(x,y) in D}(P[y,x] AND M[y,x])',U='Mcount+sum_{(x,y) in D}(P[y,x] AND NOT M[y,x])',
   condition='I*U0-I0*U>=0 with U>0',empty='UNRESOLVED; do not supply division by zero',integer_counts_exact=True),
  quality=dict(condition='actual training IoU*exp(-actual training median/.02)>=saved training quality',
   ideal_consequence='IoU>=saved training IoU and median<=saved training median imply REAL quality nonregression',
   numeric='actual unchanged binary64 quality POSTcheck required'),
  motion=dict(scope='all151 clocks/actual timestamps; all6ORIGINAL object caps, not only maximum linear cones',
   saved_maps=str(D/'dynamic-motion-linear-map.npz'),actual_p95='ORIGINAL percentile operator, not assumed convex',
   angular='fixed rotations only; original angular speed/acceleration/p95 still verified in candidate gate'),
  uncertainty=dict(no_training_missing_points_created_or_deleted=True,unknown_projection_or_KD_boundary_NOT_FALSE=True,
   human_or_object_source_accuracy_NOT_established=True,original95frame50misses_136robotmissing_31of32heldout_failures_UNCHANGED=True))
 template_ref=save('NEW-factored-logical-templates-RETURN.json',templates)
 emit(dict(kind='AFTER_NEW_template_constructor_RETURN',templates=template_ref))
 frozen=S/'e037-thumb-ray-independent-validation-v1/frozen';manifest=read(frozen/'manifest.json')
 mesh_path=frozen/manifest['objects'][0]['mesh'];mesh=trimesh.load(mesh_path,force='mesh',process=False)
 V=np.asarray(mesh.vertices,float);faces=np.asarray(mesh.faces,int)
 source_refs=dict(mesh=pointer(mesh_path),manifest=pointer(frozen/'manifest.json'),
  training_statistics=pointer(T/'all151-NEW-training-source-statistics.json'),
  dynamic_data=pointer(D/'dynamic-source-data.npz'),
  saved_all6_motion_caps=pointer(D/'runtime-config-and-original-authority.json'),
  saved_299_motion_maps=pointer(D/'dynamic-motion-linear-map.npz'),
  saved_299_motion_constants=pointer(D/'dynamic-motion-constants.npy'),
  old19training_controls=pointer(S/'compute/training-source-checkpoint-v1.json'),
  old7proxy_controls_and_defects=pointer(S/'compute/training-orderstat-checkpoint-v1.json'),
  old10roundoff_callback_dual_controls=pointer(S/'compute/hard-training-conic-checkpoint-v1.json'),
  NEW15contracts=pointer(S/'e037-exact-training-source-contracts-v1/report.json'),
  NEWreference_receipt=pointer(S/'e037-exact-training-source-reference-v1/reference-receipt.json'))
 save('NEW-source-and-prior-evidence-pointers-BY-HASH.json',source_refs)
 with np.load(D/'dynamic-source-data.npz')as z:
  RC=z['camera_from_world_linear'];poses=z['original_camera_from_object'];K=z['K'];times=z['timestamps_s'];frames=z['frame_indices']
 stats=read(T/'all151-NEW-training-source-statistics.json');records=[]
 totals=dict(points=0,free_thresholds=0,point_vertex_distance_predicates=0,
  radius_point_vertex_event_instances=0,free_integer_projected_coordinates=0,
  triangle_raster_program_instances=0,EVEN_boolean_pixel_expressions=0)
 for f in range(151):
  emit(dict(kind='BEFORE_NEW_frame_formula',frame=f,all_choices_FREE=True,actual_pose_NOT_assigned=True))
  row=stats[f]
  # Cached source arrays are referenced, not resampled/re-lifted/rasterized or NN-queried.
  partition=Path(row['partition']['path']);point_data=Path(row['point_data']['path'])
  with np.load(partition)as z:n=len(z['pixels']);shape=list(z['domain'].shape)
  R=poses[f,:3,:3];t=poses[f,:3,3]
  rq=[[q(x)for x in r]for r in R];wq=[[q(x)for x in r]for r in RC[f]];tq=[q(x)for x in t]
  B=[[-sum(rq[i][k]*wq[i][j]for i in range(3))for j in range(3)]for k in range(3)]
  local_shift=[-sum(tq[i]*rq[i][k]for i in range(3))for k in range(3)]
  # Exact coefficients for (Nu,Nv,Z) in (Vx,Vy,Vz,1,delta_x,delta_y,delta_z).
  projection=[]
  for h in [[q(K[0,0]),Q(0),q(K[0,2])],[Q(0),q(K[1,1]),q(K[1,2])],[Q(0),Q(0),Q(1)]]:
   projection.append([sum(h[i]*rq[i][k]for i in range(3))for k in range(3)]+
    [sum(h[i]*tq[i]for i in range(3))]+[sum(h[i]*wq[i][j]for i in range(3))for j in range(3)])
  rank=free_rank_formula(n,row['training_mesh_vertex_median_m'])
  factors=save(f'frame-{f:03d}-NEW-exact-factor-coefficients-RETURN.json',dict(
   R=rq,t=tq,Rcw=wq,B=B,point_local_constant_shift=local_shift,
   projection_Nu_Nv_Z_coefficients=projection,coordinate_order=['Vx','Vy','Vz','1','world_dx','world_dy','world_dz'],
   local_coefficients_BIND_exact_ideal_BINARYinputs_NOT_all_floating_intermediates=True))
  threshold_count=len(rank['threshold_symbols']);pixels=shape[0]*shape[1]//2
  r=dict(frame=f,source_frame_index=int(frames[f]),timestamp_s=float(times[f]),
   coordinate_symbols=[f'world_delta_{f}_{k}'for k in range(3)],all_continuous_UNASSIGNED=True,
   threshold_symbols=[f'{a}_{f}'for a in rank['threshold_symbols']],rank_formula=rank,
   training_point_array=pointer(point_data),training_partition=pointer(partition),
   original_mask_depth_provenance=dict(mask_sha256=row['source_mask_sha256'],depth_sha256=row['source_depth_sha256']),
   saved_training_only_I0=int(row['training_intersection']),saved_training_only_U0=int(row['training_union']),
   saved_training_only_m0=q(row['training_mesh_vertex_median_m']),saved_training_only_quality=q(row['training_quality']),
   source_visual_vertices=len(V),source_visual_faces=len(faces),all_vertices_and_faces_INCLUDED=True,
   nearest_vertex_choices_and_rank_subsets='FREE union/cardinality, no fixed NN or point-prefix map',
   integer_vertex_projection_choices='FREE clipped/truncated cells, not fixed to saved raster',
   factor_coefficients=factors,template_ref=template_ref,
   counts=dict(points=n,point_vertex_distance_predicates=n*len(V),
    radius_point_vertex_event_instances=n*len(V)*threshold_count,
    free_thresholds=threshold_count,free_integer_projected_coordinates=2*len(V),
    triangle_raster_program_instances=len(faces),EVEN_boolean_pixel_expressions=pixels),
   expanded_predicates_NOT_materialized=True,raster_and_NN_actual_episode_calls=0,
   unresolved=['KD/norm original floating-operator boundary equivalence',
    'projection outward error or exact original numerical execution at crossed cells',
    'installed raster binary full equivalence beyond NEW synthetic controls',
    'no source accuracy/common physical trajectory certified'])
  rec_ref=save(f'frame-{f:03d}-NEW-free-source-formula-RETURN.json',r)
  emit(dict(kind='AFTER_NEW_frame_constructor_RETURN',frame=f,formula=rec_ref));records.append(r)
  for k in totals:totals[k]+=r['counts'][k]
 save('all151-NEW-free-source-formulas-RETURN.json',records)
 report=dict(schema='v2d.NEW-free-rank-raster-representation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  frames=151,continuous_world_translation_symbols=453,**totals,
  NEW_derived_coefficients_per_frame=33,all_NEW_derived_factor_coefficients=151*33,
  source_geometry_vertices=len(V),source_geometry_faces=len(faces),
  status='FORMAL_IDEAL_EQUIVALENCE_AND_EXPLICIT_NUMERICAL_BOUNDARY_OBLIGATIONS',
  no_solver_pose_NN_raster_source_or_native_episode_evaluation=True,
  no_old_source_controls_data_producer_audit_or_failed_point_replay=True,
  every_choice_unassigned_no_new_moment_command_or_reference=True,
  free_rank_vertex_projection_choices_not_fixed_as_observations=True,
  exact_real_model_NOT_original_numerical_operator_accuracy_or_qualification=True,
  F_components_missing_source_model_parameters_gates_reserved_episodes_UNCHANGED=True)
 save('report.json',report);emit(dict(kind='AFTER_NEW_formulation_RETURN',actual_RETURN=report));print(json.dumps(report))
