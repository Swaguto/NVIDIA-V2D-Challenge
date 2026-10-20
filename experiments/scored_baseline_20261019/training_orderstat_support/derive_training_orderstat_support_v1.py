"""NEW contracts and saved-data proxy inventory; no source/operator/solver replay."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,os,time
import numpy as np,trimesh
from training_orderstat_support_v1 import selected_proxy_prefix,proposed_support_rows,source_filter_decision
from thumb_conic_geometry_v1 import plain,journal

S=Path(__file__).resolve().parents[1];O=S/'e037-training-orderstat-support-v1'
T=S/'e037-training-source-constraints-v1';D=S/'e037-object-dynamic-runtime-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
 p=O/name;p.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def arrays(name,**values):
 p=O/name;np.savez_compressed(p,**values);p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def rat(x):
 q=F(x);return dict(numerator=str(q.numerator),denominator=str(q.denominator))

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(x):journal(log,x)
 scope=dict(schema='v2d.NEW-order-statistic-support-declaration.v1',
  source='SAVED151EVEN training arrays ONLY; no old failed point or fullpixel/heldout gates',
  method='ONE proxy-prefix inventory; NO solver/pose proposal/evaluation selected',
  necessity_scope='SUFFICIENT restricted construction for a training median upper bound; NOT a necessary source-family condition',
  count_proof='For n samples, lo=floor((n-1)/2),hi=floor(n/2). Nested saved-point subsets of lo+1 and hi+1 with proxy norms <= saved radii bound the two actual nearest-distance order statistics.',
  exact_model='nearest distance to all source vertices <= distance to any fixed source vertex. Rank and nearest correspondences may change.',
  original_candidate_tolerances_UNCHANGED=True,
  numerical_saved_F_proxy_defects_will_be_retained_NOT_clipped_or_accepted=True,
  hard_training_raster_median_quality_oracle_and_actual_p95_STILL_REQUIRED=True,
  original_fullpixel_heldout_and_actual_native_self_swept_motor_grasp_gates_SEPARATE=True)
 save('ONE-declared-proxy-prefix-scope-BEFORE-inventory.json',scope);emit(dict(kind='BEFORE_NEW_contracts',scope=scope))
 controls=[]
 def control(name,ret,test):
  emit(dict(kind='AFTER_NEW_analytic_RETURN',name=name,actual_RETURN=ret));save(name+'-RETURN.json',ret)
  assert test, name
  controls.append(name)
 # New cardinality proof for BOTH order-statistic ranks, including even averaging.
 for n in [5,6]:
  distances=np.array([.5,.125,.75,.25,.625,.375][:n]);p=selected_proxy_prefix(distances)
  u=np.full(n,2.);u[p['indices']]=p['radius_caps'];actual=np.minimum(u,np.full(n,1.5));med=float(np.median(actual))
  control('NEW-nested-prefix-two-rank-'+str(n),dict(n=n,proposal=p,synthetic_actual_nearest_upper=actual,median=med),
          med<=p['nominal_median'] and len(p['indices'])==n//2+1)
 # Fixed proxy vertices are upper bounds, not mandatory actual nearest vertices.
 V=np.array([[0.,0,0],[1.,0,0]]);P=np.array([.25,0,0]);shift=np.array([.5,0,0]);before=abs(V-P).sum(1);after=np.linalg.norm(V-(P+shift),axis=1)
 control('NEW-NN-switch-does-not-invalidate-proxy-upper-bound',dict(before=before,after=after,fixed_proxy=after[0],actual_nearest=after.min()),
         np.argmin(before)!=np.argmin(after)and after.min()<=after[0])
 # Independent scalar row expansion binds the SOC sign, coordinate order and adjoint.
 B=np.array([[0.,1.,0],[-1.,0,0],[0,0,-1.]]);c=np.array([[.25,.5,.75],[.125,.375,.625]]);r=np.array([1.,2.]);x=np.array([.03125,-.0625,.125]);A,b=proposed_support_rows(c,B,r)
 rows=b-np.einsum('nij,j->ni',A,x);expected=np.array([[r[k],*[c[k,i]+sum(B[i,j]*x[j]for j in range(3))for i in range(3)]]for k in range(2)])
 z=np.arange(8,dtype=float).reshape(2,4)/8;lhs=float(np.sum(z*(rows-b)));rhs=float(-sum(x[j]*sum(A[k,i,j]*z[k,i]for k in range(2)for i in range(4))for j in range(3)))
 control('NEW-proxy-SOC4-world-map-and-adjoint-sign',dict(A=A,b=b,x=x,actual_rows=rows,independent_rows=expected,lhs=lhs,rhs=rhs),np.array_equal(rows,expected)and lhs==rhs)
 # Zero radius demands the exact zero vector. It is not a normalized direction.
 AZ,bZ=proposed_support_rows(np.zeros((1,3)),np.eye(3),np.zeros(1));z0=bZ[0]-AZ[0]@np.zeros(3);z1=bZ[0]-AZ[0]@np.array([.125,0,0])
 control('NEW-zero-radius-proxy-is-exact-equality',dict(zero=z0,nonzero=z1),np.linalg.norm(z0[1:])==z0[0] and np.linalg.norm(z1[1:])>z1[0])
 # One bad source frame cannot be hidden by an aggregate merit or by missing queries.
 good=[dict(frame=0,passes_TRAINING_construction=True),dict(frame=1,passes_TRAINING_construction=True)];bad=[good[0],dict(frame=1,passes_TRAINING_construction=False)]
 decisions=[source_filter_decision(a,[0,1])for a in [good,bad,good[:1],[good[0],good[0]]]]
 control('NEW-hard-source-filter-completeness-and-rejection',dict(actual_returns=decisions),decisions[0]['accepted']and all(not d['accepted']for d in decisions[1:]))
 # Actual p95 is NOT preserved just by interpolating two feasible norm vectors.
 a=np.zeros(150);a[:7]=1;bq=np.zeros(150);bq[7:14]=1;m=(a+bq)/2
 quantiles=[float(np.percentile(q,95))for q in [a,bq,m]]
 control('NEW-p95-nonconvexity-requires-actual-trial-filter',dict(endpoint_and_midpoint_p95=quantiles,cap=.1,synthetic_scalar_speed_vectors=True),quantiles[0]<=.1 and quantiles[1]<=.1 and quantiles[2]>.1)
 assert len(controls)==7
 save('all7-NEW-orderstat-filter-contracts.json',dict(contracts=controls,old19controls_reused_BY_HASH_NOT_rerun=True))
 # New maps only: never re-query nearest vertices or recompute the old source metric.
 manifest=read(S/'e037-thumb-ray-independent-validation-v1/frozen/manifest.json')
 mesh=trimesh.load(S/'e037-thumb-ray-independent-validation-v1/frozen'/manifest['objects'][0]['mesh'],force='mesh',process=False)
 V=np.asarray(mesh.vertices,float);records=read(T/'all151-NEW-training-source-statistics.json')
 with np.load(D/'dynamic-source-data.npz')as z:Rcw=z['camera_from_world_linear'];poses=z['original_camera_from_object']
 total=0;nominal_defects=0;max_defect=F(0);frames=[]
 for frame in range(151):
  emit(dict(kind='BEFORE_NEW_proxy_prefix_map',frame=frame,unassigned_continuous_world_delta=True))
  saved=T/f'frame-{frame:03d}-NEW-training-median-RETURN.npz'
  with np.load(saved)as z:
   distances=z['nearest_vertex_distances'];ids=z['nearest_vertex_indices'];local=z['point_local']
  p=selected_proxy_prefix(distances);subset=p['indices'];caps=p['radius_caps'];offsets=local[subset]-V[ids[subset]]
  B=-np.einsum('ji,jk->ik',poses[frame,:3,:3],Rcw[frame],optimize=False)
  A,bb=proposed_support_rows(offsets,B,caps)
  ret=arrays(f'frame-{frame:03d}-NEW-proxy-SOC-maps-RETURN.npz',selected_training_point_indices=subset,
   fixed_proxy_vertex_indices=ids[subset],nominal_radius_m=caps,local_offsets_m=offsets,
   world_translation_linear=B,canonical_A=A,canonical_b=bb)
  emit(dict(kind='AFTER_NEW_proxy_map_constructor_RETURN',frame=frame,arrays=ret,actual_prefix_RETURN=p))
  # Independent exact entry/packing signs and nominal radius defects. This is
  # NOT an old nearest-distance/PSD/source audit and never calls a KD tree.
  assert np.array_equal(subset,np.argsort(distances,kind='stable')[:len(distances)//2+1])
  assert len(set(map(int,subset)))==len(subset)
  assert int(np.count_nonzero(caps<=p['lower_radius']))>=p['lower_count']
  assert np.count_nonzero(caps<=p['upper_radius'])==p['upper_count']
  assert p['nominal_median']==records[frame]['training_mesh_vertex_median_m']
  map_errors=[]
  for i in range(3):
   for j in range(3):
    direct=-sum(F(float(poses[frame,k,i]))*F(float(Rcw[frame,k,j]))for k in range(3))
    # Preserve exact upstream-vs-rounded map error as a separate model binding.
    err=F(float(B[i,j]))-direct;map_errors.append(dict(row=i,column=j,exact_error=rat(err)))
  save(f'frame-{frame:03d}-NEW-exact-affine-map-errors.json',map_errors)
  emit(dict(kind='AFTER_NEW_exact_world_map_error_RETURN',frame=frame,actual_RETURN=map_errors))
  assert all(abs(F(int(e['exact_error']['numerator']),int(e['exact_error']['denominator'])))<=F(1,10**14)for e in map_errors)
  local_defects=0;local_max=F(0)
  for k,(off,radius)in enumerate(zip(offsets,caps,strict=True)):
   assert bb[k,0]==radius and np.array_equal(bb[k,1:],off)and np.array_equal(A[k,1:],-B)
   sq=sum(F(float(v))**2 for v in off);diff=sq-F(float(radius))**2
   if diff>0:local_defects+=1;local_max=max(local_max,diff)
  nominal_defects+=local_defects;max_defect=max(max_defect,local_max);total+=len(subset)
  binding=dict(frame=frame,scope='NEW sufficient source-median construction maps ONLY',
   saved_training_RETURN=str(saved),saved_training_RETURN_sha256=sha(saved),maps=ret,
   selected_proxy_count=len(subset),lower_rank=int(p['lower_rank']),upper_rank=int(p['upper_rank']),
   lower_radius_m=float(p['lower_radius']),upper_radius_m=float(p['upper_radius']),
   required_lower_count=int(p['lower_count']),required_upper_count=int(p['upper_count']),
   nominal_median_m=float(p['nominal_median']),unassigned_world_translation_columns=3,
   nominal_F_exact_squared_norm_exceeds_saved_radius_count=local_defects,
   maximum_positive_exact_squared_defect=rat(local_max),
   defects_UNALTERED_no_guard_or_radius_repair_selected=True,
   initial_F_feasibility_for_this_exact_proxy_model_NOT_claimed=True,
   no_actual_nonzero_episode_point_or_solver_call=True)
  save(f'frame-{frame:03d}-NEW-independent-proxy-binding.json',binding);emit(dict(kind='AFTER_NEW_proxy_binding',actual_RETURN=binding));frames.append(binding)
 report=dict(schema='v2d.NEW-training-orderstat-support.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  NEW_analytic_contracts=7,frames=151,new_proxy_SOC4_cones=total,unassigned_world_translation_columns=453,
  canonical_rows=4*total,nominal_F_exact_squared_radius_defects=nominal_defects,
  maximum_positive_exact_squared_defect=rat(max_defect),
  all_maps_and_raw_saved_values_UNALTERED=True,no_old_training_dataset_or_NN_or_source_or_control_replay=True,
  no_actual_nonzero_episode_pose_solver_native_physics_GPU_PPO_or_trajectory_qualification=True,
  sufficient_restricted_construction_NOT_original_source_gate_or_global_exclusion=True,
  original_fullpixel_source_and_actual_training_oracle_actual_p95_and_full_physical_gates_STILL_REQUIRED=True)
 save('all151-NEW-orderstat-proxy-bindings.json',frames);save('report.json',report);emit(dict(kind='AFTER_NEW_proxy_inventory_RETURN',actual_RETURN=report));print(json.dumps(report))
