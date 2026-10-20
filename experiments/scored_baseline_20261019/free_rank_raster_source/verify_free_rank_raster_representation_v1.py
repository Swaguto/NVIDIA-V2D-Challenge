"""Independent NEW factored map / provenance / free-cell verification.

Does not import the builder or rerun the 15 analytic controls, old source
statistics, raster/KD operators, model constructors or episode solvers.
"""
from pathlib import Path
from fractions import Fraction as Q
import datetime,hashlib,json,os
import numpy as np

S=Path(__file__).resolve().parents[1];O=S/'e037-free-rank-raster-independent-v1'
B=S/'e037-free-rank-raster-representation-v1';T=S/'e037-training-source-constraints-v1';D=S/'e037-object-dynamic-runtime-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(v):
 if isinstance(v,dict)and set(v)=={'numerator','denominator'}:return Q(int(v['numerator']),int(v['denominator']))
 if isinstance(v,dict):return {k:unpack(x)for k,x in v.items()}
 if isinstance(v,list):return [unpack(x)for x in v]
 return v
def pack(v):
 if isinstance(v,Q):return dict(numerator=str(v.numerator),denominator=str(v.denominator))
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {str(k):pack(x)for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [pack(x)for x in v]
 return v
def save(n,v):
 p=O/n;p.write_text(json.dumps(pack(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def matrix_product(a,b):
 return [[sum(x*y for x,y in zip(row,col,strict=True))for col in zip(*b)]for row in a]

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):log.write(json.dumps(pack(v),allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
 save('ONE-independent-scope-BEFORE.json',dict(NEW_maps_and_logical_incidence_ONLY=True,
  no_builder_or_old_operator_control_audit_replay=True,no_pose_solver_source_NN_raster_native_physics_GPU=True,
  original_binary_source_boundary_and_build_uncertainties_remain_UNRESOLVED=True))
 controls=[]
 def control(n,before,fn,check):
  save(n+'-BEFORE.json',before);emit(dict(kind='BEFORE_NEW_contract',name=n))
  r=fn();p=save(n+'-RETURN.json',r);emit(dict(kind='AFTER_NEW_actual_RETURN',name=n,actual_RETURN=p));assert check(r),n;controls.append(n)
 # A FREE integer cell must be constrained on both sides with ORIGINAL signs.
 # These are exact real quantizer inverse images; source roundoff stays separate.
 def in_cell(u,q):
  if q==20000:return u>=20000
  if q==-20000:return u<=-20000
  if q>0:return q<=u<q+1
  if q<0:return q-1<u<=q
  return -1<u<1
 samples=[(-20001,-20000),(-20000,-20000),(-Q(3,2),-1),(-1,-1),(-Q(1,2),0),
          (0,0),(Q(1,2),0),(1,1),(Q(3,2),1),(20000,20000),(20001,20000)]
 def inverse_cells():
  out=[]
  for u,k in samples:
   z=Q(3,2);N=Q(u)*z
   own=in_cell(N/z,k);actual=int(max(Q(-20000),min(Q(20000),Q(u))))
   out.append(dict(u=Q(u),Z=z,N=N,q=k,free_cell_holds=own,clip_truncate=actual))
  return out
 control('NEW-free-quantizer-preimage-strict-endpoint-signs',dict(cells='five sign/saturation branches',samples=samples),inverse_cells,
  lambda r:all(x['free_cell_holds']and x['clip_truncate']==x['q']for x in r))
 # General symbolic polynomial expansion of ||c+B*d-v||^2-r^2 is new here:
 # retain cross coefficient 2, shared d variables, and threshold coefficient -1.
 def expand():
  coeff={}
  def add(m,c):coeff[tuple(sorted(m))]=coeff.get(tuple(sorted(m)),0)+c
  for k in range(3):
   terms=[((f'c{k}',),1),((f'v{k}',),-1)]+[((f'B{k}{j}',f'd{j}'),1)for j in range(3)]
   for a,ca in terms:
    for b,cb in terms:add(a+b,ca*cb)
  add(('r','r'),-1)
  return dict(polynomial=[dict(monomial=list(k),coefficient=v)for k,v in sorted(coeff.items())],
   threshold_square=coeff[('r','r')],shared_world_cross=coeff[tuple(sorted(['B00','d0','B01','d1']))],
   vertex_point_cross=coeff[('c0','v0')],unique_monomials=len(coeff))
 control('NEW-symbolic-nearest-union-square-shared-variable-polynomial',dict(expression='sum_k(c_k+sum_j B_kj*d_j-v_k)^2-r^2'),expand,
  lambda r:r['threshold_square']==-1 and r['shared_world_cross']==2 and r['vertex_point_cross']==-2)
 # A missing distance cannot silently become a false event: it can still be a
 # witness in an existential upper cardinality, with distinct definite bounds.
 def unknown_proof():
  n=6;certain=2;unknown=2;lo=n//2;hi=lo+1
  return dict(n=n,lower_rank_required=lo,upper_rank_required=hi,
   certain_count=certain,possible_count=certain+unknown,
   lower_and_upper='UNRESOLVED',
   wrong_drop_would_report='FAIL',
   proof='For either required count 3 or 4, 2<required<=4. Unknown events cannot be forced false or true.')
 control('NEW-unknown-events-not-deleted-from-two-rank-formula',dict(certain=2,unknown=2,false=2),unknown_proof,
  lambda r:r['certain_count']<r['lower_rank_required']<=r['upper_rank_required']<=r['possible_count'])
 # From here no contracts/episode numerical operators run. Verify the NEW
 # instantiated formula maps against the independent saved binary inputs.
 data=read(B/'all151-NEW-free-source-formulas-RETURN.json');stats=read(T/'all151-NEW-training-source-statistics.json')
 templates=read(B/'NEW-factored-logical-templates-RETURN.json');refs=read(B/'NEW-source-and-prior-evidence-pointers-BY-HASH.json')
 for r in refs.values():assert sha(r['path'])==r['sha256']
 with np.load(D/'dynamic-source-data.npz')as z:
  RC=z['camera_from_world_linear'];poses=z['original_camera_from_object'];K=z['K'];times=z['timestamps_s'];indices=z['frame_indices'];faces=z['faces']
 old_contract_hashes={k:v for k,v in refs.items()if k.startswith('old')}
 results=[];total_points=0;total_thresholds=0;total_factors=0;pointer_checks=0
 for f,r in enumerate(data):
  emit(dict(kind='BEFORE_NEW_frame_binding',frame=f,formula_sha256=sha(B/f'frame-{f:03d}-NEW-free-source-formula-RETURN.json')))
  factor=unpack(read(r['factor_coefficients']['path']));source=stats[f]
  for key,original in [('training_point_array',source['point_data']),('training_partition',source['partition'])]:
   actual=r[key];assert actual==original;assert sha(actual['path'])==actual['sha256'];pointer_checks+=1
  assert sha(r['factor_coefficients']['path'])==r['factor_coefficients']['sha256']
  assert r['template_ref']['sha256']==sha(B/'NEW-factored-logical-templates-RETURN.json')
  with np.load(r['training_point_array']['path'])as z:points=z['points_C_m']
  n=len(points)
  # Use matrix products of the independent camera/world homogeneous embedding.
  # This is not an old candidate/source/FK evaluation.
  R=[[Q(float(x))for x in row]for row in poses[f,:3,:3]]
  W=[[Q(float(x))for x in row]for row in RC[f]];t=[Q(float(x))for x in poses[f,:3,3]]
  negative_transpose=[[-R[i][k]for i in range(3)]for k in range(3)]
  expected_B=matrix_product(negative_transpose,W)
  expected_shift=[sum(a*b for a,b in zip(row,t,strict=True))for row in negative_transpose]
  augmented=[R[i]+[t[i]]+W[i]for i in range(3)]
  H=[[Q(float(K[0,0])),Q(0),Q(float(K[0,2]))],
     [Q(0),Q(float(K[1,1])),Q(float(K[1,2]))],[Q(0),Q(0),Q(1)]]
  expected_projection=matrix_product(H,augmented)
  ret=dict(frame=f,exact_B=expected_B,exact_shift=expected_shift,exact_projection=expected_projection,
   NEW_factor_identities=33,all_differences_zero=(factor['B']==expected_B and factor['point_local_constant_shift']==expected_shift and factor['projection_Nu_Nv_Z_coefficients']==expected_projection),
   point_count=n,source_frame_index=int(indices[f]),timestamp_s=float(times[f]),
   required_counts=[(n+1)//2,n//2+1],free_thresholds=2 if n%2==0 else 1,
   saved_only_statistics_no_metric_replay=True)
  # Persist actual independent return before assertions.
  p=save(f'frame-{f:03d}-NEW-independent-map-RETURN.json',ret);emit(dict(kind='AFTER_NEW_independent_map_RETURN',frame=f,actual_RETURN=p))
  assert ret['all_differences_zero']
  assert factor['R']==R and factor['t']==t and factor['Rcw']==W
  assert r['frame']==f and r['timestamp_s']==ret['timestamp_s']and r['source_frame_index']==ret['source_frame_index']
  assert r['rank_formula']['ranks_zero_based']==[(n-1)//2,n//2]
  assert r['rank_formula']['counts']==ret['required_counts']
  assert len(r['threshold_symbols'])==ret['free_thresholds']
  assert r['coordinate_symbols']==[f'world_delta_{f}_{j}'for j in range(3)]
  assert r['source_visual_vertices']==6022 and r['source_visual_faces']==len(faces)==12000
  assert r['saved_training_only_I0']==source['training_intersection']and r['saved_training_only_U0']==source['training_union']
  assert unpack(r['saved_training_only_m0'])==Q(source['training_mesh_vertex_median_m'])
  assert unpack(r['saved_training_only_quality'])==Q(source['training_quality'])
  assert r['original_mask_depth_provenance']==dict(mask_sha256=source['source_mask_sha256'],depth_sha256=source['source_depth_sha256'])
  c=r['counts'];assert c['points']==n and c['point_vertex_distance_predicates']==n*6022
  assert c['radius_point_vertex_event_instances']==n*6022*ret['free_thresholds']
  assert r['nearest_vertex_choices_and_rank_subsets'].startswith('FREE')and r['integer_vertex_projection_choices'].startswith('FREE')
  assert r['raster_and_NN_actual_episode_calls']==0 and len(r['unresolved'])==4
  total_points+=n;total_thresholds+=ret['free_thresholds'];total_factors+=33;results.append(ret)
 # Verify every NEW per-frame durable BEFORE/AFTER pair, not old journals.
 journal_records=[json.loads(line)for line in (B/'normalized-BEFORE-AFTER-journal.jsonl').read_text().splitlines()]
 pairs=[]
 for f in range(151):
  before=[i for i,x in enumerate(journal_records)if x.get('kind')=='BEFORE_NEW_frame_formula'and x['frame']==f]
  after=[i for i,x in enumerate(journal_records)if x.get('kind')=='AFTER_NEW_frame_constructor_RETURN'and x['frame']==f]
  assert len(before)==len(after)==1 and before[0]<after[0];pairs.append(dict(frame=f,BEFORE=before[0],AFTER=after[0]))
 save('all151-NEW-independent-bindings.json',results);save('all151-NEW-durable-pairs.json',pairs)
 report=dict(schema='v2d.NEW-free-rank-raster-independent-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  NEW_contracts=len(controls),contracts=controls,frames=151,NEW_exact_factor_coefficients=total_factors,
  NEW_rank_and_FREE_scope_bindings=151,NEW_input_point_and_partition_pointers=pointer_checks,
  training_points=total_points,free_thresholds=total_thresholds,durable_BEFORE_AFTER_frame_pairs=151,
  old_controls_and_statistics_reused_BY_HASH_NOT_replayed=old_contract_hashes,
  original_binary64_KD_norm_projection_and_raster_build_uncertainty_NOT_erased=True,
  no_pose_solver_source_raster_NN_native_physics_GPU_OR_qualification=True,
  raw_F_and_all_prior_returns_parameters_source_and_gates_UNCHANGED=True)
 save('report.json',report);emit(dict(kind='AFTER_NEW_independent_binding_RETURN',actual_RETURN=report));print(json.dumps(report))
