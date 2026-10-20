"""ONE NEW exact cache/affine alias and conditional camera-bound construction.

No camera projection, raster, KD source statistic, pose or solver is evaluated.
The all-vertex arithmetic binds a NEW distinction in the free-source formula;
it does not replay an old source/geometry/native closure audit.
"""
from pathlib import Path
from fractions import Fraction as Q
import ast,datetime,hashlib,json,os
import numpy as np,trimesh
from source_numerical_enclosures_v1 import common_dyadic,cached_camera_alias,cache_error_affine_coefficients

S=Path(__file__).resolve().parents[1];C=S/'compute';O=S/'e037-source-numerical-bindings-v1'
D=S/'e037-object-dynamic-runtime-v1';R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pack(v):
 if isinstance(v,Q):return dict(numerator=str(v.numerator),denominator=str(v.denominator))
 if isinstance(v,np.ndarray):return pack(v.tolist())
 if isinstance(v,np.generic):return pack(v.item())
 if isinstance(v,dict):return {str(k):pack(x)for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [pack(x)for x in v]
 return v
def save(n,v):
 p=O/n;p.write_text(json.dumps(pack(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def pointer(p):return dict(path=str(p),sha256=sha(p))

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):log.write(json.dumps(pack(v),allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
 scope=dict(schema='v2d.NEW-source-numerical-binding-declaration.v1',
  NEW_cache_model_alias='exact saved C0 minus exact(V*R^T+t) for all source visual vertices',
  camera_bound='error_i(H)=alpha_i0+sum_j alpha_ij H_j, with ALL H_j UNASSIGNED',
  conditional_scope='finite RN binary64 cached-camera operator C0+einsum(Rcw,delta)',
  no_new_algorithmic_or_original_translation_box_selected=True,
  ORIGINAL_k1_fullpixel_matmul_and_training_k2_einsum_cache_are_distinct=True,
  no_NN_raster_projection_old_control_or_audit_reexecution=True,
  no_new_pose_solver_physics_GPU_PPO=True,source_accuracy_NOT_established=True)
 save('ONE-NEW-numerical-binding-scope-BEFORE.json',scope);emit(dict(kind='BEFORE_NEW_bindings',scope=scope))
 original=R/'scripts/track3/select_pose_hypothesis.py';training=C/'object_training_source_semantics_v1.py';oracle=C/'object_training_constraint_oracle_v1.py'
 def calls(path,function):
  tree=ast.parse(path.read_text());fn=next(x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name==function)
  found=[]
  for node in ast.walk(fn):
   if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)and node.func.attr=='query':
    found.append(dict(line=node.lineno,expression=ast.unparse(node),keyword_AST={k.arg:ast.dump(k.value)for k in node.keywords}))
  return found
 identities=dict(original_fullpixel=dict(source=pointer(original),query_calls=calls(original,'observed_fit'),
  k='omitted; public1.16.3query signature default1',point_local_operator='NumPy matmul',camera_operator='vertices@R.T+t',source_parities='original BOTH'),
  EVEN_training=dict(source=pointer(training),query_calls=calls(training,'vertex_median_semantics'),
   k=2,point_local_operator='einsum optimize=False',camera_operator='SAVED C0 + einsum(Rcw,delta)',source_parities='saved EVEN training only'),
  training_oracle=pointer(oracle),
  source_public_reference=pointer(S/'e037-source-numerical-reference-v1/reference-receipt.json'),
  original_fullpixel_tolerance=1e-10,training_nominal_restriction_NOT_original_tolerance=True,
  no_k1_k2_build_roundoff_search_equivalence_claim=True,
  no_original_matmul_training_einsum_numeric_identity_claim=True)
 save('NEW-actual-operator-scope-bindings-RETURN.json',identities);emit(dict(kind='AFTER_NEW_operator_scope_RETURN',actual_RETURN=identities))
 Fdir=S/'e037-thumb-ray-independent-validation-v1/frozen';manifest=read(Fdir/'manifest.json');mesh_path=Fdir/manifest['objects'][0]['mesh']
 V=np.asarray(trimesh.load(mesh_path,force='mesh',process=False).vertices,float);Vint,EV=common_dyadic(V)
 with np.load(D/'dynamic-source-data.npz')as z:poses=z['original_camera_from_object'];RC=z['camera_from_world_linear'];C0=z['C0'];K=z['K'];frames=z['frame_indices'];times=z['timestamps_s']
 inputs=dict(original_visual_mesh=pointer(mesh_path),saved_source_data=pointer(D/'dynamic-source-data.npz'),
  old_free_formula=pointer(S/'e037-free-rank-raster-representation-v1/all151-NEW-free-source-formulas-RETURN.json'),
  old_exact_factors_and_unknowns=pointer(S/'e037-free-rank-raster-independent-v1/report.json'),
  old_roundoff_constants_and_proof=pointer(S/'e037-training-proxy-roundoff-v1/report.json'),
  NEW10contracts=pointer(S/'e037-source-numerical-contracts-v1/report.json'))
 save('NEW-input-provenance-BY-HASH.json',inputs)
 records=[];nonzero=0;maximum=Q(0)
 for f in range(151):
  emit(dict(kind='BEFORE_NEW_cache_alias',frame=f,source_data_sha256=inputs['saved_source_data']['sha256'],delta_UNASSIGNED=True))
  ret=cached_camera_alias(Vint,EV,poses[f,:3,:3],poses[f,:3,3],C0[f])
  # Exact signed integers are stored losslessly as text, never rounded to floats.
  strings=[[str(int(n))for n in row]for row in ret['defect_numerators']]
  width=max(len(n)for row in strings for n in row)
  path=O/f'frame-{f:03d}-NEW-exact-cache-alias-numerators-RETURN.npz'
  np.savez_compressed(path,signed_integer_numerators=np.array(strings,dtype=f'U{width}'),
   denominator_exponent=np.array(ret['denominator_exponent'],np.int64))
  path.chmod(0o600)
  with path.open('rb')as fp:os.fsync(fp.fileno())
  array_return=pointer(path);emit(dict(kind='AFTER_NEW_exact_alias_constructor_RETURN',frame=f,actual_RETURN=array_return))
  counts=int(sum(int(n)!=0 for row in ret['defect_numerators']for n in row))
  Cabs=[Q(float(x))for x in np.max(np.abs(C0[f]),axis=0)]
  coefficients=cache_error_affine_coefficients(Cabs,ret['maximum_absolute_defects'],RC[f])
  r=dict(frame=f,source_frame_index=int(frames[f]),timestamp_s=float(times[f]),vertices=len(V),
   exact_signed_defects=array_return,nonzero_exact_alias_components=counts,
   max_absolute_alias_by_component=ret['maximum_absolute_defects'],max_witness_vertex_indices=ret['witness_indices'],
   cached_C0_component_absolute=Cabs,
   NEW_conditional_camera_error_coefficients=coefficients,
   coefficient_order=['constant','H_world_x','H_world_y','H_world_z'],
   world_component_halfwidth_symbols_UNASSIGNED=[f'H_{f}_{j}'for j in range(3)],
   arithmetic_scope='finite RN cached operator C0 + einsum(Rcw,delta); no new box or gate',
   uniform_over_all_original_source_visual_vertices=True,
   cached_C0_and_exact_affine_formula_remain_distinct_UNALTERED=True,
   no_NEW_camera_projection_NN_raster_source_pose_or_solver_evaluation=True)
  ref=save(f'frame-{f:03d}-NEW-cache-numerical-map-RETURN.json',r);emit(dict(kind='AFTER_NEW_conditional_error_map_RETURN',frame=f,actual_RETURN=ref))
  assert len(V)==6022 and all(x>=0 for row in coefficients for x in row)
  nonzero+=counts;maximum=max(maximum,*ret['maximum_absolute_defects']);records.append(r)
 # Complete actual acceptance protocol: these are staging obligations, not a
 # new numerical score or acceptance tolerance. There is no candidate here.
 protocol=dict(schema='v2d.NEW-source-numerical-acceptance-protocol.v1',
  proposal_before_operator='Durably seal NEW derived object/robot arrays, full151clocks, ORIGINAL-data/code/runtime hashes and declared construction; retain raw constructor and solver/intermediate returns',
  zero_F='Reuse SAVED training cache BY HASH, no baseline source/raster/NN replay',
  training_nonzero='Execute unchanged training k2/einsum/cache/fulltriangle-raster operator on a separately declared NEW proposal only; retain ALL raw outputs, infinities, rank/NN changes, missing/unknown cases',
  pruning='Conditional projection/leaf-norm intervals can retain unresolved branches; they cannot establish original gate passes or global KD-search equivalence',
  hard_source_training='All151 actual training IoU/median/quality restrictions must pass; no tolerance/threshold repair or heldout/fullpixel feedback',
  object_motion='ALL6ORIGINAL151clock object gates, actual timestamps/p95 and original1e-9 comparisons',
  original_fullpixel='Only after construction, ORIGINAL fullpixel k1/matmul scorer on the NEW candidate with original1e-10 comparisons and original authority; never substitute a training interval, cached result or k2 identity',
  physical='Changed reference requires independent whole151 source/URDF/all root+finger+object motion, fresh FULL1mm native hand/object/support AND self, full swept, actual motor/grasp and whole source before GPU/PPO',
  unknown_or_failure='Retain and REJECT as a reference; do not refit heldout/fullpixel failures or promote intermediates',
  remaining_unknowns=['global KD-search/compiler/sqrt equivalence outside tested synthetic scopes',
   'full installed raster-binary equivalence outside tested synthetic scopes',
   'candidate-specific complete pose construction and acceptance binding not selected',
   'source accuracy and physical trajectory qualification remain unproved'],
  candidate_tolerances_source_parameters_and_F_UNCHANGED=True,no_solver_or_pose_queued=True)
 save('NEW-complete-source-acceptance-protocol-RETURN.json',protocol)
 save('all151-NEW-source-numerical-bindings-RETURN.json',records)
 report=dict(schema='v2d.NEW-source-numerical-bindings.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  frames=151,source_visual_vertices=6022,exact_cache_alias_components=151*6022*3,
  nonzero_exact_cache_alias_components=nonzero,maximum_exact_absolute_cache_alias=maximum,
  maximum_exact_absolute_cache_alias_m_float=float(maximum),NEW_conditional_camera_error_coefficients=151*12,
  all_H_world_halfwidths_UNASSIGNED_no_new_box=True,all_defects_raw_arrays_and_old_records_UNALTERED=True,
  numerical_operator_k1_k2_and_cache_matmul_scopes_explicit=True,
  original_numeric_search_and_raster_build_uncertainties_remain=True,
  no_old_source_NN_raster_or_audit_replay=True,no_episode_pose_solver_native_physics_GPU_or_qualification=True)
 save('report.json',report);emit(dict(kind='AFTER_NEW_numerical_binding_report_RETURN',actual_RETURN=report));print(json.dumps(pack(report)))
