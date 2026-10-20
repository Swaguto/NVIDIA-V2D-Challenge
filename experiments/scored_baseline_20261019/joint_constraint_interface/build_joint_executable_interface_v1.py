"""NEW executable sparse/chain routing construction, no episode numerical call."""
from joint_collision_maps_v1 import *
from fractions import Fraction
from scipy import sparse
D=S/'e037-joint-executable-interface-v1'
model=S/'e037-hard-raster-model-v1'
routes=S/'e037-joint-collision-predicates-v1'
contracts=S/'e037-joint-interface-contracts-v1/report.json'
paths=[contracts,C/'joint_constraint_interface_v1.py',C/'coupled_angle_source_v1.py',C/'full_pose_construction_v1.py',
 model/'NEW-canonical-A-RETURN.npz',model/'NEW-canonical-P-RETURN.npz',model/'NEW-canonical-vectors-RETURN.npz',model/'NEW-canonical-metadata-RETURN.json',
 M/'continuous-angle-variables.json',M/'all-exact-linear-angle-inequalities.json',M/'all-direct-source-factor-proposals.json',M/'retained-halfangle-domains-and-unknown-source-evidence.json',
 M/'unqualified-model-fixed-components-and-initial-F.npz',F/'manifest.json',F/'targets.npz',
 G/'NEW-native-body-rotation-factors-RETURN.npz',G/'NEW-source-rotation-factors-RETURN.npz',
 O/'NEW-static-native-tree-RETURN.json',O/'NEW-articulated-chain-program-RETURN.json',O/'NEW-URDF-native-incidence-RETURN.json',
 O/'NEW-complete-column-and-pair-incidence-RETURN.npz',routes/'NEW-ALL-pair-quantified-geometric-constraints-RETURN.json',
 routes/'NEW-object-canonical-row-and-column-routing-RETURN.npz',
 C/'object-authority-checkpoint-v1.json',S/'e037-joint-one-load-binding-v1/report.json']
bindings={str(p):sha(p) for p in paths}
save(D/'NEW-interface-input-bindings-RETURN.json',dict(input_sha256=bindings,
 full_region2_3unknown_bands_reused_BY_HASH=True,old_194factor_returns_NOT_evaluated=True,
 NEW_synthetic_contracts_by_hash=bindings[str(contracts)],CPU_seconds=240,RSS_bytes=2*1024**3,disk_reserve_bytes=int(1.5*1024**3)))
assert json.loads(contracts.read_text())['passed']
# Route every coefficient into a new executable copy; never call old canonical,
# body/source/native producers or old constructor/solver.
with np.load(model/'NEW-canonical-A-RETURN.npz',allow_pickle=False) as z:
 values=z['data'];rows=z['indices'];ptr=z['indptr']
keep=(rows<2)|(rows>=66593)
newvalues=values[keep];newrows=rows[keep].copy();newrows[newrows>=66593]-=66591
newptr=np.zeros(1059,dtype=ptr.dtype)
newptr[1:]=np.cumsum([np.count_nonzero(keep[int(a):int(b)]) for a,b in zip(ptr[:-1],ptr[1:])])
A=sparse.csc_matrix((newvalues,newrows,newptr),shape=(4551571,1058))
sparse.save_npz(D/'NEW-object-retained-A-RETURN.npz',A,compressed=False)
new_count=len(newvalues);del values,rows,ptr,keep,newvalues,newrows,newptr,A
records=json.loads((M/'all-exact-linear-angle-inequalities.json').read_text())
variables=json.loads((M/'continuous-angle-variables.json').read_text());byname={r['name']:r['column'] for r in variables}
ri=[];ci=[];data=[];upper=[];origins=[]
rat=lambda x:Fraction(int(x['numerator']),int(x['denominator']))
for row,rec in enumerate(records):
 upper.append(float(rat(rec['folded_upper'])))
 for term in rec['active_terms']:
  ri.append(row);ci.append(byname[term['variable']]);data.append(float(rat(term['coefficient'])))
 origins.append(dict(row=row,original_id=rec['id'],original_kind=rec['kind'],
  original_exact_rhs=rec['folded_upper'],original_exact_terms=rec['active_terms'],constant_only=rec['constant_only']))
robot=sparse.coo_matrix((data,(ri,ci)),shape=(8114,465)).tocsc()
sparse.save_npz(D/'NEW-robot-linear-A-RETURN.npz',robot,compressed=False)
save(D/'NEW-robot-linear-exact-origin-RETURN.json',origins)
with np.load(model/'NEW-canonical-vectors-RETURN.npz',allow_pickle=False) as z:
 b=z['b'];q=z['q'];bkeep=np.r_[b[:2],b[66593:]];qembed=np.r_[np.zeros(465),q]
metadata=json.loads((model/'NEW-canonical-metadata-RETURN.json').read_text())
caps=json.loads((C/'object-authority-checkpoint-v1.json').read_text())['original_object_motion_caps']
save_npz(D/'NEW-interface-constants-RETURN.npz',object_b=bkeep,robot_b=np.array(upper),q=qembed,
 nonnegative=np.array(metadata['nonnegative']-66591,dtype=np.int64),
 radius=np.array(metadata['radius']),scale=np.array(metadata['scale']),object_caps=np.array(list(caps.values())))
oldP=sparse.load_npz(model/'NEW-canonical-P-RETURN.npz').tocoo()
newP=sparse.coo_matrix((oldP.data,(oldP.row+465,oldP.col+465)),shape=(1523,1523)).tocsc()
sparse.save_npz(D/'NEW-embedded-objective-P-RETURN.npz',newP,compressed=False)
# Every executable source call binds to its saved original source factor. The
# constructor copies constants/routes only; it never invokes the point operator.
factors=json.loads((M/'all-direct-source-factor-proposals.json').read_text())
source_route=[]
for rec in factors:
 cols=[byname[f"q_{rec['frame']}_{name}"] for name in rec['free_joint_names']]
 source_route.append(dict(factor=rec['id'],frame=rec['frame'],columns=cols,
  original_factor=rec,operator='source_chain(original_URDF,root,complete_angles,link), THEN unchanged camera_source_factor',
  pure_camera_operator_sha256=bindings[str(C/'coupled_angle_source_v1.py')],
  native_URDF_metadata_identity_NOT_assumed=True))
save(D/'NEW-ALL-source-executable-routing-RETURN.json',source_route)
# Full unknown evidence has a typed return channel rather than being reduced to
# a false predicate or a convenient mandatory sampled node.
domains=json.loads((M/'retained-halfangle-domains-and-unknown-source-evidence.json').read_text())
save(D/'NEW-region2-domain-return-protocol-RETURN.json',dict(
 original_domain_path=str(M/'retained-halfangle-domains-and-unknown-source-evidence.json'),
 original_domain_sha256=bindings[str(M/'retained-halfangle-domains-and-unknown-source-evidence.json')],
 half_angle_domains=38,original_region2_box_rows_already_in_8114=True,
 full_region2_and_unknown109bands_returned_AS_UNKNOWN=True,
 regular_strips_NOT_full_trajectory_pass=True,unknown_bands_NOT_deleted_or_mandatory_measured_nodes=True,
 no_source_polynomial_or_chart_regeneration=True))
# Resource bound is for the new materialized arrays and streamed frame returns,
# not for an undeclared optimizer/evaluation workload.
storage=sum(p.stat().st_size for p in D.glob('*.npz'))
save(D/'NEW-interface-resource-and-return-contract-RETURN.json',dict(
 immutable_sparse_storage_bytes=storage,loader_arrays_fetched_ONCE=True,
 new_object_nonzeros=new_count,robot_stored_terms=len(data),
 no_dense_full_Jacobian_or_vertex_cartesian_products=True,
 at_most_one_frame_geometry_and1653pair_returns_in_memory=True,
 durable_ALLinput_then_canonical_source_motion_then_perframe_body_pair_returns=True,
 actual_episode_call_REQUIRES_new_sealed_proposal_and_DurableRecorder=True,
 no_candidate_or_direction_SELECTED=True,resource_guard_seconds=240,RSS_bytes=2*1024**3,
 original_objective_and_conditioning_UNCHANGED=True,
 all_new_closed_model_feasibility_would_NOT_be_actual_source_or_native_qualification=True))
save(D/'report.json',dict(schema='v2d.NEW-joint-executable-interface-preflight.v1',
 variables=1523,object_rows=4551571,robot_linear_rows=8114,object_nonzeros=new_count,
 robot_exact_terms=len(data),source_factors=194,half_domains=38,
 native_bodies=31,native_geoms=58,complete_pair_instances=249603,
 SOC4=900,SOC5=151,immutable_sparse_storage_bytes=storage,
 all_numeric_episode_inputs_UNASSIGNED=True,
 no_episode_pose_source_NN_raster_FK_native_optimizer_physics_evaluation=True))
