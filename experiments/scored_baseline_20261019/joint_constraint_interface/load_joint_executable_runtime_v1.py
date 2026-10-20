"""ONE NEW loader-constructor receipt; no coordinates/directions are supplied."""
from joint_constraint_derivatives_v1 import *
import time
D=S/'e037-joint-runtime-loader-v1'
binding=S/'e037-joint-executable-binding-v1/report.json'
assert json.loads(binding.read_text())['passed']
start=time.monotonic()
api=JointExecutableInterface(S/'e037-joint-executable-interface-v1')
# The object's full immutable payload is represented by lossless byte hashes and
# input artifact references; no new candidate, source, geometry or matrix-vector
# return is calculated. Source/geometry arrays themselves remain private/pinned.
array_record=lambda a:dict(shape=list(np.shape(a)),dtype=str(np.asarray(a).dtype),
                         bytes_sha256=hashlib.sha256(np.asarray(a).tobytes()).hexdigest())
state={}
for name,value in api.__dict__.items():
 if sparse.issparse(value):
  state[name]=dict(kind='sparse_CSC',shape=value.shape,data=array_record(value.data),
                   indices=array_record(value.indices),indptr=array_record(value.indptr))
 elif isinstance(value,np.ndarray):state[name]=dict(kind='array',**array_record(value))
 elif isinstance(value,dict) and value and all(isinstance(v,np.ndarray) for v in value.values()):
  state[name]=dict(kind='array_mapping',fields={str(k):array_record(v) for k,v in value.items()})
 elif isinstance(value,Path):state[name]=dict(kind='path',value=str(value))
 elif isinstance(value,(dict,list)):
  state[name]=dict(kind='immutable_json',canonical_sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),length=len(value))
 else:state[name]=dict(kind='scalar',value=value)
save(D/'ONE-NEW-runtime-loader-constructor-RETURN.json',dict(
 class_name=type(api).__name__,attributes=state,constructor_wall_seconds=time.monotonic()-start,
 immutable_input_manifest_path=str(api.root/'NEW-interface-input-bindings-RETURN.json'),
 immutable_input_manifest_sha256=sha(api.root/'NEW-interface-input-bindings-RETURN.json'),
 no_x_or_direction_OR_episode_function_supplied=True,
 no_matrix_vector_source_FK_native_geometry_solver_queries=True))
assert api.A.shape==(4551571,1058) and api.robot.shape==(8114,465)
assert len(api.source)==194 and len(api.domains['retained_domain_factors'])==38
assert len(api.flags)==1653 and api.incidence['all_frames'].shape==(151,)
save(D/'report.json',dict(schema='v2d.NEW-joint-runtime-loader-constructor.v1',
 loader_return_durable=True,attributes=len(state),
 immutable_arrays_loaded_ONCE=True,all_episode_coordinates_directions_UNASSIGNED=True,
 actual_object_root_hand_pose_source_geometry_native_solver_calls=0,
 passed=True,wall_seconds=time.monotonic()-start))
