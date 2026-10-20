"""NEW concrete serializer controls; no unavailable loader/state rerun."""
from joint_lossless_returns_v1 import *
from joint_collision_maps_v1 import S,save,sha
import struct
D=S/'e037-joint-lossless-return-contracts-v1'
records=[]
def report(name,raw,passed):
 save(D/(name+'-RETURN.json'),raw);records.append(dict(name=name,passed=bool(passed)));assert passed,name
value={(3,'finger'):np.array([.1,-0.,.3]),'3':{(0,1):[None,True,17]},3:'integer key',Path('/synthetic'):'path key'}
p=seal_return(D,'NEW-tuple-key-raw',value);roundtrip=restore_NEW_synthetic(p)
report('NEW-tuple-key-exact-types',dict(path=str(p),tuple_key_present=(3,'finger') in roundtrip,
 integer_distinct_from_string=roundtrip[3]=='integer key' and isinstance(roundtrip['3'],dict),
 synthetic_array_bytes_exact=roundtrip[(3,'finger')].tobytes()==value[(3,'finger')].tobytes()),
 (3,'finger') in roundtrip and roundtrip[3]=='integer key' and roundtrip[(3,'finger')].tobytes()==value[(3,'finger')].tobytes())
# IEEE bit patterns are inspected independently of hex scalar formatting.
bits=np.array([0,1<<63,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000042],dtype=np.uint64)
floats=bits.view(np.float64);payload=dict(a=floats,scalar=np.float64(-0.),python_scalar=-0.)
p=seal_return(D,'NEW-IEEE-raw',payload);decoded=restore_NEW_synthetic(p)
actual_bits=decoded['a'].view(np.uint64)
report('NEW-IEEE-signedzero-nonfinite-payload',dict(expected_hex=[hex(int(b)) for b in bits],actual_hex=[hex(int(b)) for b in actual_bits],
 numpy_scalar_hex=decoded['scalar'].tobytes().hex(),python_scalar_bits=struct.pack('>d',decoded['python_scalar']).hex()),
 np.array_equal(actual_bits,bits) and decoded['scalar'].tobytes()==np.float64(-0.).tobytes() and struct.pack('>d',decoded['python_scalar'])==struct.pack('>Q',1<<63))
# Keep explicit zeros, sparse index order and pointers, without coalescing.
a=sparse.csc_matrix((np.array([0.,.4,-.2]),np.array([2,0,1],dtype=np.int32),np.array([0,2,3],dtype=np.int32)),shape=(3,2))
p=seal_return(D,'NEW-sparse-raw',dict(A=a));b=restore_NEW_synthetic(p)['A']
report('NEW-sparse-storage-not-repaired',dict(data_exact=a.data.tobytes()==b.data.tobytes(),
 indices_exact=a.indices.tobytes()==b.indices.tobytes(),indptr_exact=a.indptr.tobytes()==b.indptr.tobytes(),stored_entries=b.nnz),
 a.data.tobytes()==b.data.tobytes() and a.indices.tobytes()==b.indices.tobytes() and a.indptr.tobytes()==b.indptr.tobytes() and b.nnz==3)
save(D/'NEW-failed-loader-boundary-scope-RETURN.json',dict(
 failed_guard_path=str(S/'e037-joint-runtime-loader-v1/guard.json'),
 failed_guard_sha256=sha(S/'e037-joint-runtime-loader-v1/guard.json'),
 failed_writer_source_sha256=sha(S/'compute/load_joint_executable_runtime_v1.py'),
 exception='TypeError tuple mapping key during metadata hashing',
 old_runtime_object_state_and_constructor_receipt_UNAVAILABLE=True,
 old_loader_constructor_NOT_repeated=True,old_state_NOT_reconstructed=True,
 new_serializer_controls_ONLY_NEW_synthetic_inputs=True,
 future_NEW_method_constructor_inputs_and_ALL_raw_returns_must_be_sealed_with_typed_serializer=True))
save(D/'report.json',dict(schema='v2d.NEW-joint-lossless-return-contracts.v1',contracts=records,passed=True,
 old_loader_and_unavailable_state_NOT_replayed_or_reconstructed=True))
