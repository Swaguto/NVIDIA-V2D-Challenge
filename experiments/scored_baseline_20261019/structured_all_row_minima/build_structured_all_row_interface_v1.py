"""Lossless NEW all-row routing/CSR interface from the frozen canonical only."""
from structured_all_rows_v1 import *
O=PREFLIGHT
m=json.loads((MODEL/'NEW-canonical-metadata-RETURN.json').read_text());groups,cones=layout(m)
dump(O/'NEW-routing-declaration-BEFORE.json',dict(groups=groups,cones=cones,all_original_rows=True,no_source_or_model_regeneration=True))
A=sparse.load_npz(MODEL/'NEW-canonical-A-RETURN.npz').tocsr()
for name,v in [('data',A.data),('indices',A.indices),('indptr',A.indptr)]:
 with (O/f'NEW-CSR-{name}-RETURN.npy').open('wb') as f:np.save(f,v);f.flush();os.fsync(f.fileno())
flags=[]
for f in range(151):
 p=MODEL/f'frame-{f:03d}-NEW-hard-cell-coefficients-RETURN.npz'
 with np.load(p) as z:flags.append(dict(frame=f,path=str(p),sha256=sha(p),rows=len(z['rhs']),strict_open=int(z['strict_open'].sum()),unknown_semantics='ALL saved active/strict/row_kind/target arrays retained by immutable pointer'))
out=dict(schema='v2d.NEW-structured-all-row-layout.v1',shape=list(A.shape),stored_entries=A.nnz,groups=groups,cones=cones,strict_frame_pointers=flags,canonical_A=dict(path=str(MODEL/'NEW-canonical-A-RETURN.npz'),sha256=sha(MODEL/'NEW-canonical-A-RETURN.npz')),canonical_vectors=dict(path=str(MODEL/'NEW-canonical-vectors-RETURN.npz'),sha256=sha(MODEL/'NEW-canonical-vectors-RETURN.npz')),canonical_P=dict(path=str(MODEL/'NEW-canonical-P-RETURN.npz'),sha256=sha(MODEL/'NEW-canonical-P-RETURN.npz')),all_rows_every_unique_query=True,geometry_rows_per_frame=441,source_rows_per_frame=30111,no_constraint_drop_or_new_box=True,objective_and_conditioning_UNCHANGED=True)
dump(O/'NEW-structured-layout-RETURN.json',out)
cover=np.zeros(A.shape[0],np.uint8)
for a,z,_,_ in groups:cover[a:z]+=1
for a,k,_ in cones:cover[a:a+k]+=1
arrays(O/'NEW-routing-coverage-RETURN.npz',row_multiplicity=cover)
assert np.all(cover==1)
dump(O/'report.json',dict(schema='v2d.NEW-structured-all-row-preflight.v1',rows=A.shape[0],cols=A.shape[1],stored_entries=A.nnz,affine_ALL_groups=len(groups),SOC_cones=len(cones),unique_rows_routed_once=int((cover==1).sum()),unknowns_strict_flags_preserved=True,no_episode_point_or_optimizer=True))
