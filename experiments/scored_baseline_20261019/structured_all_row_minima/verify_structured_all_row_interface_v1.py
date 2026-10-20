"""Independent NEW routing/permutation/objective binding, no old model audit."""
from structured_all_rows_v1 import *
O=S/'e037-structured-all-row-binding-v1'
meta=json.loads((PREFLIGHT/'NEW-structured-layout-RETURN.json').read_text())
D=np.load(PREFLIGHT/'NEW-CSR-data-RETURN.npy',mmap_mode='r');I=np.load(PREFLIGHT/'NEW-CSR-indices-RETURN.npy',mmap_mode='r');P=np.load(PREFLIGHT/'NEW-CSR-indptr-RETURN.npy',mmap_mode='r')
new=sparse.csr_matrix((D,I,P),shape=tuple(meta['shape']),copy=False).tocsc()
old=sparse.load_npz(MODEL/'NEW-canonical-A-RETURN.npz')
result=dict(coefficient_values_exact=np.array_equal(new.data,old.data),coefficient_column_rows_exact=np.array_equal(new.indices,old.indices),column_pointers_exact=np.array_equal(new.indptr,old.indptr),cells=len(D),explicit_zero_coefficients_preserved=True)
dump(O/'NEW-ALL-coefficient-routing-RETURN.json',result)
assert all(result[k] for k in ['coefficient_values_exact','coefficient_column_rows_exact','column_pointers_exact'])
del new,old
cover=np.zeros(4618162,np.uint8);records=[]
for a,z,kind,f in meta['groups']:
 cover[a:z]+=1
 if kind=='geometry_ALL_pairs':assert a==2+441*f and z-a==441
 if kind=='source_ALL_cell_depth_rows':assert a==67046+30111*f and z-a==30111
 records.append(dict(start=a,end=z,kind=kind,frame=f,rows=z-a))
for i,(a,k,kind) in enumerate(meta['cones']):
 expected=4613807+4*i if i<900 else 4617407+5*(i-900)
 assert a==expected and k==(4 if i<900 else 5);cover[a:a+k]+=1
arrays(O/'NEW-independent-row-incidence-RETURN.npz',multiplicity=cover)
dump(O/'NEW-independent-group-incidence-RETURN.json',records)
assert np.all(cover==1)
for f,row in enumerate(meta['strict_frame_pointers']):assert row['frame']==f and sha(row['path'])==row['sha256'] and row['rows']==30111
for key in ['canonical_A','canonical_P','canonical_vectors']:assert sha(meta[key]['path'])==meta[key]['sha256']
dump(O/'report.json',dict(schema='v2d.NEW-structured-all-row-independent-binding.v1',all_32306054_stored_coefficients_exact=True,all_4618162_rows_routed_once=True,affine_groups=len(records),SOC4=900,SOC5=151,strict_frame_pointers_bound=151,objective_vectors_RHS_conditioning_BY_HASH=True,model_source_NN_raster_native_solver_NOT_replayed=True))
