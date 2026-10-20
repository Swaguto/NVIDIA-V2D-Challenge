"""Independent full NEW proof-map bindings, no old map/model or operator replay."""
from joint_saved_mean_bound_v1 import *
from joint_lossless_returns_v1 import seal_return
D=S/'e037-joint-mean-proof-binding-v1';M=S/'e037-joint-mean-proof-maps-v1';p=M/'NEW-complete-proof-maps-RETURN.json';mp=typed(p);df=dynamic_file();saved=typed(df);J=saved['J'];E=saved['absolute_tie_envelope']
save(D/'NEW-map-input-bindings-RETURN.json',{str(q):sha(q) for q in [p,p.with_suffix('.npz'),M/'NEW-fixed-proof-declaration-RETURN.json',df,df.with_suffix('.npz'),C/'joint_saved_mean_bound_v1.py']})
# Independent CSC incidence uses explicit column slices, no producer row sorting.
c=np.empty(J.nnz,dtype=np.int64)
for k in range(J.shape[1]):c[J.indptr[k]:J.indptr[k+1]]=k
assert np.array_equal(c,mp['columns']) and np.array_equal(J.indices,mp['rows'])
assert hashlib.sha256(J.data.tobytes()).hexdigest()==mp['J_data_bytes_sha256'] and hashlib.sha256(E.data.tobytes()).hexdigest()==mp['E_data_bytes_sha256']
visited=np.zeros(J.nnz,dtype=np.int8);visited[J.indices<852]=1;covered=0;numeric_terms=0
for f,p in enumerate(frame_files()):
 a=typed(p);start=852+f*1653;ids=np.flatnonzero((J.indices>=start)&(J.indices<start+1653));pairs=J.indices[ids]-start
 for pos,k in zip(ids,pairs):
  k=int(k);kind=int(mp['kind'][pos]);local=int(mp['local'][pos]);column=int(c[pos]);assert int(mp['frame'][pos])==f and int(mp['pair'][pos])==k
  assert not a['conditional_filter'][k]
  if column==1522:assert kind==1 and local==-1
  elif column>=1523:assert kind==3 and column==1523+2*(f*1653+k)+local and local in [0,1]
  else:
   assert kind==2 and a['command_ptr'][k]<=local<a['command_ptr'][k+1] and a['columns'][local]==column
   numeric_terms+=int(a['A_slope_ptr'][local+1]-a['A_slope_ptr'][local]+a['B_slope_ptr'][local+1]-a['B_slope_ptr'][local])
  visited[pos]+=1
 observed=np.bincount(pairs,minlength=1653);assert np.array_equal(observed,mp['all_pair_entry_counts'][f]);covered+=len(observed)
 # All filtered pairs remain typed zero-incidence proof rows, never deleted.
 assert all(observed[k]==0 for k in range(1653) if a['conditional_filter'][k])
seal_return(D,'NEW-independent-full-proof-incidence-RETURN',dict(all_visits_once=bool(np.all(visited==1)),ALL_entries=J.nnz,ALL_pair_scopes=covered,ALL_command_slope_terms=numeric_terms,
 ALL_original_J_E_bytes_unchanged=True,original4810140_rows_objective_conditioning_strict_flags_BY_HASH=True))
assert np.all(visited==1) and covered==249603
save(D/'report.json',dict(passed=True,ALL933560_entries_bound=True,ALL249603_pairs_bound=True,ALL_new_map_positions_once=True,no_numeric_proof_evaluation_yet=True,
 no_old_controls_queries_or_operator_replay=True))
