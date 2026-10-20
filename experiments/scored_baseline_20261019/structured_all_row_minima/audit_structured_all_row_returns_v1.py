"""Independent SAVED NEW returns only; no optimizer/matvec/source replay."""
from structured_all_rows_v1 import *
E=S/'e037-structured-all-row-evaluation-v1';O=S/'e037-structured-all-row-return-audit-v1'
meta=json.loads((PREFLIGHT/'NEW-structured-layout-RETURN.json').read_text());groups=meta['groups'];cones=meta['cones']
# Read only the small subset of rows needed for auditing the returned SEARCH J.
# This is a NEW-return coefficient identity, never evaluating A@x again.
D=np.load(PREFLIGHT/'NEW-CSR-data-RETURN.npy',mmap_mode='r');I=np.load(PREFLIGHT/'NEW-CSR-indices-RETURN.npy',mmap_mode='r');P=np.load(PREFLIGHT/'NEW-CSR-indptr-RETURN.npy',mmap_mode='r')
A=sparse.csr_matrix((D,I,P),shape=(4618162,1058),copy=False)
records=[];maxj=0.;allrows=0
paths=sorted(E.glob('query-*-NEW-ALL-row-RETURN.npz'))
for k,p in enumerate(paths):
 with np.load(p) as z:r=z['raw_rows'];v=z['values'];J=z['search_J'];ties=z['tie_rows'];offset=z['tie_offsets']
 with np.load(E/f'query-{k:03d}-NEW-input-BEFORE.npz') as z:x=z['x']
 proof=[];je=0.;ce=0.
 for i,(a,b,kind,f) in enumerate(groups):
  mn=min(r[a:b]);ix=np.flatnonzero(r[a:b]==mn)+a
  exact_ties=np.array_equal(ties[offset[i]:offset[i+1]],ix)
  expected=np.asarray(A[ix].sum(axis=0)).ravel()/(-len(ix));err=float(np.max(abs(expected-J[i])))
  proof.append(dict(group=i,rows=b-a,minimum=float(mn),returned_value=float(v[i]),tie_rows=int(len(ix)),all_ties_exact=exact_ties,J_error=err));je=max(je,err)
 for i,(a,size,kind) in enumerate(cones):
  row=len(groups)+i;length=float(np.sqrt(sum(float(z)*float(z) for z in r[a+1:a+size])))
  val=float(r[a]-length);err=abs(val-float(v[row]));ce=max(ce,err)
  # Independent component accumulation of saved cone derivative identity.
  expected=-A[a].toarray().ravel()
  if length:
   for j in range(1,size):expected+=A[a+j].toarray().ravel()*(r[a+j]/length)
  je=max(je,float(np.max(abs(expected-J[row]))))
 record=dict(query=k,ALL_saved_rows=len(r),finiteness=bool(np.isfinite(r).all()),affine_value_identities=all(t['minimum']==t['returned_value'] for t in proof),ALL_tie_identities=all(t['all_ties_exact'] for t in proof),max_cone_value_error=ce,max_SEARCH_J_error=je,unknowns=json.loads((E/f'query-{k:03d}-NEW-unknown-RETURN.json').read_text()),input_sha256=hashlib.sha256(x.tobytes()).hexdigest())
 dump(O/f'query-{k:03d}-NEW-saved-group-identities-RETURN.json',proof);dump(O/f'query-{k:03d}-NEW-saved-binding-RETURN.json',record)
 assert len(r)==4618162 and record['finiteness'] and record['affine_value_identities'] and record['ALL_tie_identities'] and not record['unknowns'] and ce<1e-12 and je<1e-10
 records.append(record);maxj=max(maxj,je);allrows+=len(r)
report=json.loads((E/'report.json').read_text());events=[json.loads(x) for x in (E/'NEW-method-journal.jsonl').read_text().splitlines()]
new_events=[z for z in events if z['kind']=='AFTER_ALL_rows_NEW_RETURN']
assert len(new_events)==len(paths)==report['unique_queries']
callbacks=list(E.glob('callback-*-NEW-actual-RETURN.npz'));assert len(callbacks)==report['callbacks']
terminal=E/'ONE-actual-OptimizeResult-RETURN.npz'
terminal_check=None
if terminal.exists():
 with np.load(terminal) as z:x=z['x']
 matching=[]
 for k in range(len(paths)):
  with np.load(E/f'query-{k:03d}-NEW-input-BEFORE.npz') as z:
   if np.array_equal(x,z['x']):matching.append(k)
 terminal_check=dict(raw_return_matches_queries=matching)
 dump(O/'NEW-raw-terminal-binding-RETURN.json',terminal_check);assert matching
else:assert report.get('no_candidate_selected')
result=dict(schema='v2d.NEW-structured-all-row-saved-return-audit.v1',unique_queries=len(paths),callbacks=len(callbacks),ALL_saved_row_values_inspected=allrows,ALL_affine_minimum_tie_incidence_verified=True,ALL_1051_cones_each_query_verified=True,max_SEARCH_J_identity_error=maxj,diagnostic_bounds_NOT_acceptance_tolerances=True,terminal=terminal_check,actual_source_queries=report['actual_source_queries'],optimizer_matvec_model_source_NN_raster_native_NOT_replayed=True,no_physical_qualification=True)
dump(O/'report.json',result)
