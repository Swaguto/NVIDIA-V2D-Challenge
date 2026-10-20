"""Independent NEW exact bound/origin/functionals before certificate evaluation."""
from exact_terminal_dual_bound_v1 import *
O=S/'e037-exact-terminal-dual-binding-v1';A=load_csr()
with np.load(M/'NEW-canonical-vectors-RETURN.npz') as z:b=z['b']
mp=json.loads((P/'NEW-complete-certificate-map-RETURN.json').read_text());bs=json.loads((P/'NEW-exact-rational-constructor-bounds-RETURN.json').read_text());pr=json.loads((P/'NEW-bound-origin-necessity-proofs-RETURN.json').read_text());records=[]
# Independent exact coordinate proofs from actual sparse cone components.
for item in bs:
 col=item['column'];lo=F(item['lower']);hi=F(item['upper']);mag=F(item['magnitude'])
 if col==1057:
  aa,bb=row(A,b,0);cc,dd=row(A,b,1);expected=(F(0),F(1));assert aa==[(1057,F(-1))] and bb==0 and cc==[(1057,F(1))] and dd==1
 elif col%7<6:
  f,j=divmod(col,7);top=NN+4*598+8*f+(4 if j>=3 else 0);r=top+1+j%3;topterms,T=row(A,b,top);terms,center=row(A,b,r);assert not topterms and T>0 and len(terms)==1 and terms[0][0]==col
  a=terms[0][1];expected=tuple(sorted([(center+T)/a,(center-T)/a]))
 else:
  f=col//7;terms,U=row(A,b,66895+f);assert terms==[(col,F(1))] and U>=0
  a0,b0=row(A,b,4617407+5*f);a4,b4=row(A,b,4617407+5*f+4);assert a0==[(col,F(-1))] and a4==[(col,F(-1))]
  c0=a0[0][1];c4=a4[0][1];poly=[b0*b0-b4*b4,-2*b0*c0+2*b4*c4,c0*c0-c4*c4]
  assert poly==[0,4,0]
  for j in range(3):assert row(A,b,4617407+5*f+1+j)==([(7*f+3+j,F(-2))],F(0))
  expected=(F(0),U)
 record=dict(column=col,lower=str(lo),upper=str(hi),expected_lower=str(expected[0]),expected_upper=str(expected[1]),magnitude=str(mag),exact=expected==(lo,hi) and mag==max(abs(lo),abs(hi)))
 records.append(record)
dump(O/'NEW-independent-1058-bound-proofs-RETURN.json',records);assert all(r['exact'] for r in records)
with np.load(M/'NEW-canonical-A-RETURN.npz') as z:ptr=z['indptr'];indices=z['indices'];entries=len(z['data'])
cursor=0
for j,rec in enumerate(mp['ALL_column_functionals']):
 assert rec['column']==j and rec['CSC_start']==int(ptr[j])==cursor and rec['CSC_end']==int(ptr[j+1]);cursor=rec['CSC_end']
assert cursor==32306054 and len(mp['ALL_column_functionals'])==1058 and np.all((indices>=0)&(indices<ROWS))
assert mp['by_functional']['row_start']==0 and mp['by_functional']['row_end']==ROWS and mp['orthant_defect']['row_start']==0 and mp['orthant_defect']['row_end']==NN
layout=json.loads((I/'NEW-structured-layout-RETURN.json').read_text());cursor=NN
for rec,(a,k,kind) in zip(mp['SOC_defect'],layout['cones']):assert rec['start']==a==cursor and rec['size']==k;cursor+=k
assert cursor==ROWS and len(mp['SOC_defect'])==1051 and mp['strict_pointers']==layout['strict_frame_pointers']
for p,h in mp['inputs'].items():assert sha(p)==h
assert mp['bounds_file_sha256']==sha(P/'NEW-exact-rational-constructor-bounds-RETURN.json') and mp['proof_file_sha256']==sha(P/'NEW-bound-origin-necessity-proofs-RETURN.json')
# Scope dimensions bind the selected terminal file and the normalized phase.
with np.load(E/'ONE-actual-raw-terminal-RETURN.npz') as z:y=z['u']
assert len(y)==ROWS and np.isfinite(y).all() and mp['phase_column']==1057 and mp['root_bits']==96 and mp['exact_arithmetic_bit_cap']==8192
result=dict(schema='v2d.NEW-exact-terminal-dual-independent-map-binding.v1',ALL1058rational_bounds_exact=True,ALL32306054coefficient_functional_entries_bound=True,ALL4618162by_rows_bound=True,ALL4613807orthant_rows_bound=True,ALL900SOC4_151SOC5_bound=True,ALL151strict_pointers_bound_BY_HASH=True,original_raw_terminal_dual_unmodified=True,no_numeric_dual_bound_evaluation_yet=True,no_old_controls_model_solver_source_NN_raster_native_audits_replayed=True)
dump(O/'report.json',result)
