"""Independent NEW exact certificate product/sign/bound audit.

Alternative IEEE754 bit-decoded fixed 2^-2148 superaccumulator verifies ALL
new coefficient products; the producer's denominator-bin algorithm is NOT
called. No completed model/solver/source/NN/raster/native audit is replayed.
"""
from exact_terminal_dual_bound_v1 import *
O=S/'e037-exact-terminal-dual-audit-v1';V=S/'e037-exact-terminal-dual-evaluation-v1'

def independent_bit_dot(a,b):
 a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
 if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('bit-dot unknown input')
 mask=(a!=0)&(b!=0);ab=a[mask].view(np.uint64);bb=b[mask].view(np.uint64);total=0
 for av,bv in zip(ab,bb):
  x=int(av);y=int(bv);ex=(x>>52)&2047;ey=(y>>52)&2047;mx=x&((1<<52)-1);my=y&((1<<52)-1)
  if ex:mx|=1<<52;px=ex-1075
  else:px=-1074
  if ey:my|=1<<52;py=ey-1075
  else:py=-1074
  term=(mx*my)<<(px+py+2148)
  total=total-term if (x^y)>>63 else total+term
 return F(total,1<<2148),int(mask.sum())
# TWO NEW independent bit-representation contracts, no episode input/control replay.
a=np.array([float.fromhex('0x1.0000000000001p+200'),-float.fromhex('0x1.8p-400'),3*np.nextafter(0.,1.)]);b=np.array([2.**-100,2.**200,2.**500]);actual,n=independent_bit_dot(a,b);expected=sum(q(x)*q(y) for x,y in zip(a,b));dump(O/'NEW-bit-decoder-contract1-RETURN.json',dict(actual=actual,expected=expected,terms=n));assert actual==expected and n==3
actual,n=independent_bit_dot(np.array([np.finfo(float).max,-0.]),np.array([np.nextafter(0.,1.),-2.]));expected=q(np.finfo(float).max)*q(np.nextafter(0.,1.));dump(O/'NEW-bit-decoder-contract2-RETURN.json',dict(actual=actual,expected=expected,terms=n));assert actual==expected and n==1
assert (V/'ONE-actual-exact-bound-RETURN.json').exists()
A=sparse.load_npz(M/'NEW-canonical-A-RETURN.npz');csr=load_csr()
with np.load(M/'NEW-canonical-vectors-RETURN.npz') as z:b=z['b']
with np.load(E/'ONE-actual-raw-terminal-RETURN.npz') as z:y=z['u']
bs=json.loads((P/'NEW-exact-rational-constructor-bounds-RETURN.json').read_text());magnitudes=[F(z['magnitude']) for z in bs];support=F(0);entries=0;nonzero=0
for j in range(COLS):
 start,end=A.indptr[j:j+2];value,n=independent_bit_dot(A.data[start:end],y[A.indices[start:end]]);record=json.loads((V/f'column-{j:04d}-NEW-exact-dot-RETURN.json').read_text());termrec=json.loads((V/f'column-{j:04d}-NEW-exact-support-RETURN.json').read_text());coef=value+(1 if j==1057 else 0);lo=F(bs[j]['lower']);hi=F(bs[j]['upper']);term=min(coef*lo,coef*hi)
 proof=dict(column=j,independent_exact_ATy=value,returned_exact_ATy=record['value'],support_term=term,all_stored_entries=int(end-start),nonzero_products=n,exact=record['status']=='exact_dyadic_dot' and F(record['value'])==value and F(termrec['phase_coefficient'])==coef and F(termrec['exact_box_infimum_term'])==term)
 dump(O/f'column-{j:04d}-NEW-independent-product-RETURN.json',proof);assert proof['exact'] and record['entries']==end-start and record['nonzero_products']==n
 support+=term;entries+=end-start;nonzero+=n
by=F(0);cursor=0
for path in sorted(V.glob('by-*-NEW-exact-dot-RETURN.json')):
 rec=json.loads(path.read_text());start=rec['row_start'];end=rec['row_end'];assert start==cursor
 value,n=independent_bit_dot(b[start:end],y[start:end]);proof=dict(row_start=start,row_end=end,independent_value=value,returned_value=rec['value'],nonzero_products=n,exact=F(rec['value'])==value)
 dump(O/path.name.replace('NEW-exact-dot','NEW-independent-bit-dot'),proof);assert proof['exact'] and rec['nonzero_products']==n;by+=value;cursor=end
assert cursor==ROWS and entries==32306054
Dorth=F(0);found=[]
for path in sorted(V.glob('orthant-*-NEW-exact-defect-RETURN.json')):
 records=json.loads(path.read_text());new=[]
 for rec in records:
  r=rec['row'];begin,end=csr.indptr[r:r+2];Sbound=abs(q(b[r]))
  for j,a in zip(csr.indices[begin:end],csr.data[begin:end]):Sbound+=abs(q(a))*magnitudes[int(j)]
  weight=-q(y[r]);term=weight*Sbound;new.append(dict(row=r,necessary_upper=Sbound,defect_term=term,exact=weight>0 and q(y[r])==F(rec['exact_raw_y']) and Sbound==F(rec['necessary_slack_upper']) and term==F(rec['defect_pairing_upper'])));found.append(r);Dorth+=term
 dump(O/path.name.replace('NEW-exact-defect','NEW-independent-defect'),new);assert all(t['exact'] for t in new)
assert np.array_equal(np.asarray(found),np.flatnonzero(y[:NN]<0))
Dsoc=F(0);mp=json.loads((P/'NEW-complete-certificate-map-RETURN.json').read_text())
for k,c in enumerate(mp['SOC_defect']):
 rec=json.loads((V/f'cone-{k:04d}-NEW-exact-defect-RETURN.json').read_text());a=c['start'];size=c['size'];square=sum(q(z)**2 for z in y[a+1:a+size]);integer=int(rec['grid_integer']);up=F(integer,2**96);d=max(up-q(y[a]),F(0));begin,end=csr.indptr[a:a+2];Sbound=abs(q(b[a]))
 for j,z in zip(csr.indices[begin:end],csr.data[begin:end]):Sbound+=abs(q(z))*magnitudes[int(j)]
 term=d*Sbound;exact=F(rec['exact_tail_square'])==square and F(rec['outward_norm_upper'])==up and F(rec['whole_cone_dual_defect'])==d and F(rec['necessary_slack_top_upper'])==Sbound and F(rec['defect_pairing_upper'])==term and up*up>=square and (integer==0 or F(integer-1,2**96)**2<square)
 proof=dict(cone=k,ALL_components=size,independent_square=square,integer=integer,whole_cone_defect_term=term,exact=exact);dump(O/f'cone-{k:04d}-NEW-independent-defect-RETURN.json',proof);assert exact;Dsoc+=term
lower=support-by-Dorth-Dsoc;final=json.loads((V/'ONE-actual-exact-bound-RETURN.json').read_text());radius=q(json.loads((M/'NEW-canonical-metadata-RETURN.json').read_text())['radius'])
result=dict(schema='v2d.NEW-exact-terminal-dual-independent-certificate-audit.v1',NEW_bit_decoder_contracts=2,ALL32306054stored_coefficient_products_verified=True,nonzero_coefficient_products=nonzero,ALL4618162by_products_verified=True,ALL4613807orthant_defect_incidence_verified=True,negative_orthant_terms=len(found),ALL900SOC4_151SOC5_outward_root_and_defect_certificates_verified=True,ALL1058box_support_terms_verified=True,independent_box_support=str(support),independent_by=str(by),independent_Dorth=str(Dorth),independent_Dsoc=str(Dsoc),independent_normalized_phase_lower_bound=str(lower),independent_physical_phase_lower_m=str(radius*lower),raw_terminal_dual_SHA_UNCHANGED=sha(E/'ONE-actual-raw-terminal-RETURN.npz')==final['raw_terminal_dual_SHA'],positive_bound=lower>0,scope_ONLY_frozen_conditional_numeric_construction=True,no_model_solver_source_NN_raster_native_or_old_audit_replayed=True,no_source_truth_task_family_exclusion_or_trajectory_qualification=True)
dump(O/'ONE-NEW-independent-exact-certificate-RETURN.json',result)
assert F(final['normalized_phase_lower_bound'])==lower and F(final['physical_phase_lower_bound_m'])==radius*lower and F(final['exact_box_support'])==support and F(final['exact_by'])==by and F(final['exact_orthant_defect'])==Dorth and F(final['exact_whole_SOC_defect'])==Dsoc and result['raw_terminal_dual_SHA_UNCHANGED']
dump(O/'report.json',result)
