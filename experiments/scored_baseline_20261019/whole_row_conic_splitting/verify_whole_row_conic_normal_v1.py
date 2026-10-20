"""Independent NEW all-row normal/projection maps, not old model numeric audit.

Sources use their immutable local7 maps; other rows use independent rowwise
outer-product accumulation. Every input row contributes, including zero rows.
Long-double enclosures are conditional arithmetic diagnostics, not original
candidate tolerances. The exact expression is the factored ALL-row A'A map.
"""
from whole_row_conic_split_v1 import *
from fractions import Fraction as F
O=S/'e037-whole-row-conic-binding-v1'
A,b,q,P,meta=load_frozen()
with np.load(N/'NEW-ALL-row-normal-RETURN.npz') as z:G=z['normal']
with np.load(N/'NEW-Cholesky-constructor-RETURN.npz') as z:rawL=z['factor'];lower=bool(z['lower'])
ld=np.longdouble;E=np.zeros((1058,1058),dtype=ld);B=np.zeros_like(E)
source_st=67046;source_en=4613807
records=[]
for f in range(151):
 p=M/f'frame-{f:03d}-NEW-hard-cell-coefficients-RETURN.npz'
 with np.load(p) as z:V=z['coefficients'].astype(ld)
 e=np.zeros((7,7),dtype=ld);a=np.zeros_like(e)
 for j in range(7):
  for k in range(j,7):
   product=V[:,j]*V[:,k];e[j,k]=product.sum(dtype=ld);a[j,k]=abs(product).sum(dtype=ld)
 E[7*f:7*f+7,7*f:7*f+7]+=e;B[7*f:7*f+7,7*f:7*f+7]+=a
 records.append(dict(frame=f,rows=len(V),all49_normal_coefficient_maps=True,input_sha256=sha(p)))
 save(O/f'f{f:03d}-NEW-normal-block-binding-RETURN.npz',upper_expected=e,upper_absolute_terms=a)
# Independently accumulate every nonsource row via length-grouped outer maps.
rows=np.r_[np.arange(source_st,dtype=np.int64),np.arange(source_en,A.shape[0],dtype=np.int64)];lengths=np.diff(A.indptr)[rows]
counts=[]
for n in np.unique(lengths):
 rr=rows[lengths==n];counts.append(dict(row_length=int(n),rows=int(len(rr))))
 if not n:continue
 slots=A.indptr[rr,None]+np.arange(n)[None,:];cols=A.indices[slots];vals=A.data[slots].astype(ld)
 assert np.all(np.diff(cols,axis=1)>0) if n>1 else True
 for j in range(n):
  for k in range(j,n):
   product=vals[:,j]*vals[:,k]
   np.add.at(E,(cols[:,j],cols[:,k]),product);np.add.at(B,(cols[:,j],cols[:,k]),abs(product))
E+=np.triu(E,1).T;B+=np.triu(B,1).T
pd=P.diagonal().astype(ld);E[np.diag_indices(1058)]+=pd;B[np.diag_indices(1058)]+=abs(pd)
# (4*m+16)*eps64 strictly exceeds combined normal summation error with
# round-to-nearest finite binary inputs and any summation order, no overflow.
eps=F(1,2**52);g=(4*A.shape[0]+16)*eps;gf=ld(g.numerator)/ld(g.denominator)
bounds=np.nextafter(gf*B,np.longdouble(np.inf));error=abs(G.astype(ld)-E)
save(O/'NEW-ALL-normal-binding-RETURN.npz',independent_expected=E,absolute_term_sums=B,outward_bounds=bounds,errors=error)
dump(O/'NEW-ALL-row-incidence-RETURN.json',dict(source_records=records,nonsource_length_groups=counts,ALL_rows=151*30111+len(rows),exact_gamma=str(g),conditional_RN_finite_assumptions=True,source_and_nonsource_maps_disjoint_complete=True))
assert 151*30111+len(rows)==4618162 and np.all(error<=bounds) and np.isfinite(G).all()
witness=json.loads((N/'NEW-coordinate-SPD-witness-RETURN.json').read_text());square=[]
for rec in witness:
 col=rec['column'];row=rec['row'];indices=A.indices[A.indptr[row]:A.indptr[row+1]];values=A.data[A.indptr[row]:A.indptr[row+1]]
 nz=[(int(i),float(v)) for i,v in zip(indices,values) if v!=0]
 assert nz==[tuple(rec['nonzero'][0])] and len(nz)==1 and nz[0][0]==col
 square.append(F.from_float(nz[0][1])**2)
assert len(square)==1058 and min(square)>0
# Factor residual is diagnostic of actual SAVED factor, no refactorization.
L=np.tril(rawL);reconstruction=L@L.T;factorerr=abs(reconstruction-G);scale=np.maximum(1.,abs(L)@abs(L).T+abs(G));relative=float(np.max(factorerr/scale))
save(O/'NEW-SAVED-factor-binding-RETURN.npz',factor_residual=factorerr,comparison_scale=scale)
assert lower and np.isfinite(L).all() and np.all(np.diag(L)>0) and relative<1e-12
mapdata=json.loads((N/'NEW-complete-factored-normal-map-RETURN.json').read_text())
for a in ['data','indices','indptr']:assert mapdata[f'input_{a}_sha256']==sha(I/f'NEW-CSR-{a}-RETURN.npy')
assert mapdata['P_prefix_BY_HASH']==sha(M/'NEW-canonical-P-RETURN.npz')
cursor=4613807
for k,(a,n,kind) in enumerate(mapdata['cones']):assert a==cursor and n==(4 if k<900 else 5);cursor+=n
assert cursor==4618162 and len(mapdata['cones'])==1051
# Explicit component row bounds are NECESSARY to the frozen cone construction;
# they are not original object authority or candidate acceptance tolerances.
old=json.loads((M/'NEW-canonical-metadata-RETURN.json').read_text());U=old['scale']*np.pi/24
lo=np.tile(np.r_[np.full(3,-.125),np.full(3,-U),0.],151);hi=np.tile(np.r_[np.full(3,.125),np.full(3,U),U*U],151);lo=np.r_[lo,0.];hi=np.r_[hi,1.]
save(O/'NEW-constructor-component-outer-bounds-RETURN.npz',lower=lo,upper=hi)
dump(O/'report.json',dict(schema='v2d.NEW-whole-row-conic-independent-binding.v1',ALL4618162_normal_row_maps=True,source_local_blocks=151,nonsource_rows=len(rows),normal_coefficient_entries=1058*1058,all_within_conditional_operationwise_bounds=True,max_normal_error=float(error.max()),coordinate_SPD_witnesses=1058,exact_SPD_lower_bound=str(min(square)),all900SOC4_151SOC5_and_strict_pointers_retained_BY_HASH=True,actual_saved_factor_relative_error=relative,diagnostics_NOT_candidate_tolerances=True,source_NN_raster_old_model_solver_NOT_replayed=True,no_episode_point_or_method_evaluation=True))
