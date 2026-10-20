"""Independent ALL new hard-raster canonical binding, streamed by frame."""
from pathlib import Path
import datetime,json,hashlib,os
from fractions import Fraction as F
import numpy as np
from scipy import sparse
S=Path(__file__).resolve().parents[1];C=S/'compute';I=S/'e037-hard-raster-model-v1';B=S/'e037-full-pose-preflight-v1';G=S/'e037-object-full-pose-factors-v1';O=S/'e037-hard-raster-binding-v2';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def save(n,d):
 with (O/n).open('w')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def arrays(n,**d):
 with (O/n).open('wb')as f:np.savez_compressed(f,**d);f.flush();os.fsync(f.fileno())
m=read(I/'NEW-canonical-metadata-RETURN.json');decl=read(I/'NEW-hard-raster-construction-BEFORE.json');A=sparse.load_npz(I/'NEW-canonical-A-RETURN.npz');baseA=sparse.load_npz(B/'NEW-canonical-A-RETURN.npz');old=read(B/'NEW-canonical-metadata-RETURN.json');stats=read(S/'e037-training-source-constraints-v1/all151-NEW-training-source-statistics.json')
with np.load(I/'NEW-canonical-vectors-RETURN.npz')as z:b=z['b'];q=z['q']
with np.load(B/'NEW-canonical-vectors-RETURN.npz')as z:baseb=z['b']
with np.load(G/'NEW-source-rotation-factors-RETURN.npz')as z:V=z['visual_vertices'];T=z['original_camera_from_object'];RC=z['camera_from_world_translation_linear'];K=z['K']
r=m['radius'];s=m['scale'];bound=m['curvature_scale'];u=F(1,2**53);gamma=float(32*u/(1-32*u));save('NEW-independent-binding-BEFORE.json',dict(model_declaration_sha256=sha(I/'NEW-hard-raster-construction-BEFORE.json'),canonical_A_sha256=sha(I/'NEW-canonical-A-RETURN.npz'),scope='ALL sourcecell/lift/prefix/cones/RHS/objective coefficients; no model/NN/raster/source/solver replay',operation_bound='gamma32 times per-operand absolute sums, local conditionalRN packing diagnostics NOT candidate tolerance'))
# Exact rational bound on the common quadratic curvature, no spectral solve.
Ba,Bb,Bv,Bw=[F.from_float(bound[k])for k in ['A_entry_bound','b_entry_bound','V_entry_bound','w_entry_bound']];lam=F.from_float(bound['uniform_negative_curvature_bound']);ss=F.from_float(r)*F.from_float(s)**2;ok=lam>=3*Bw*Bb+81*Bw*Ba*Bv+F.from_float(.01)and ss>=lam
save('NEW-exact-global-curvature-bound-RETURN.json',dict(exact_lower_required=str(3*Bw*Bb+81*Bw*Ba*Bv+F.from_float(.01)),lambda_bound=str(lam),r_scale_square=str(ss),passes=ok,conditional_saved_binary_numeric_model_only=True));assert ok
assert np.max(abs(T[:,:3,:3]))<=float(Ba)and np.max(abs(T[:,:3,3]))<=float(Bb)and np.max(abs(V))<=float(Bv)
maxcoef=0.;maxrhs=0.;sourcecoeffs=0;rhscount=0;nominal=0;strict_total=0
for f in range(151):
 part=Path(stats[f]['raster']['path']);assert sha(part)==stats[f]['raster']['sha256']
 with np.load(part)as z:target=z['uv_integer']
 assert np.max(abs(target))<=20000
 n=len(V);R=T[f,:3,:3];p=T[f,:3,3];L=RC[f];camera=V@R.T+p;absC=abs(V)@abs(R).T+abs(p);blocks=[];rhsblocks=[];bounds=[];rhsbounds=[];active=[];strict=[]
 for axis in range(2):
  k=target[:,axis];ff=float(K[axis,axis]);cc=float(K[axis,2]);lo=np.where(k>0,k,np.where(k<0,k-1,-1)).astype(float);hi=np.where(k>0,k+1,np.where(k<0,k,1)).astype(float);lo[k==20000]=20000;hi[k==-20000]=-20000
  for lower in [True,False]:
   w1=ff if lower else-ff;wz=cc-lo if lower else hi-cc;on=(k!=-20000)if lower else(k!=20000);op=((k<=0)if lower else(k>=0))&on;op[abs(k)==20000]=False;E=np.zeros((n,7));E[:,:3]=-(w1*L[axis]+wz[:,None]*L[2]);avec=w1*R[axis]+wz[:,None]*R[2];E[:,3:6]=-2*np.cross(V,avec)/(r*s);E[:,6]=1.;rr=(w1*camera[:,axis]+wz*camera[:,2])/r;absw1=abs(w1);abswz=abs(wz);cb=np.zeros_like(E);cb[:,:3]=gamma*(absw1*abs(L[axis])+abswz[:,None]*abs(L[2]));ap=absw1*abs(R[axis])+abswz[:,None]*abs(R[2]);rotbudget=np.empty((n,3))
   for j,(a,c)in enumerate([(1,2),(2,0),(0,1)]):rotbudget[:,j]=2*(abs(V[:,a])*ap[:,c]+abs(V[:,c])*ap[:,a])/(r*s)
   cb[:,3:6]=gamma*rotbudget;rb=gamma*(absw1*absC[:,axis]+abswz*absC[:,2])/r;E[~on]=0;rr[~on]=0;cb[~on]=0;rb[~on]=0;blocks.append(E);rhsblocks.append(rr);bounds.append(cb);rhsbounds.append(rb);active.append(on);strict.append(op)
 E=np.zeros((n,7));E[:,:3]=-L[2];E[:,3:6]=-2*np.cross(V,np.tile(R[2],(n,1)))/(r*s);E[:,6]=1.;rr=(camera[:,2]-.01)/r;cb=np.zeros_like(E);cb[:,3:6]=gamma*2*np.column_stack([abs(V[:,1])*abs(R[2,2])+abs(V[:,2])*abs(R[2,1]),abs(V[:,2])*abs(R[2,0])+abs(V[:,0])*abs(R[2,2]),abs(V[:,0])*abs(R[2,1])+abs(V[:,1])*abs(R[2,0])])/(r*s);rb=gamma*(absC[:,2]+.01)/r;blocks.append(E);rhsblocks.append(rr);bounds.append(cb);rhsbounds.append(rb);active.append(np.ones(n,bool));strict.append(np.ones(n,bool))
 expected=np.vstack(blocks+[np.r_[-L[2],np.zeros(3),1.][None,:]]);erhs=np.r_[np.concatenate(rhsblocks),(p[2]-.05)/r];cbound=np.vstack(bounds+[np.zeros((1,7))]);rbound=np.r_[np.concatenate(rhsbounds),gamma*(abs(p[2])+.05)/r];begin=m['hard_source_start']+f*m['hard_source_rows_per_frame'];end=begin+m['hard_source_rows_per_frame'];actual=np.empty_like(expected)
 for j in range(7):
  col=7*f+j;a0,a1=A.indptr[col:col+2];rows=A.indices[a0:a1];values=A.data[a0:a1];lo0=np.searchsorted(rows,begin);hi0=np.searchsorted(rows,end);assert np.array_equal(rows[lo0:hi0],np.arange(begin,end));actual[:,j]=values[lo0:hi0]
 de=actual-expected;db=b[begin:end]-erhs;passc=bool(np.all(abs(de)<=np.nextafter(cbound,np.inf)));passb=bool(np.all(abs(db)<=np.nextafter(rbound,np.inf)));maxc=float(np.max(abs(de)));maxb=float(np.max(abs(db)));arrays(f'NEW-frame-{f:03d}-ALL-cell-binding-residuals-RETURN.npz',coefficient_error=de,coefficient_RN_bound=cbound,RHS_error=db,RHS_RN_bound=rbound)
 with np.load(I/f'frame-{f:03d}-NEW-hard-cell-coefficients-RETURN.npz')as z:canonicalcopy=bool(np.array_equal(actual,z['coefficients'])and np.array_equal(b[begin:end],z['rhs']));act=z['active'];st=z['strict_open']
 expectedactive=np.r_[np.concatenate(active),True];expectedstrict=np.r_[np.concatenate(strict),True];flags=bool(np.array_equal(act,expectedactive)and np.array_equal(st,expectedstrict));save(f'NEW-frame-{f:03d}-binding-decision.json',dict(coefficients=actual.size,RHS_entries=len(db),max_coefficient_error=maxc,max_RHS_error=maxb,within_predeclared_operation_bounds=passc and passb,canonical_copy_exact=canonicalcopy,all_slot_and_strict_flags=flags,NOT_candidate_tolerance=True));assert passc and passb and canonicalcopy and flags and np.isfinite(actual).all()and np.isfinite(b[begin:end]).all()
 maxcoef=max(maxcoef,maxc);maxrhs=max(maxrhs,maxb);sourcecoeffs+=actual.size;rhscount+=len(db);nominal+=int(np.count_nonzero(b[begin:end]<0));strict_total+=int(st.sum())
# OLD canonical prefix BY HASH transformed only by the NEW variable map.
prefixerr=0.;prefixcount=0
for j in range(907):
 col=7*(j//6)+j%6 if j<906 else 1057;factor=1. if j==906 or j%6<3 else 2/s;orig=baseA.getcol(j);mapped=orig.indices.copy();mapped[mapped>=old['nonnegative']]+=151+m['hard_source_rows'];a0,a1=A.indptr[col:col+2];inds=A.indices[a0:a1];values=A.data[a0:a1];pos=np.searchsorted(inds,mapped);assert np.array_equal(inds[pos],mapped);err=values[pos]-orig.data*factor;prefixerr=max(prefixerr,float(np.max(abs(err),initial=0)));prefixcount+=len(err)
assert prefixerr==0;assert np.array_equal(b[:old['nonnegative']],baseb[:old['nonnegative']]);assert np.array_equal(b[m['nonnegative']:m['psi_SOC_start']],baseb[old['nonnegative']:])
# Each lifted epigraph cone and sigma upper row is independently bound.
for f in range(151):
 row=old['nonnegative']+f;col=7*f+6;assert A[row,col]==1 and b[row]==(s*np.pi/24)**2
 row=m['psi_SOC_start']+5*f;assert b[row]==1 and b[row+4]==-1 and A[row,col]==-1 and A[row+4,col]==-1
 for j in range(3):assert A[row+1+j,7*f+3+j]==-2 and b[row+1+j]==0
p=sparse.load_npz(I/'NEW-canonical-P-RETURN.npz');bp=sparse.load_npz(B/'NEW-canonical-P-RETURN.npz');expected=np.zeros(1058)
for j,value in enumerate(bp.diagonal()):
 c=7*(j//6)+j%6 if j<906 else 1057;fac=1. if j==906 or j%6<3 else 2/s;expected[c]=value*fac**2
assert np.array_equal(p.diagonal(),expected)and np.array_equal(q,np.r_[np.zeros(1057),1.]);save('NEW-prefix-lift-SOC-objective-binding-RETURN.json',dict(prefix_coefficients=prefixcount,prefix_error=prefixerr,old900SOC4_plus_NEW151SOC5=True,all151sigma_upper_rows=True,objective_matches=True,all_rows_columns_bound=True))
save('report.json',dict(schema='v2d.NEW-hard-raster-independent-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=m['rows'],columns=m['columns'],canonical_nnz=A.nnz,NEW_source_coefficient_cells=sourcecoeffs,NEW_source_RHS_entries=rhscount,OLD_prefix_coefficient_copies=prefixcount,NEW_source_max_coefficient_error=maxcoef,NEW_source_max_RHS_error=maxrhs,all_within_predeclared_gamma32_operator_bounds=True,all151frame_copy_bindings_exact=True,all_lift_cones_and_objective_bound=True,exact_uniform_curvature_certificate=True,nominal_negative_source_constants=nominal,strict_faces_actual_postgate_required=True,model_or_source_renderer_NN_solver_NOT_replayed=True,no_candidate_or_original_source_accuracy_claim=True));print(json.dumps(dict(success=True,rows=m['rows'],columns=m['columns'],source_coefficients=sourcecoeffs,max_coef_error=maxcoef,max_RHS_error=maxrhs)))
