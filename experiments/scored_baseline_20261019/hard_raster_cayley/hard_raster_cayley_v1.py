"""NEW all-cell hard raster proposal factors; no old source polynomial replay.

Three object-local Cayley coordinates t and lifted translation eta=D*delta,
D=1+t.t. Every finite t is a proper rotation. The chart omits angle pi; it is a
construction restriction, never measured orientation or original authority.
"""
import math,numpy as np
from fractions import Fraction as F

def skew(a):
 x,y,z=np.asarray(a,float);return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
def cayley(t):
 a=np.asarray(t,float)
 if a.shape!=(3,)or not np.isfinite(a).all():raise ValueError('finite three-coordinate Cayley input')
 d=1+float(a@a);N=(1-float(a@a))*np.eye(3)+2*np.outer(a,a)+2*skew(a);return N/d,d

def pow2_up(x):
 if not np.isfinite(x)or x<=0:raise ValueError('positive finite magnitude bound')
 m,e=math.frexp(float(x));return math.ldexp(1.,e-1 if m==.5 else e)

def curvature_scale(A,b,V,K,r):
 Ba=pow2_up(max(float(np.max(abs(A))),np.finfo(float).tiny));Bb=pow2_up(max(float(np.max(abs(b))),np.finfo(float).tiny));Bv=pow2_up(max(float(np.max(abs(V))),np.finfo(float).tiny));Bw=pow2_up(max(float(abs(K[0,0])),float(abs(K[1,1])),float(abs(K[0,2])+20001),float(abs(K[1,2])+20001),1.));L=pow2_up(3*Bw*Bb+81*Bw*Ba*Bv+1.);s=float(np.nextafter(np.sqrt(L/r),np.inf));proof=F.from_float(r)*F.from_float(s)**2>=F.from_float(L)
 return dict(A_entry_bound=Ba,b_entry_bound=Bb,V_entry_bound=Bv,w_entry_bound=Bw,uniform_negative_curvature_bound=L,scale=s,exact_scale_square_bound=proof,derivation='||H||2 <= |w.b| +3||w.A||1||v||1 <= 3BwBb+81BwBaBv; r*s*s >=L exactly on saved binary inputs')

def frame_rows(A,b,L,V,K,integer_uv,r,s):
 """Rows a*x <= rhs for x=(eta/r, u=s*t, sigma>=u.u).

All four pixel-cell slots plus every strict vertex-depth slot retained. Open
faces are closed for the proposal canonical ONLY; actual integer-cell equality
and original strict depth remain mandatory on the raw nonlinear proposal.
 """
 n=len(V);weights=[];active=[];strict=[];kinds=[]
 for axis in range(2):
  k=integer_uv[:,axis];f=K[axis,axis];c=K[axis,2]
  lower=np.where(k>0,k,np.where(k<0,k-1,-1)).astype(float);upper=np.where(k>0,k+1,np.where(k<0,k,1)).astype(float)
  lower=np.where(k==20000,20000,lower);upper=np.where(k==-20000,-20000,upper)
  for side in range(2):
   W=np.zeros((n,3));W[:,axis]=f if side==0 else-f;W[:,2]=(c-lower)if side==0 else(upper-c);on=(k!=-20000)if side==0 else(k!=20000);op=(k<=0)if side==0 else(k>=0);op=np.where(abs(k)==20000,False,op);weights.append(W);active.append(on);strict.append(op&on);kinds.append(np.full(n,1+axis*2+side,np.int8))
 # Vertex strict positive camera depth, separate from pixel halfspaces.
 weights.append(np.tile([0.,0.,1.],(n,1)));active.append(np.ones(n,bool));strict.append(np.ones(n,bool));kinds.append(np.full(n,5,np.int8))
 W=np.concatenate(weights);on=np.concatenate(active);opened=np.concatenate(strict);kind=np.concatenate(kinds);VV=np.tile(V,(5,1));AV=np.einsum('ij,vj->vi',A,V,optimize=False)+b
 const=np.einsum('vi,vi->v',W,np.tile(AV,(5,1)),optimize=False);const[4*n:]-=.01
 # -2 A[v]x t; explicit component expansion avoids old numerical source calls.
 Rj=np.empty((n,3,3))
 for j,(a,c)in enumerate([(1,2),(2,0),(0,1)]):Rj[:,:,j]=2*(V[:,a,None]*A[None,:,c]-V[:,c,None]*A[None,:,a])
 linear=np.einsum('vi,vij->vj',W,np.tile(Rj,(5,1,1)),optimize=False)
 coeff=np.empty((5*n,7));coeff[:,:3]=-W@L;coeff[:,3:6]=-linear/(r*s);coeff[:,6]=1.;rhs=const/r
 coeff[~on]=0.;rhs[~on]=0.
 return dict(coefficients=coeff,rhs=rhs,active=on,strict_open=opened,row_kind=kind,weights=W)

def exact_polynomial(A,b,L,v,t,eta,w,depth=False):
 # Independent rational meaning helper used only in NEW synthetic contracts.
 def q(x):return F.from_float(float(x))
 tt=list(map(q,t));ee=list(map(q,eta));vv=list(map(q,v));D=1+sum(x*x for x in tt);N=[[(1-sum(x*x for x in tt))*(1 if i==j else 0)+2*tt[i]*tt[j]+2*q(skew(t)[i,j])for j in range(3)]for i in range(3)];C=[sum(q(A[i,j])*N[j][k]*vv[k]for j in range(3)for k in range(3))+q(b[i])*D+sum(q(L[i,j])*ee[j]for j in range(3))for i in range(3)];return sum(q(w[i])*C[i]for i in range(3))-(q(.01)*D if depth else 0)
