"""NEW exact fixed-raw-dual certificate for the SAME frozen conic model.

For Ax+s=b,s in K, x in necessary constructor box B and fixed raw y:
z_rho >= inf_B (e_rho+A'y)'x - b'y - D.
D bounds negative y's slack pairing, including EVERY orthant/SOC defect.
No raw value is repaired; the original objective/constraints are unchanged.
This proves facts about a conditional numeric construction only, never source
accuracy, a task family, measured forces or trajectory qualification.
"""
from pathlib import Path
from fractions import Fraction as F
from math import isqrt
import datetime,hashlib,json,os
import numpy as np
from scipy import sparse
S=Path(__file__).resolve().parents[1];C=S/'compute';M=S/'e037-hard-raster-model-v1';I=S/'e037-structured-all-row-preflight-v1';E=S/'e037-whole-row-conic-evaluation-v1';P=S/'e037-exact-terminal-dual-map-v1'
BIT_CAP=8192;ROOT_BITS=96;NN=4613807;ROWS=4618162;COLS=1058
q=lambda x:F.from_float(float(x))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def norm(x):
 if isinstance(x,F):return str(x)
 if isinstance(x,np.ndarray):return norm(x.tolist())
 if isinstance(x,np.generic):return norm(x.item())
 if isinstance(x,dict):return {str(k):norm(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [norm(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return {'nonfinite':repr(x)}
 return x
def dump(p,x):
 with Path(p).open('w') as f:json.dump(norm(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def cap(x):
 x=F(x)
 if max(abs(x.numerator).bit_length(),x.denominator.bit_length())>BIT_CAP:raise ArithmeticError('NEW8192bit certificate cap; no extension')
 return x
def exact_dot(a,b):
 a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
 if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():return dict(status='UNRESOLVED_nonfinite_or_shape',value=None)
 active=(a!=0)&(b!=0);bins={}
 for x,y in zip(a[active],b[active]):
  n,d=float(x).as_integer_ratio();m,e=float(y).as_integer_ratio();k=d.bit_length()+e.bit_length()-2
  bins[k]=bins.get(k,0)+n*m
 if bins:
  k=max(bins);integer=sum(n<<(k-j) for j,n in bins.items());value=F(integer,1<<k)
 else:value=F(0)
 return dict(status='exact_dyadic_dot',value=value,entries=a.size,nonzero_products=int(np.count_nonzero(active)),exact_zero_products=int(a.size-np.count_nonzero(active)),denominator_exponent_bins={str(k):str(v) for k,v in sorted(bins.items())},max_result_bits=max(abs(value.numerator).bit_length(),value.denominator.bit_length()))
def root_upper(x):
 x=F(x)
 if x<0:raise ValueError('negative squared norm')
 scaled=x.numerator<<(2*ROOT_BITS);k=isqrt(scaled//x.denominator)
 if k*k*x.denominator<scaled:k+=1
 return F(k,1<<ROOT_BITS),k

def load_csr():
 d=np.load(I/'NEW-CSR-data-RETURN.npy',mmap_mode='r');j=np.load(I/'NEW-CSR-indices-RETURN.npy',mmap_mode='r');p=np.load(I/'NEW-CSR-indptr-RETURN.npy',mmap_mode='r');return sparse.csr_matrix((d,j,p),shape=(ROWS,COLS),copy=False)
def row(A,b,r):
 z=A[r];return [(int(i),q(v)) for i,v in zip(z.indices,z.data) if v!=0],q(b[r])
def support_slack_bound(A,b,r,magnitudes):
 indices,constant=row(A,b,r);return cap(abs(constant)+sum(abs(v)*magnitudes[j] for j,v in indices))
def constructor_bounds(A,b):
 bounds=[None]*COLS;proof=[]
 for f in range(151):
  base=NN+4*598+8*f
  for top,cols in [(base,range(7*f,7*f+3)),(base+4,range(7*f+3,7*f+6))]:
   ts,T=row(A,b,top)
   if ts or T<=0:raise ValueError('nonconstant/malformed saved norm top')
   for offset,col in enumerate(cols):
    r=top+1+offset;items,beta=row(A,b,r)
    if len(items)!=1 or items[0][0]!=col or not items[0][1]:raise ValueError('missing coordinate proof')
    coef=items[0][1];lo,hi=sorted([(beta-T)/coef,(beta+T)/coef]);bounds[col]=(lo,hi)
    proof.append(dict(column=col,kind='constant_top_SOC_coordinate',top_row=top,component_row=r,top=T,component_rhs=beta,coefficient=coef,lower=lo,upper=hi))
  sr=66895+f;items,upper=row(A,b,sr);col=7*f+6
  if items!=[(col,F(1))] or upper<0:raise ValueError('sigma upper malformed')
  cr=4617407+5*f
  expected=[([(col,F(-1))],F(1))]+[([(7*f+3+j,F(-2))],F(0)) for j in range(3)]+[([(col,F(-1))],F(-1))]
  actual=[row(A,b,cr+j) for j in range(5)]
  if actual!=expected:raise ValueError('lift proof malformed')
  bounds[col]=(F(0),upper);proof.append(dict(column=col,kind='SOC5_sigma_nonnegative_and_saved_upper',upper_row=sr,lift_start=cr,lower=F(0),upper=upper,identity='(1+sigma)^2-(sigma-1)^2-4sum(u_j^2)=4sigma-4sum(u_j^2)'))
 if row(A,b,0)!=([(1057,F(-1))],F(0)) or row(A,b,1)!=([(1057,F(1))],F(1)):raise ValueError('phase rows malformed')
 bounds[1057]=(F(0),F(1));proof.append(dict(column=1057,kind='phase_saved_rows',rows=[0,1],lower=F(0),upper=F(1)))
 return bounds,proof
