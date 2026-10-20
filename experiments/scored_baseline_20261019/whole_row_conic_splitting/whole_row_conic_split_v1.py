"""NEW direct conic splitting algebra for the frozen all-row canonical.

min .5*x'P*x+q'x, Ax+s=b, s in product K (orthant,SOC4,SOC5).
With scaled dual u and rho=1, N=P+A'A; x=N^-1(A'(b-s-u)-q),
s=Pi_K(b-Ax-u), u+=Ax+s-b. Positive u is in K* at an exact step;
no multiplier is a physical force. This interface never selects row minima.
"""
from pathlib import Path
import json,os,datetime,hashlib
import numpy as np
from scipy import sparse
S=Path(__file__).resolve().parents[1];C=S/'compute';M=S/'e037-hard-raster-model-v1';I=S/'e037-structured-all-row-preflight-v1';N=S/'e037-whole-row-conic-normal-v1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def normalized(x):
 if isinstance(x,np.ndarray):return normalized(x.tolist())
 if isinstance(x,np.generic):return normalized(x.item())
 if isinstance(x,dict):return {str(k):normalized(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [normalized(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return {'nonfinite':repr(x)}
 return x
def dump(p,x):
 with Path(p).open('w') as f:json.dump(normalized(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def save(p,**x):
 with Path(p).open('wb') as f:np.savez(f,**x);f.flush();os.fsync(f.fileno())
def load_frozen():
 meta=json.loads((I/'NEW-structured-layout-RETURN.json').read_text())
 d=np.load(I/'NEW-CSR-data-RETURN.npy',mmap_mode='r');j=np.load(I/'NEW-CSR-indices-RETURN.npy',mmap_mode='r');p=np.load(I/'NEW-CSR-indptr-RETURN.npy',mmap_mode='r')
 A=sparse.csr_matrix((d,j,p),shape=tuple(meta['shape']),copy=False)
 with np.load(M/'NEW-canonical-vectors-RETURN.npz') as z:b=z['b'];q=z['q']
 P=sparse.load_npz(M/'NEW-canonical-P-RETURN.npz');return A,b,q,P,meta

def project_product(v,nn,cones):
 """Whole K projection, retaining all coordinates. Exact real formula;

Numerical norm/sqrt/branch returns remain raw and are not exact certificates.
All nonfinite or malformed inputs fail with explicit unresolved returns.
Strict/open source flags stay closed here and require actual POST checks.
 """
 v=np.asarray(v,float);out=np.empty_like(v);branches=[]
 if not np.isfinite(v).all():return dict(status='UNRESOLVED_nonfinite',value=v.copy(),branches=[])
 out[:nn]=np.maximum(v[:nn],0.)
 cursor=nn
 for a,k,kind in cones:
  if a!=cursor or k<2 or a+k>len(v):return dict(status='UNRESOLVED_malformed_incidence',value=v.copy(),branches=branches)
  w=v[a:a+k];t=float(w[0]);norm=float(np.linalg.norm(w[1:]))
  if norm<=t:out[a:a+k]=w;region='inside_or_boundary'
  elif norm<=-t:out[a:a+k]=0.;region='polar_or_boundary'
  else:
   top=.5*(norm+t);out[a]=top;out[a+1:a+k]=(top/norm)*w[1:];region='outside'
  branches.append(dict(start=a,size=k,region=region,norm=norm,t=t));cursor=a+k
 if cursor!=len(v):return dict(status='UNRESOLVED_incomplete_incidence',value=v.copy(),branches=branches)
 return dict(status='projected_ALL_product_coordinates',value=out,branches=branches)

def cone_diagnostics(v,nn,cones):
 out=[float(np.min(v[:nn]))]
 out.extend(float(v[a]-np.linalg.norm(v[a+1:a+k])) for a,k,_ in cones)
 return np.array(out)

def coordinate_witnesses(A,meta):
 # Proof submatrix only. ALL other rows remain used by every normal/product.
 # Last302 old cones are translation/rotation norm pairs, 8 rows per frame.
 witnesses=[]
 for f in range(151):
  base=4613807+4*598+8*f
  for j in range(3):witnesses.append((7*f+j,base+1+j))
  for j in range(3):witnesses.append((7*f+3+j,base+5+j))
  witnesses.append((7*f+6,66895+f))
 witnesses.append((1057,1))
 return witnesses
