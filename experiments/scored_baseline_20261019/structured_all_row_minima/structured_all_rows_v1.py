"""Frozen all-row residual interface. No row deletion or source operator calls.

For every finite affine block, min(b-Ax)>=0 iff all its rows hold. At a
minimum tie the directional derivative is the minimum of branch derivatives;
the returned arithmetic mean of ALL tied slopes is a concave supergradient
for SEARCH only. It is not a derivative/stationarity/acceptance certificate.
"""
from pathlib import Path
import datetime, hashlib, json, os
import numpy as np
from scipy import sparse
S=Path(__file__).resolve().parents[1]
C=S/'compute'
MODEL=S/'e037-hard-raster-model-v1'
PREFLIGHT=S/'e037-structured-all-row-preflight-v1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(1048576),b''):h.update(z)
 return h.hexdigest()
def normjson(x):
 if isinstance(x,np.ndarray):return normjson(x.tolist())
 if isinstance(x,np.generic):return normjson(x.item())
 if isinstance(x,dict):return {str(k):normjson(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [normjson(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return {'nonfinite':repr(x)}
 return x
def dump(p,x):
 with Path(p).open('w') as f:
  json.dump(normjson(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def arrays(p,**x):
 with Path(p).open('wb') as f:
  np.savez(f,**x);f.flush();os.fsync(f.fileno())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def layout(m):
 # Exact row incidence of the SAVED canonical. Singletons also stay present.
 g=[]
 for i in range(2):g.append((i,i+1,'phase_bound',-1))
 for f in range(151):g.append((2+441*f,2+441*(f+1),'geometry_ALL_pairs',f))
 for i in range(66593,m['hard_source_start']):g.append((i,i+1,'frozen_search_or_sigma_upper',-1))
 for f in range(151):
  a=m['hard_source_start']+f*m['hard_source_rows_per_frame']
  g.append((a,a+m['hard_source_rows_per_frame'],'source_ALL_cell_depth_rows',f))
 cones=[(m['nonnegative']+4*k,4,'original_SOC4') for k in range(900)]
 cones.extend((m['psi_SOC_start']+5*f,5,'lift_SOC5') for f in range(151))
 return g,cones

def reduce_all(A,b,x,groups,cones):
 """EVERY original row is evaluated once. Nonfinite rows remain UNKNOWN."""
 r=b-A@x
 n=A.shape[1];vals=[];jac=[];ties=[];offset=[0];unknown=[]
 for a,z,_,_ in groups:
  block=r[a:z]
  if not np.isfinite(block).all():
   vals.append(np.nan);jac.append(np.full(n,np.nan));unknown.append((a,z));offset.append(len(ties));continue
  v=float(np.min(block));ix=np.flatnonzero(block==v)+a
  vals.append(v);jac.append(-np.asarray(A[ix].sum(axis=0)).ravel()/len(ix))
  ties.extend(ix.tolist());offset.append(len(ties))
 for a,k,_ in cones:
  block=r[a:a+k]
  if not np.isfinite(block).all():
   vals.append(np.nan);jac.append(np.full(n,np.nan));unknown.append((a,a+k));continue
  length=float(np.linalg.norm(block[1:]));vals.append(float(block[0]-length))
  J=-A[a].toarray().ravel()
  if length!=0:J+=np.asarray(A[a+1:a+k].T@(block[1:]/length)).ravel()
  jac.append(J)
 return dict(raw_rows=r,values=np.asarray(vals),search_J=np.asarray(jac),tie_rows=np.asarray(ties,dtype=np.int64),tie_offsets=np.asarray(offset,dtype=np.int64),unknown=unknown)

def acceptance_identity(sealed_id,source_ids,motion_passes,cells,strict_depth,model_finite):
 return bool(sealed_id and len(source_ids)==151 and all(x==sealed_id for x in source_ids) and len(motion_passes)==6 and all(motion_passes) and cells and strict_depth and model_finite)

def load_interface():
 meta=json.loads((PREFLIGHT/'NEW-structured-layout-RETURN.json').read_text())
 a=np.load(PREFLIGHT/'NEW-CSR-data-RETURN.npy',mmap_mode='r')
 j=np.load(PREFLIGHT/'NEW-CSR-indices-RETURN.npy',mmap_mode='r')
 p=np.load(PREFLIGHT/'NEW-CSR-indptr-RETURN.npy',mmap_mode='r')
 A=sparse.csr_matrix((a,j,p),shape=tuple(meta['shape']),copy=False)
 with np.load(MODEL/'NEW-canonical-vectors-RETURN.npz') as z:b=z['b'];q=z['q']
 P=sparse.load_npz(MODEL/'NEW-canonical-P-RETURN.npz')
 return A,b,q,P,meta['groups'],meta['cones'],meta
