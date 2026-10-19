"""NEW observable-product maps within existing degree4 tensor columns only."""
from fractions import Fraction as F
from math import comb
from collections import Counter

def requested(left,right):
 out=Counter(left or ());out.update(right or ());return dict(sorted(out.items()))
def available(scopes,req):
 if any(type(p)is not int or p<0 or p>4 for p in req.values()):return []
 return sorted([s for s in scopes if set(req)<=set(s)],key=lambda s:(len(s),s))
def moment_map(scope,entries,req):
 if not set(req)<=set(scope)or any(type(p)is not int or not 0<=p<=4 for p in req.values()):raise ValueError('UNRESOLVED_unavailable_power_or_scope')
 out={}
 for idx,col in entries.items():
  z=F(1)
  for name,k in zip(scope,idx,strict=True):
   r=req.get(name,0);z*=F(comb(k,r),comb(4,r))if k>=r else F(0)
  if z:out[col]=z
 return out
def mass_variance(m0,m1,m2):
 if m0<0:return dict(classification='UNRESOLVED_signed_mass_NOT_positive_measure',mass=m0)
 if m0==0:return dict(classification='ZERO_positive_measure_ONLY'if m1==m2==0 else'ZERO_mass_nonzero_moments_NOT_positive_measure',mass=m0)
 return dict(classification='UNNORMALIZED_variance_necessity_ONLY',mass=m0,variance_numerator=m0*m2-m1*m1)
def same_domain(canonical,copied,names):return all(name in canonical and name in copied and canonical[name]==copied[name]for name in names)
