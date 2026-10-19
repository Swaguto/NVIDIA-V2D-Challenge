"""New complementary-corner necessities for a SAVED diagnostic moment matrix.

No solver, moment completion, normalization or robot command construction.
All algebra is exact over supplied saved rational coefficients.
"""
from fractions import Fraction as F
from itertools import product
import conditional_moment_cones_v1 as c
REPRESENTATIVES=tuple(product((0,), (0,1), (0,1)))
ZERO=(0,0,0)
def plus(a,b):
 out=dict(a)
 for k,v in b.items():out[k]=out.get(k,F(0))+F(v)
 return {k:v for k,v in out.items()if v}
def times(a,b):
 out={}
 for i,x in a.items():
  for j,y in b.items():
   k=tuple(u+v for u,v in zip(i,j,strict=True));out[k]=out.get(k,F(0))+x*y
 return {k:v for k,v in out.items()if v}
def corner(v):
 p={ZERO:F(1)}
 for i,bit in enumerate(v):
  e=tuple(int(i==j)for j in range(3));p=times(p,{e:F(1)}if bit else {ZERO:F(1),e:F(-1)})
 return p
def paired(v):return plus(corner(v),corner(tuple(1-i for i in v)))
def polynomial_record(p):return [dict(powers=list(k),coefficient=c.rec(v))for k,v in sorted(p.items())]
def decode_polynomial(rows):return {tuple(r['powers']):c.dec(r['coefficient'])for r in rows}
def evaluate(p,point):return sum(v*product_value([F(x)**k for x,k in zip(point,powers,strict=True)])for powers,v in p.items())
def product_value(v):
 a=F(1)
 for b in v:a*=b
 return a
def matrix_map(p,indices):
 out={}
 for powers,v in p.items():
  support=[indices[i]for i,k in enumerate(powers)if k]
  if any(k>1 for k in powers)or len(support)>2:raise ValueError('UNRESOLVED_unavailable_higher_moment')
  entry=(0,0)if not support else (0,support[0])if len(support)==1 else tuple(sorted(support))
  out[entry]=out.get(entry,F(0))+v
 return {k:v for k,v in out.items()if v}
def map_record(mp):return [dict(matrix_entry=list(k),coefficient=c.rec(v))for k,v in sorted(mp.items())]
def expectation(mp,M):
 missing=[list(k)for k in mp if M[k[0]][k[1]] is None]
 if missing:return dict(classification='UNRESOLVED_absent_moments',missing_entries=missing,no_mean_products_or_completion_inserted=True)
 value=sum(v*M[i][j]for (i,j),v in mp.items())
 return dict(classification='NEGATIVE_rejects_THIS_matrix_as_cube_supported_positive_measure'if value<0 else 'NECESSARY_support_test_passes_ONLY',value=c.rec(value))
def mass_case(mass,moments):
 if mass<0:return 'SIGNED_mass_NOT_positive_measure'
 if mass==0:return 'ZERO_positive_measure_ONLY'if all(v==0 for v in moments)else 'ZERO_mass_nonzero_moments_NOT_positive_measure'
 return 'POSITIVE_mass_unnormalized_necessities_ONLY'
