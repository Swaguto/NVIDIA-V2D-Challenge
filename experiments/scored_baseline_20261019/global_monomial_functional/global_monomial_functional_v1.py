"""Symbolic quotient Gram functional; no moment values or solver calls.

Original variables and Y_ij=t_i*t_j have canonical polynomial exponents.
Missing monomials are shared existential symbols, never observed values.
"""
from fractions import Fraction as F
from collections import Counter
from itertools import combinations_with_replacement

def basis(variables,pairs):
 n=len(variables);zero=(0,)*n;out=[zero]
 for name in variables:out.append(tuple(int(name==v)for v in variables))
 for pair in pairs:
  co=Counter(pair);out.append(tuple(co.get(v,0)for v in variables))
 return out

def classes(B):
 out={}
 for i,j in combinations_with_replacement(range(len(B)),2):
  e=tuple(a+b for a,b in zip(B[i],B[j],strict=True));out.setdefault(e,[]).append((i,j))
 return {k:out[k]for k in sorted(out)}
def sparse_equal_mod_rows(left,right,certificate,rows):
 d=dict(left)
 for col,v in right.items():d[col]=d.get(col,F(0))-v
 d={k:v for k,v in d.items()if v};derived={};bound=F(0)
 for row,z in certificate.items():
  for col,v in rows[row]['coefficients'].items():derived[col]=derived.get(col,F(0))+z*v
  bound+=z*rows[row]['bound']
 return d=={k:v for k,v in derived.items()if v}and bound==0

def monomial_bound(mass,positive_measure):
 if not positive_measure or mass<0:return dict(classification='UNRESOLVED_signed_or_unknown_measure_NO_unit_box')
 return dict(classification='POSITIVE_cube_measure_bound_ONLY',lower=F(0),upper=mass,unit_mass_REQUIRED_for_upper_one=mass==1)
def evaluate_monomial(e,t):
 z=F(1)
 for r,x in zip(e,t,strict=True):z*=x**r
 return z

def assemble_maps(B,class_maps):
 n=len(B);matrix=[[None]*n for _ in range(n)]
 for e,cells in classes(B).items():
  for i,j in cells:matrix[i][j]=matrix[j][i]=dict(class_maps[e])
 return matrix

def adjoint(matrix,D):
 out={}
 for i,row in enumerate(matrix):
  for j,co in enumerate(row):
   for col,v in co.items():out[col]=out.get(col,F(0))+D[i][j]*v
 return {k:v for k,v in out.items()if v}
