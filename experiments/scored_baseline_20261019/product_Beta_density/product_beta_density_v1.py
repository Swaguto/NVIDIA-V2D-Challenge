"""One fixed continuous product-Beta density ansatz for a saved diagnostic Gram.

Exact rational parameter/moment algebra. Never normalize/repair saved entries.
A positive result concerns only the separate first/second matrix, not the
original higher tensor moments, source/time support or any robot trajectory.
"""
from fractions import Fraction as F
from itertools import combinations
import conditional_moment_cones_v1 as c

def recpoly(p):return [dict(powers=list(k),coefficient=c.rec(v))for k,v in sorted(p.items())]
def recparams(ps):return [{k:c.rec(v)for k,v in p.items()}for p in ps]
def beta_moment(a,b,r):
 if a<=0 or b<=0 or type(r)is not int or r<0:raise ValueError('UNRESOLVED_Beta_domain')
 z=F(1)
 for k in range(r):z*=F(a+k,a+b+k)
 return z

def parameters(M,emit=lambda x:None):
 if len(M)<2 or any(len(row)!=len(M)for row in M):return dict(classification='UNRESOLVED_matrix_shape')
 if any(x is None for row in M for x in row):return dict(classification='UNRESOLVED_missing_moments_NO_mean_product_fill')
 mass=M[0][0]
 if mass<=0:return dict(classification='UNRESOLVED_zero_or_signed_mass_NO_division',mass=c.rec(mass))
 if mass!=1:return dict(classification='UNRESOLVED_nonunit_mass_NO_normalization',mass=c.rec(mass))
 ps=[];issues=[]
 for i in range(1,len(M)):
  emit(dict(kind='BEFORE_NEW_Beta_parameter',index=i,mass=c.rec(mass),first=c.rec(M[0][i]),second=c.rec(M[i][i])))
  mu=M[0][i];v=M[i][i]-mu*mu
  if not 0<mu<1 or not 0<v<mu*(1-mu):
   issue=dict(index=i,mu=c.rec(mu),variance=c.rec(v),classification='UNRESOLVED_boundary_or_invalid_Beta_variance');emit(dict(kind='AFTER_NEW_Beta_parameter_RETURN',RETURN=issue));issues.append(issue);continue
  kappa=mu*(1-mu)/v-1;p=dict(mu=mu,variance=v,kappa=kappa,a=mu*kappa,b=(1-mu)*kappa);emit(dict(kind='AFTER_NEW_Beta_parameter_RETURN',index=i,RETURN={k:c.rec(z)for k,z in p.items()}));ps.append(p)
 if issues:return dict(classification='UNRESOLVED_invalid_parameter_domains',issues=issues)
 return dict(classification='VALID_positive_Beta_parameters_ONLY',parameters=ps)

def density(M,ps,emit=lambda x:None):
 n=len(ps);zero=(0,)*n;poly={zero:F(1)};corrections=[]
 def add(p,z):poly[p]=poly.get(p,F(0))+z
 for i,j in combinations(range(n),2):
  emit(dict(kind='BEFORE_NEW_fixed_cross_correction',indices=[i,j],saved_cross=c.rec(M[i+1][j+1]),mu_i=c.rec(ps[i]['mu']),mu_j=c.rec(ps[j]['mu']),variance_i=c.rec(ps[i]['variance']),variance_j=c.rec(ps[j]['variance'])))
  cov=M[i+1][j+1]-ps[i]['mu']*ps[j]['mu'];lam=cov/(ps[i]['variance']*ps[j]['variance']);r=dict(indices=[i,j],covariance=c.rec(cov),coefficient=c.rec(lam));emit(dict(kind='AFTER_NEW_fixed_cross_correction_RETURN',RETURN=r));corrections.append(r)
  ei=tuple(int(k==i)for k in range(n));ej=tuple(int(k==j)for k in range(n));eij=tuple(int(k in[i,j])for k in range(n));add(zero,lam*ps[i]['mu']*ps[j]['mu']);add(ei,-lam*ps[j]['mu']);add(ej,-lam*ps[i]['mu']);add(eij,lam)
 return {k:v for k,v in poly.items()if v},corrections

def evaluate(poly,t):
 value=F(0)
 for powers,co in poly.items():
  z=co
  for x,k in zip(t,powers,strict=True):z*=x**k
  value+=z
 return value

def base_expectation(ps,powers):
 z=F(1)
 for p,r in zip(ps,powers,strict=True):z*=beta_moment(p['a'],p['b'],r)
 return z

def weighted_moment(ps,poly,powers):
 return sum(v*base_expectation(ps,tuple(a+b for a,b in zip(k,powers,strict=True)))for k,v in poly.items())
