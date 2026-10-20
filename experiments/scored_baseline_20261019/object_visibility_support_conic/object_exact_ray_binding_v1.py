"""NEW exact input-model ray maps and explicit saved-map discrepancy bounds.

Binds SAVED binary geometry/camera inputs, not unsaved floating intermediates,
upstream source truth, first-surface visibility, or candidate acceptance gates.
"""
from fractions import Fraction as Q
import itertools

def solve3(A,b):
 m=[list(r)+[b[i]]for i,r in enumerate(A)]
 for i in range(3):
  j=next((j for j in range(i,3)if m[j][i]),None)
  if j is None:return None
  m[i],m[j]=m[j],m[i];s=m[i][i];m[i]=[x/s for x in m[i]]
  for j in range(3):
   if j==i:continue
   a=m[j][i];m[j]=[x-a*y for x,y in zip(m[j],m[i],strict=True)]
 return [r[-1]for r in m]

def exact_input_ray_map(vertices,pose,pixel,K):
 v=[[sum((Q(float(pose[k,j]))*Q(float(v[j]))for j in range(3)),Q(0))+Q(float(pose[k,3]))for k in range(3)]for v in vertices];r=[(Q(int(pixel[0]))-Q(float(K[0,2])))/Q(float(K[0,0])),(Q(int(pixel[1]))-Q(float(K[1,2])))/Q(float(K[1,1])),Q(1)];B=[[v[1][k]-v[0][k],v[2][k]-v[0][k],-r[k]]for k in range(3)];c=solve3(B,[-x for x in v[0]])
 if c is None:return dict(status='UNRESOLVED_exact_input_singular')
 columns=[solve3(B,[-Q(int(i==j))for i in range(3)])for j in range(3)];J=[[columns[j][i]for j in range(3)]for i in range(3)];assert all(sum(B[i][j]*c[j]for j in range(3))==-v[0][i]for i in range(3));assert all(sum(B[i][k]*J[k][j]for k in range(3))==-Q(int(i==j))for i in range(3)for j in range(3));return dict(status='exact_SAVED_binary_input_model_ONLY',constants=c,derivatives=J,B=B,triangle_C_exact=v)

def rat(x):
 if isinstance(x,Q):return dict(numerator=x.numerator,denominator=x.denominator)
 if isinstance(x,list):return [rat(v)for v in x]
 if isinstance(x,dict):return {str(k):rat(v)for k,v in x.items()}
 return x

def discrepancy(saved_constants,saved_J,exact,RC,halfwidth):
 h=Q(float(halfwidth));bounds=[]
 for k in range(3):
  dc=Q(float(saved_constants[k]))-exact['constants'][k];dj=[Q(float(saved_J[k][j]))-exact['derivatives'][k][j]for j in range(3)];dw=[sum(dj[j]*Q(float(RC[j,l]))for j in range(3))for l in range(3)];bounds.append(abs(dc)+h*sum(abs(v)for v in dw))
 return bounds
