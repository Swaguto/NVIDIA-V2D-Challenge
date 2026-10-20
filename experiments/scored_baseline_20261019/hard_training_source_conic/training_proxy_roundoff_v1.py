"""NEW uniform binary64 transform-to-proxy proposal bounds.

These outward bounds belong to a candidate generator. They do not bound source
accuracy or certify the KD-tree/raster/actual source acceptance operators.
"""
from fractions import Fraction as Q
import math
U=Q(1,2**53);ETA=Q(1,2**1074)
G5=5*U/(1-5*U);T5=5*ETA/(1-5*U)
def fraction(x):return Q(float(x))
def pack(q):
 q=Q(q);return dict(numerator=str(q.numerator),denominator=str(q.denominator))
def unpack(q):return Q(int(q['numerator']),int(q['denominator']))
def outward(q):
 q=Q(q)
 if q<0:raise ValueError('nonnegative outward bound required')
 x=float(q)
 if not math.isfinite(x):raise ValueError('UNRESOLVED overflow')
 if Q(x)<q:x=math.nextafter(x,math.inf)
 return x
def dot3_bound(a,e,c):
 a=list(map(Q,a));e=list(map(Q,e));c=[abs(Q(x))for x in c]
 exact=sum(x*y for x,y in zip(a,c,strict=True))
 err=sum(x*y for x,y in zip(e,c,strict=True))+G5*sum((x+z)*y for x,z,y in zip(a,e,c,strict=True))+T5
 return exact,err
def pipeline_bound(P,T,R,RC,V,H):
 """All P/V are componentwise absolute bounds; fixed coefficients are binary."""
 P=list(map(Q,P));T=list(map(Q,T));R=[[Q(x)for x in r]for r in R];RC=[[Q(x)for x in r]for r in RC];V=list(map(Q,V));H=Q(H)
 if min(P+V+[H])<0:raise ValueError('absolute component bounds and halfwidth required')
 cm=[];ce=[]
 for i in range(3):
  m,e=dot3_bound([H]*3,[Q(0)]*3,RC[i]);cm.append(m);ce.append(e)
 tm=[abs(T[i])+cm[i]for i in range(3)]
 te=[ce[i]+U*(tm[i]+ce[i])+ETA for i in range(3)]
 am=[P[i]+tm[i]for i in range(3)]
 ae=[te[i]+U*(am[i]+te[i])+ETA for i in range(3)]
 lm=[];le=[]
 for k in range(3):
  m,e=dot3_bound(am,ae,[R[j][k]for j in range(3)]);lm.append(m);le.append(e)
 off=[le[k]+U*(lm[k]+le[k]+V[k])+ETA for k in range(3)]
 return dict(camera_delta_absolute=cm,camera_delta_error=ce,updated_translation_absolute=tm,
  updated_translation_error=te,point_difference_absolute=am,point_difference_error=ae,
  local_absolute=lm,local_error=le,rounded_offset_error=off)
def proxy_allowance(P,T,R,RC,V,H,map_errors):
 future=pipeline_bound(P,T,R,RC,V,H);base=pipeline_bound(P,T,R,RC,V,Q(0))
 bias=[Q(H)*sum(abs(Q(v))for v in row)for row in map_errors]
 component=[future['local_error'][i]+base['rounded_offset_error'][i]+bias[i]for i in range(3)]
 return dict(future=future,baseline=base,affine_map_error_on_box=bias,
             component_error=component,L1_distance_allowance=sum(component))
