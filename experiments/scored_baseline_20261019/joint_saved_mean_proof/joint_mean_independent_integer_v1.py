"""Independent IEEE integer proof arithmetic, never an old operator replay."""
from fractions import Fraction as Q
import struct
from functools import lru_cache
T=Q(1,2**1074);S2=T*T;U=Q(1,2**53)
def integer1074(x):
 b=struct.unpack('>Q',struct.pack('>d',float(x)))[0];e=(b>>52)&2047;f=b&((1<<52)-1)
 if e==2047:raise ValueError('UNKNOWN_nonfinite_IEEE_payload')
 v=f if e==0 else ((1<<52)|f)<<(e-1)
 return -v if b>>63 else v
@lru_cache(maxsize=None)
def constants(n):
 m=2*n+16;return m*U/(1-m*U),m*T/(1-m*U)
def exact_side(integer_values,scale,input_errors=None):
 n=len(integer_values);g,b=constants(n);mean=Q(sum(integer_values),n)*scale;absmean=Q(sum(map(abs,integer_values)),n)*scale
 ie=sum(input_errors,Q(0))/n if input_errors is not None else Q(0)
 return mean,ie+g*(absmean+ie)+b,max(map(abs,integer_values))*scale

def independent_bound(a,b,scale,ea=None,eb=None):
 ma,da,ta=exact_side(a,scale,ea);mb,db,tb=exact_side(b,scale,eb)
 gamma=da+db+U*(abs(ma)+abs(mb)+da+db)+T
 return (ma+mb,ta+tb,gamma,ta+tb+gamma)

def exact_dot_side(rows,direction):
 vals=[];errors=[];g,b=constants(3)
 for row in rows:
  products=[x*y for x,y in zip(row,direction,strict=True)];vals.append(sum(products));errors.append(g*sum(map(abs,products))*S2+b)
 return vals,errors
