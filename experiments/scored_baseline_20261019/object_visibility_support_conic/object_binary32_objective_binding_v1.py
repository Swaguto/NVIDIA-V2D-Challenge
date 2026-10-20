"""NEW independent integer/rational IEEE binary32 rounding for saved objective.

Explains scalar-promotion precision; never changes actual coefficients/returns.
"""
from fractions import Fraction as Q

def binary32_round(x):
 x=Q(x)
 if x==0:return Q(0)
 sign=-1 if x<0 else 1;x=abs(x);e=x.numerator.bit_length()-x.denominator.bit_length()
 while x<Q(2)**e:e-=1
 while x>=Q(2)**(e+1):e+=1
 if e>127:raise ValueError('UNRESOLVED binary32 overflow')
 step=Q(2)**max(e-23,-149);scaled=x/step;lo=scaled.numerator//scaled.denominator;rem=scaled-Q(lo)
 if rem>Q(1,2)or(rem==Q(1,2)and lo%2):lo+=1
 return sign*lo*step

def promoted_depth_residual(z0,depth,scale):
 z=binary32_round(Q(float(z0)));d=Q(float(depth));difference=binary32_round(z-d);divisor=binary32_round(Q(float(scale)));r=binary32_round(difference/divisor)
 return r
