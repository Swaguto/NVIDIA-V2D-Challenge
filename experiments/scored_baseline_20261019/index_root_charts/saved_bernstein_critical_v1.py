"""Exact coefficient-error/derivative bounds from SAVED Bernstein charts.

The earlier float Bernstein transform is not rerun. Its coefficient grid is
expanded independently into power form, compared with the frozen source
polynomial, and bounded exactly on the unit square. All new subdivision uses
rational De Casteljau arithmetic. Upstream trigonometric/source accuracy is
outside this frozen-coefficient arithmetic scope.
"""
from fractions import Fraction as F
import math

def rational(v):return v if isinstance(v,F) else F(float(v))

def grid(values):return [[rational(v) for v in row] for row in values]

def compose_power(p,bounds):
    p=grid(p);a,b=[rational(v) for v in bounds[0]];c,d=[rational(v) for v in bounds[1]];w=b-a;z=d-c
    n=len(p)-1;m=len(p[0])-1;out=[[F(0) for _ in range(m+1)] for _ in range(n+1)]
    for k in range(n+1):
        for l in range(m+1):
            for i in range(k+1):
                for j in range(l+1):out[i][j]+=p[k][l]*math.comb(k,i)*a**(k-i)*w**i*math.comb(l,j)*c**(l-j)*z**j
    return out

def bernstein_power(b):
    b=grid(b);n=len(b)-1;m=len(b[0])-1;out=[[F(0) for _ in range(m+1)] for _ in range(n+1)]
    for i in range(n+1):
        for j in range(m+1):
            for k in range(i,n+1):
                for l in range(j,m+1):out[k][l]+=b[i][j]*math.comb(n,i)*math.comb(n-i,k-i)*(-1)**(k-i)*math.comb(m,j)*math.comb(m-j,l-j)*(-1)**(l-j)
    return out

def audit_saved_chart(power_coefficients,bernstein_coefficients,bounds):
    source=compose_power(power_coefficients,bounds);saved=bernstein_power(bernstein_coefficients);b=grid(bernstein_coefficients)
    n=len(b)-1;m=len(b[0])-1;width=rational(bounds[0][1])-rational(bounds[0][0])
    if width<=0:raise ValueError('Nonzero proximal width required for derivative chart')
    error=[[source[i][j]-saved[i][j] for j in range(m+1)] for i in range(n+1)]
    source_error=sum(abs(v) for row in error for v in row)
    derivative_error=sum(i*abs(error[i][j]) for i in range(1,n+1) for j in range(m+1))/width
    derivative=[[n*(b[i+1][j]-b[i][j])/width for j in range(m+1)] for i in range(n)]
    return dict(source_coefficients=b,derivative_coefficients=derivative,source_error=source_error,
                derivative_error=derivative_error,exact_difference_power=error,
                source_lower=min(v for row in b for v in row)-source_error,
                source_upper=max(v for row in b for v in row)+source_error)

def split(coefficients,axis):
    b=grid(coefficients)
    if axis==1:b=[list(v) for v in zip(*b,strict=True)]
    n=len(b)-1;m=len(b[0]);left=[[F(0) for _ in range(m)] for _ in range(n+1)];right=[row.copy() for row in left];work=[row.copy() for row in b]
    left[0]=work[0].copy();right[n]=work[n].copy()
    for k in range(1,n+1):
        work=[[(a+bb)/2 for a,bb in zip(rowa,rowb,strict=True)] for rowa,rowb in zip(work[:-1],work[1:],strict=True)]
        left[k]=work[0].copy();right[n-k]=work[-1].copy()
    if axis==1:left=[list(v) for v in zip(*left,strict=True)];right=[list(v) for v in zip(*right,strict=True)]
    return left,right

def encode_grid(values):return [[dict(numerator_hex=hex(v.numerator),denominator_hex=hex(v.denominator)) for v in row] for row in grid(values)]

def decode_grid(values):return [[F(int(v['numerator_hex'],16),int(v['denominator_hex'],16)) for v in row] for row in values]

def encode_fraction(v):
    v=F(v);return dict(numerator_hex=hex(v.numerator),denominator_hex=hex(v.denominator),approx=float(v))

def evaluate_power(p,x,y):return sum(v*x**i*y**j for i,row in enumerate(p) for j,v in enumerate(row))

def evaluate_bernstein(b,x,y):
    b=grid(b);n=len(b)-1;m=len(b[0])-1
    return sum(b[i][j]*math.comb(n,i)*x**i*(1-x)**(n-i)*math.comb(m,j)*y**j*(1-y)**(m-j) for i in range(n+1) for j in range(m+1))
