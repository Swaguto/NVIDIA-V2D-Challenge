"""Exact rational projection of SAVED floating polynomial coefficients.

This proves arithmetic statements about that frozen coefficient model only.
It does not certify camera/source accuracy or directed rounding of the earlier
trigonometric/chart conversion. Complex roots and unresolved root intervals
remain explicit. No source polynomial is regenerated here.
"""
from fractions import Fraction as F
import math
import numpy as np

BIT_CAP=32768

def trim(p):
    p=list(p)
    while p and p[-1]==0:p.pop()
    return p

def add(a,b):
    z=[F(0)]*max(len(a),len(b))
    for i,v in enumerate(a):z[i]+=v
    for i,v in enumerate(b):z[i]+=v
    return trim(z)

def mul(a,b):
    if not a or not b:return []
    z=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):z[i+j]+=x*y
    return trim(z)

def derivative(p):return trim([i*v for i,v in enumerate(p)][1:])

def evaluate(p,x):
    z=F(0)
    for v in reversed(p):z=z*x+v
    return z

def rational(x):return x if isinstance(x,F) else F(float(x))

def primitive(p):
    p=trim(p)
    if not p:return []
    denominator=1
    for x in p:denominator=math.lcm(denominator,F(x).denominator)
    z=[int(F(x)*denominator) for x in p];content=math.gcd(*z)
    z=[v//content for v in z]
    if max(abs(v).bit_length() for v in z)>BIT_CAP:raise ArithmeticError('Exact polynomial bit cap; domain unresolved')
    return trim(z)

def encode(p):return [hex(v) for v in primitive(p)]

def decode(values):return [int(v,16) for v in values]

def fraction_record(x):
    x=F(x)
    return dict(numerator_hex=hex(x.numerator),denominator_hex=hex(x.denominator),approx=float(x))

def bivariate(coefficients):return [trim([rational(v) for v in row]) for row in coefficients]

def derivative_u(p):return [trim([i*v for v in row]) for i,row in enumerate(p)][1:]

def specialize_v(p,v):return trim([evaluate(row,F(v)) for row in p])

def specialize_u(p,u):
    result=[]
    for i,row in enumerate(p):result=add(result,[v*F(u)**i for v in row])
    return result

def sylvester(p,q):
    p=list(p);q=list(q)
    while p and not p[-1]:p.pop()
    while q and not q[-1]:q.pop()
    if not p or not q:return None
    m=len(p)-1;n=len(q)-1;size=m+n
    matrix=[[[] for _ in range(size)] for _ in range(size)]
    for row in range(n):
        for j,entry in enumerate(reversed(p)):matrix[row][row+j]=entry
    for row in range(m):
        for j,entry in enumerate(reversed(q)):matrix[n+row][row+j]=entry
    return matrix

def resultant(p,q):
    matrix=sylvester(p,q)
    if matrix is None:return dict(classification='identically_zero_input_or_derivative',coefficients=[])
    n=len(matrix);dp={0:[F(1)]}
    for mask in range(1,1<<n):
        row=mask.bit_count()-1;columns=[j for j in range(n) if mask&(1<<j)];value=[]
        for position,j in enumerate(columns):
            term=mul(matrix[row][j],dp[mask^(1<<j)])
            if (row+position)%2:term=[-v for v in term]
            value=add(value,term)
        dp[mask]=value
    exact=dp[(1<<n)-1];integer=primitive(exact)
    return dict(classification='identically_zero_resultant' if not integer else 'nonzero_exact_saved_coefficient_resultant',
                coefficients=integer,matrix_size=n,degree=len(integer)-1,
                maximum_coefficient_bits=max((abs(v).bit_length() for v in integer),default=0),
                arithmetic_scope='exact rationals of saved binary floating coefficients')

def signed_remainder(a,b):
    a=primitive(a);b=primitive(b)
    if not b:raise ZeroDivisionError('zero Sturm divisor')
    while a and len(a)>=len(b):
        difference=len(a)-len(b);leading=a[-1];factor=abs(b[-1]);sign=1 if b[-1]>0 else -1
        z=[factor*v for v in a]
        for j,v in enumerate(b):z[difference+j]-=leading*sign*v
        a=primitive(z)
    return a

def sturm(p):
    p=primitive(p)
    if not p:raise ValueError('identically zero polynomial has unresolved root domain')
    seq=[p];d=primitive(derivative(p))
    if d:seq.append(d)
    while len(seq)>1:
        r=signed_remainder(seq[-2],seq[-1])
        if not r:break
        seq.append([-v for v in r])
    return seq

def side_sign(p,x,side):
    order=0
    while p:
        value=evaluate(p,x)
        if value:
            sign=1 if value>0 else -1
            return -sign if side<0 and order%2 else sign
        p=derivative(p);order+=1
    return 0

def variations(seq,x,side):
    signs=[side_sign(p,x,side) for p in seq];signs=[v for v in signs if v]
    return sum(a!=b for a,b in zip(signs,signs[1:]))

def isolate(p,bounds,*,width=F(1,10**10),node_cap=2047):
    p=primitive(p);a,b=map(F,bounds)
    if a>b:raise ValueError('ordered exact root bounds required')
    if not p:return dict(classification='identically_zero_root_domain_unresolved',roots=[],unresolved=[dict(lower=fraction_record(a),upper=fraction_record(b))])
    seq=sturm(p);root_intervals=[];unresolved=[];endpoints=set();proposals=[];queue=[(a,b)]
    def endpoint(x):
        if evaluate(p,x)==0:endpoints.add(x)
    endpoint(a);endpoint(b)
    while queue and len(proposals)<node_cap:
        left,right=queue.pop();count=0 if left==right else variations(seq,left,1)-variations(seq,right,-1)
        if count<0:raise ArithmeticError('negative exact Sturm count')
        row=dict(lower=fraction_record(left),upper=fraction_record(right),open_real_root_count=count)
        proposals.append(row)
        if not count:continue
        if count==1 and right-left<=width:
            root_intervals.append(dict(**row,kind='isolated_open_real_root'));continue
        middle=(left+right)/2
        if middle==left or middle==right:
            unresolved.append(row);continue
        endpoint(middle);queue.extend([(middle,right),(left,middle)])
    for left,right in queue:
        unresolved.append(dict(lower=fraction_record(left),upper=fraction_record(right),open_real_root_count=variations(seq,left,1)-variations(seq,right,-1)))
    for x in endpoints:root_intervals.append(dict(lower=fraction_record(x),upper=fraction_record(x),open_real_root_count=1,kind='exact_rational_endpoint_root'))
    root_intervals.sort(key=lambda v:F(int(v['lower']['numerator_hex'],16),int(v['lower']['denominator_hex'],16)))
    total=(int(evaluate(p,a)==0) if a==b else variations(seq,a,1)-variations(seq,b,-1)+int(evaluate(p,a)==0)+int(evaluate(p,b)==0))
    return dict(classification='exact_real_roots_isolated_for_saved_coefficient_model' if not unresolved else 'unresolved_exact_root_isolation_budget',
                roots=root_intervals,unresolved=unresolved,closed_interval_distinct_real_root_count=total,
                last_Sturm_degree=len(seq[-1])-1,nonconstant_gcd_repeated_factor_present=len(seq[-1])>1,
                integer_polynomial_hex=[hex(v) for v in p],Sturm_sequence_hex=[[hex(v) for v in row] for row in seq],
                all_root_count_nodes=proposals,nodes=len(proposals),declared_node_cap=node_cap)

def numerical_complex_roots(p):
    p=primitive(p)
    if len(p)<=1:return dict(roots=[],degree=len(p)-1,identically_zero=not p)
    largest=max(abs(v) for v in p);coefficients=np.array([float(F(v,largest)) for v in p])
    try:
        roots=np.polynomial.polynomial.polyroots(coefficients);records=[]
        for root in roots:
            absolute_terms=sum(abs(float(v))*abs(root)**i for i,v in enumerate(coefficients))
            residual=float(abs(np.polynomial.polynomial.polyval(root,coefficients))/max(absolute_terms,1e-300))
            records.append(dict(real=float(root.real),imaginary=float(root.imag),normalized_residual=residual))
        return dict(roots=records,degree=len(p)-1,all_complex_companion_roots_retained=True,
                    real_root_count_authority='exact Sturm intervals, not an imaginary-part threshold')
    except Exception as exc:return dict(roots=[],degree=len(p)-1,unresolved_numerical_complex_root_return=type(exc).__name__)
