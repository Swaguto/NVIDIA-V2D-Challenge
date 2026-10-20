"""Research-only sparse half-angle time constraints; no pose solver or native calls.

For Q=sum(k_j*q_j), q_j=2 atan(u_j), P=prod(1+sign(k_j)*i*u_j)^abs(k_j).
Its phase is Q/2 modulo 2pi and its squared modulus is >=1.  If the caller's
angle box certifies |Q-L|,|Q-U|<6<2pi, L<=Q<=U is equivalent to
Im(exp(-i L/2)P)>=0 and -Im(exp(-i U/2)P)>=0.  This avoids division by
1+u_t*u_previous, whose sign is not generally positive.

All rational operations bind saved binary inputs, not upstream trig/source truth.
Sin/cos coefficients are rational Taylor enclosures with saved remainder bounds.
"""
from fractions import Fraction as F
import math


def rec(x):
    x=F(x)
    return dict(numerator_hex=hex(x.numerator),denominator_hex=hex(x.denominator),approx=float(x))


def unrec(x):
    return F(int(x['numerator_hex'],16),int(x['denominator_hex'],16))


def sincos(x,terms=28):
    x=F(x)
    if abs(x)>4 or not isinstance(terms,int) or terms<8:
        raise ValueError('Taylor contract requires |argument|<=4 and at least8terms')
    # Absolute Taylor remainder <= |x|^(n+1)/(n+1)! on the real line.
    sine=sum(((-1)**k*x**(2*k+1)/math.factorial(2*k+1) for k in range(terms)),F(0))
    cosine=sum(((-1)**k*x**(2*k)/math.factorial(2*k) for k in range(terms)),F(0))
    se=abs(x)**(2*terms+1)/math.factorial(2*terms+1)
    ce=abs(x)**(2*terms)/math.factorial(2*terms)
    return dict(sin=(sine-se,sine+se),cos=(cosine-ce,cosine+ce),argument=rec(x),
                terms=terms,sin_remainder=rec(se),cos_remainder=rec(ce))


def phasor(weights):
    if not weights or any(type(k)is not int or abs(k)>2 for k in weights):
        raise ValueError('finite integer weights in[-2,2] required')
    n=len(weights);zero=(0,)*n;p={zero:(1,0)}
    for j,k in enumerate(weights):
        for unused in range(abs(k)):
            sign=1 if k>0 else -1;out={}
            for exponent,(a,b) in p.items():
                old=out.get(exponent,(0,0));out[exponent]=(old[0]+a,old[1]+b)
                e=list(exponent);e[j]+=1;e=tuple(e);old=out.get(e,(0,0))
                out[e]=(old[0]-sign*b,old[1]+sign*a)
            p={e:z for e,z in out.items() if z!=(0,0)}
    return p


def linear_range(weights,box):
    if len(weights)!=len(box):raise ValueError('shape mismatch')
    lo=hi=F(0)
    for k,(a,b) in zip(weights,box,strict=True):
        a,b=F(a),F(b)
        if not a<=b:raise ValueError('reversed box')
        lo+=k*(a if k>=0 else b);hi+=k*(b if k>=0 else a)
    return lo,hi


def interval_multiply(a,b):
    v=[a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]]
    return min(v),max(v)


def scale(a,k):
    return (a[0]*k,a[1]*k) if k>=0 else (a[1]*k,a[0]*k)


def subtract(a,b):return a[0]-b[1],a[1]-b[0]


def build(weights,angle_box,half_angle_box,lower,upper):
    lower,upper=F(lower),F(upper)
    if lower>upper:raise ValueError('empty target interval')
    if len(weights)!=len(half_angle_box):raise ValueError('shape mismatch')
    qrange=linear_range(weights,angle_box)
    chart=all(-3<F(a)<=F(b)<3 for a,b in angle_box)
    branch=all(abs(v-end)<6 for v in qrange for end in[lower,upper])
    status='certified_finite_angle_branch' if chart and branch else 'UNRESOLVED_angle_branch_or_half_angle_pole'
    result=dict(weights=weights,angle_combination_range=[rec(v)for v in qrange],
                requested_interval=[rec(lower),rec(upper)],angle_branch=status,
                squared_phasor_modulus_denominator_lower_bound=1,
                positive_denominator_identity='prod((1+u_j^2)^abs(k_j)) >=1',
                tangent_difference_denominator_assumed_positive=False,boundaries=[])
    if not chart or not branch:return result
    p=phasor(weights)
    for name,threshold,direction in [('lower',lower,1),('upper',upper,-1)]:
        trig=sincos(threshold/2);rows=[];uniform=F(0)
        for e,(real,imag) in sorted(p.items()):
            coefficient=scale(subtract(scale(trig['cos'],imag),scale(trig['sin'],real)),direction)
            midpoint=sum(coefficient)/2;error=(coefficient[1]-coefficient[0])/2
            monomial_abs=math.prod(max(abs(F(a)),abs(F(b)))**power for power,(a,b)in zip(e,half_angle_box,strict=True))
            uniform+=error*monomial_abs
            rows.append(dict(exponent=list(e),coefficient_lower=rec(coefficient[0]),coefficient_upper=rec(coefficient[1]),midpoint=rec(midpoint)))
        result['boundaries'].append(dict(name=name,sense='>=0',threshold=rec(threshold),
             trig_enclosure={k:([rec(v)for v in value]if k in['sin','cos']else value)for k,value in trig.items()},
             coefficients=rows,uniform_coefficient_error=rec(uniform),
             necessary_midpoint_polynomial_lower_threshold=rec(-uniform)))
    return result


def evaluate_boundary(boundary,point):
    lo=hi=F(0)
    for row in boundary['coefficients']:
        monomial=math.prod(F(v)**k for v,k in zip(point,row['exponent'],strict=True))
        a,b=scale((unrec(row['coefficient_lower']),unrec(row['coefficient_upper'])),monomial)
        lo+=a;hi+=b
    return lo,hi


def classify(record,point):
    if record['angle_branch']!='certified_finite_angle_branch':return 'UNRESOLVED_angle_branch'
    ranges=[evaluate_boundary(v,point)for v in record['boundaries']]
    if any(b<0 for a,b in ranges):return 'rejected'
    if all(a>=0 for a,b in ranges):return 'inside'
    return 'UNRESOLVED_boundary'
