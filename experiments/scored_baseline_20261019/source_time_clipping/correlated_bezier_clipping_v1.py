"""Research-only necessary filled-sublevel contractor on rational Bernstein grids.

If p<=0 and a serialized grid lower-bounds p, (x,p_lower) is in the convex
hull of (i/n,b_i). Projecting its portion below zero therefore encloses x.
This keeps filled interiors and disconnected components; it does not select
roots, poses, feasible centers or a path. Rounding is outward for lower grids.
"""
from fractions import Fraction as F
from itertools import product
from math import comb,prod

def rec(x):
    x=F(x);return dict(numerator_hex=hex(x.numerator),denominator_hex=hex(x.denominator),approx=float(x))
def dec(v):return F(int(v['numerator_hex'],16),int(v['denominator_hex'],16))
def encode(grid):return [dict(index=list(k),value=rec(v))for k,v in sorted(grid.items())]
def decode(rows):return {tuple(r['index']):dec(r['value'])for r in rows}
def indices(degree):return product(*(range(n+1)for n in degree))
def evaluate(grid,degree,point):
    return sum(v*prod(F(comb(n,i))*t**i*(1-t)**(n-i)for n,i,t in zip(degree,k,point,strict=True))for k,v in grid.items())
def power_to_bernstein(power,bounds,degree):
    local={}
    for exponent,coefficient in power.items():
        for k in product(*(range(v+1)for v in exponent)):
            local[k]=local.get(k,F(0))+coefficient*prod(F(comb(n,i))*a**(n-i)*(b-a)**i for n,i,(a,b)in zip(exponent,k,bounds,strict=True))
    return {i:sum(v*prod(F(comb(j,k),comb(n,k))for n,j,k in zip(degree,i,e,strict=True))for e,v in local.items()if all(k<=j for k,j in zip(e,i,strict=True)))for i in indices(degree)}
def split_curve(curve,t):
    work=curve.copy();left=[work[0]];right=[work[-1]]
    while len(work)>1:
        work=[(1-t)*a+t*b for a,b in zip(work[:-1],work[1:],strict=True)];left.append(work[0]);right.append(work[-1])
    return left,list(reversed(right))
def restrict(grid,degree,ranges):
    result=grid.copy()
    for axis,(lo,hi)in enumerate(ranges):
        if not 0<=lo<=hi<=1:raise ValueError('restriction outside unit interval')
        others=[k for k in range(len(degree))if k!=axis];out={}
        for rest in product(*(range(degree[k]+1)for k in others)):
            def key(i):
                r=[0]*len(degree);r[axis]=i
                for a,b in zip(others,rest,strict=True):r[a]=b
                return tuple(r)
            curve=[result[key(i)]for i in range(degree[axis]+1)]
            if lo==1:section=[curve[-1]]*len(curve)
            else:
                right=split_curve(curve,lo)[1];section=split_curve(right,(hi-lo)/(1-lo))[0]
            for i,v in enumerate(section):out[key(i)]=v
        result=out
    return result
def lower_round(grid,bits=64):
    scale=1<<bits;return {k:F((v*scale).__floor__(),scale)for k,v in grid.items()}
def clip_axis(grid,degree,axis):
    if min(grid.values())>0:return dict(empty=True,minimum_lower_coefficient=rec(min(grid.values())))
    n=degree[axis]
    if n==0:return dict(empty=False,lower=rec(0),upper=rec(1),points=[],crossings=[])
    points=[(F(i,n),min(v for k,v in grid.items()if k[axis]==i))for i in range(n+1)]
    candidates=[x for x,z in points if z<=0];crossings=[]
    for i,(x,z)in enumerate(points):
        for k,(y,w)in enumerate(points[i+1:],i+1):
            if z*w<0:
                t=(x*w-y*z)/(w-z);candidates.append(t);crossings.append(dict(pair=[i,k],zero_coordinate=rec(t)))
    assert candidates
    return dict(empty=False,lower=rec(min(candidates)),upper=rec(max(candidates)),points=[dict(coordinate=rec(x),coefficient=rec(z))for x,z in points],crossings=crossings)
def outer_quantize(value,initial,side,bits=32):
    a,b=initial;scale=1<<bits;t=(value-a)/(b-a)*scale
    q=t.__floor__()if side=='lower'else t.__ceil__();return a+(b-a)*F(q,scale)
def lagrange_basis_bound(degree):
    nodes=[F(i,degree)for i in range(degree+1)];rows=[]
    for j,x in enumerate(nodes):
        p={0:F(1)};den=F(1)
        for k,y in enumerate(nodes):
            if k==j:continue
            out={}
            for e,c in p.items():out[e]=out.get(e,F(0))-y*c;out[e+1]=out.get(e+1,F(0))+c
            p=out;den*=x-y
        power={(e,):v/den for e,v in p.items()};b=power_to_bernstein(power,[(F(0),F(1))],(degree,));norm=max(map(abs,b.values()))
        rows.append(dict(node=rec(x),power=[rec(power.get((i,),0))for i in range(degree+1)],Bernstein=encode(b),supremum_upper_bound=rec(norm)))
    return rows,sum(dec(v['supremum_upper_bound'])for v in rows)
