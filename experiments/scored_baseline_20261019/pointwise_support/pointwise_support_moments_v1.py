"""New support-localizer maps within existing per-coordinate degree4 tensors.

Temporal coefficient enclosures are SAVED inputs, never recomputed trig.
One declared96-bit dyadic representation has an explicit uniform upward guard:
g_true>=0 => g_dyadic+error>=0. This is a necessary model bound, not a robot
acceptance tolerance. No missing tensor moment is invented.
"""
from fractions import Fraction as F
from itertools import product
from math import comb
import conditional_moment_cones_v1 as c
import mixed_joint_moments_v1 as m

def polyrec(p):return [dict(power=list(k),value=c.rec(v))for k,v in sorted(p.items())]
def polydec(p):return {tuple(v['power']):c.dec(v['value'])for v in p}
def add(p,q,scale=F(1)):
    out=dict(p)
    for k,v in q.items():out[k]=out.get(k,F(0))+scale*v
    return {k:v for k,v in out.items()if v}
def multiply(p,q):
    out={}
    for a,v in p.items():
        for b,w in q.items():
            k=tuple(i+j for i,j in zip(a,b,strict=True));out[k]=out.get(k,F(0))+v*w
    return {k:v for k,v in out.items()if v}
def evaluate(p,t):
    out=F(0)
    for k,v in p.items():
        for x,r in zip(t,k,strict=True):v*=F(x)**r
        out+=v
    return out
def round_nearest(v,bits=96):
    scaled=F(v)*(1<<bits);q=scaled.numerator//scaled.denominator
    if scaled-q>F(1,2):q+=1
    return F(q,1<<bits)
def round_up(v,bits=96):
    scaled=F(v)*(1<<bits);return F(-(-scaled.numerator//scaled.denominator),1<<bits)
def upper_polynomial(intervals,bounds,emit,bits=96):
    if not bounds or any(len(box)!=2 or box[0]>box[1]for box in bounds):raise ValueError('finite ordered declared bounds required')
    dim=len(bounds);p={};error=F(0);degrees=[0]*dim;rows=[]
    for row in intervals:
        power=tuple(row['exponent']);lo=c.dec(row['coefficient_lower']);hi=c.dec(row['coefficient_upper'])
        if len(power)!=dim or lo>hi:raise ValueError('coefficient shape/enclosure')
        mid=round_nearest((lo+hi)/2,bits);e=max(abs(lo-mid),abs(hi-mid));factor=F(1)
        for axis,(r,box)in enumerate(zip(power,bounds,strict=True)):
            if type(r)is not int or r<0:raise ValueError('invalid exponent')
            factor*=max(map(abs,box))**r
            if lo!=0 or hi!=0:degrees[axis]=max(degrees[axis],r)
        error+=e*factor
        if mid:p[power]=mid
        rec=dict(power=list(power),original_lower=c.rec(lo),original_upper=c.rec(hi),dyadic_center=c.rec(mid),coefficient_error=c.rec(e),domain_absolute_monomial_bound=c.rec(factor),structural_nonzero=lo!=0 or hi!=0,rounded_center_zero_is_NOT_structural_zero=mid==0 and(lo!=0 or hi!=0))
        emit(dict(kind='AFTER_NEW_saved_coefficient_dyadic_RETURN',record=rec));rows.append(rec)
    guard=round_up(error,bits);zero=(0,)*dim;p[zero]=p.get(zero,F(0))+guard
    out=dict(dyadic_bits=bits,coefficient_records=rows,exact_uniform_error=c.rec(error),outward_uniform_guard=c.rec(guard),structural_degrees=degrees,upper_polynomial=polyrec(p),scope='g_true>=0 implies this upper polynomial>=0 on SAVED half-angle bounds; necessary only')
    emit(dict(kind='AFTER_NEW_uniform_support_upper_RETURN',record=out));return p,degrees,out
def affine_pullback(p,bounds,emit):
    if any(len(box)!=2 or box[0]>box[1]for box in bounds):raise ValueError('ordered affine bounds required')
    dim=len(bounds);out={}
    for power,value in p.items():
        emit(dict(kind='BEFORE_NEW_affine_saved_term',power=power,coefficient=c.rec(value)))
        term={}
        for exponent in product(*(range(r+1)for r in power)):
            z=value
            for r,k,(lo,hi)in zip(power,exponent,bounds,strict=True):z*=comb(r,k)*lo**(r-k)*(hi-lo)**k
            if z:term[tuple(exponent)]=z
        emit(dict(kind='AFTER_NEW_affine_saved_term_RETURN',power=power,term=polyrec(term)));out=add(out,term)
    return out
def reorder(p,old,new):
    if len(set(old))!=len(old)or set(old)!=set(new):raise ValueError('explicit equal variable scopes required')
    return {tuple(k[old.index(v)]for v in new):a for k,a in p.items()}
def multiplier_basis(degrees):
    if not degrees or any(type(v)is not int or v<0 for v in degrees):raise ValueError('structural degrees')
    # ONE maximal multilinear rectangular scope, not a degree sweep.
    candidates=list(product(range(2),repeat=len(degrees)))
    accepted=[a for a in candidates if all(r+2*k<=4 for r,k in zip(degrees,a))]
    rejected=[dict(exponent=list(a),required_per_axis=[r+2*k for r,k in zip(degrees,a)])for a in candidates if a not in accepted]
    return sorted(accepted,key=lambda a:(sum(a),a)),rejected
def localizer_maps(entries,gpoly,basis,weights,emit):
    M=[];records=[]
    for i,a in enumerate(basis):
        values=[]
        for j,b in enumerate(basis):
            shifted={tuple(k+x+y for k,x,y in zip(power,a,b,strict=True)):v for power,v in gpoly.items()}
            if any(any(r>4 for r in k)for k in shifted):raise ValueError('UNRESOLVED_unavailable_tensor_powers')
            co=m.polynomial_coefficients(entries,shifted);value=sum(v*F(weights[col])for col,v in co.items());rec=dict(i=i,j=j,shift=[x+y for x,y in zip(a,b)],coefficients=[dict(column=col,value=c.rec(v))for col,v in sorted(co.items())],actual_entry=c.rec(value));emit(dict(kind='AFTER_NEW_support_localizer_entry_RETURN',record=rec));values.append(value);records.append(rec)
        M.append(values)
    return M,records
def support_square(entries,gpoly,basis,vector,weights):
    scale=max(map(abs,vector),default=F(0))
    if not scale:raise ValueError('zero square vector')
    v=[F(a)/scale for a in vector];q2=m.squared_polynomial(basis,v);product_poly=multiply(gpoly,q2);co=m.polynomial_coefficients(entries,product_poly);expectation=sum(a*F(weights[col])for col,a in co.items());norm=max(map(abs,co.values()),default=F(0))
    return dict(basis=[list(a)for a in basis],certificate_vector=list(map(c.rec,v)),square_polynomial=polyrec(q2),support_times_square=polyrec(product_poly),expectation_coefficients=[dict(column=col,value=c.rec(a))for col,a in sorted(co.items())],actual_exact_expectation=c.rec(expectation),coefficient_linf=c.rec(norm),necessary_l1_weight_change_to_remove_THIS_witness=c.rec(-expectation/norm if expectation<0 and norm else F(0)),nonnegative_ONLY_on_declared_support_upper_polynomial=True,certificate_normalization_NOT_actual_weight_change=True,no_source_time_family_exclusion=True)
