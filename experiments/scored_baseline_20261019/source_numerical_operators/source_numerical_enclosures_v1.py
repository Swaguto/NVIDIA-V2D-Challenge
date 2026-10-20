"""NEW conditional operator enclosures, not source/trajectory acceptance gates.

All roundoff statements require round-to-nearest binary64, finite intermediate
operations and the stated operator graph. KD search, compiler/build behavior
and original source accuracy are NOT proved by a leaf norm enclosure.
"""
from fractions import Fraction as F
import math
import numpy as np
from training_proxy_roundoff_v1 import U,ETA,G5,T5  # old proof constants BY HASH

MAX=F(float.fromhex('0x1.fffffffffffffp+1023'))

def interval(lo,hi=None):
    lo=F(lo);hi=lo if hi is None else F(hi)
    if lo>hi:raise ValueError('ordered exact enclosure required')
    return (lo,hi)

def rn(a):
    lo,hi=a;m=max(abs(lo),abs(hi))
    if m+U*m+ETA>MAX:raise OverflowError('UNRESOLVED binary64 overflow')
    e=U*m+ETA
    return lo-e,hi+e

def add(a,b):return rn((a[0]+b[0],a[1]+b[1]))

def mul(a,b):
    values=[x*y for x in a for y in b]
    return rn((min(values),max(values)))

def div_positive(a,b):
    if b[0]<=0:raise ZeroDivisionError('UNRESOLVED nonpositive denominator')
    values=[x/y for x in a for y in b]
    return rn((min(values),max(values)))

def square(a):
    lo,hi=a;values=[lo*lo,hi*hi]
    bounds=(F(0)if lo<=0<=hi else min(values),max(values))
    result=rn(bounds)
    # Intersect an ENCLOSURE with mathematical nonnegativity. No returned
    # distance, weight, radius or candidate value is altered.
    return max(F(0),result[0]),result[1]

def sqrt_grid_bounds(a):
    """One fixed128bit outward certificate grid, never a candidate tolerance."""
    if a<0:raise ValueError('nonnegative radicand')
    a=F(a);scale=1<<128
    n=math.isqrt((a.numerator<<(2*128))//a.denominator)
    lo=F(n,scale);hi=lo if lo*lo==a else F(n+1,scale)
    return lo,hi

def norm_leaf_enclosure(x,v):
    if len(x)!=3 or len(v)!=3 or any(not math.isfinite(float(a))for a in list(x)+list(v)):
        return dict(status='UNRESOLVED_nonfinite_or_wrong_dimension')
    steps=[];acc=interval(0)
    try:
        for k in range(3):
            d=rn(interval(F(float(x[k]))-F(float(v[k]))));s=square(d);acc=add(acc,s)
            steps.append(dict(axis=k,difference=d,square=s,squared_sum=acc))
        lo=sqrt_grid_bounds(max(F(0),acc[0]))[0];hi=sqrt_grid_bounds(acc[1])[1]
        result=rn((lo,hi));result=(max(F(0),result[0]),result[1])
    except OverflowError:
        return dict(status='UNRESOLVED_overflow',steps=steps)
    exact=sum((F(float(a))-F(float(b)))**2 for a,b in zip(x,v,strict=True))
    return dict(status='CONDITIONAL_selected_vertex_norm_enclosure',distance=result,
                exact_squared_distance=exact,steps=steps,
                sqrt_grid_bits=128,global_nearest_search_certified=False,
                installed_sqrt_compiler_and_KD_search_NOT_proved=True)

def projected_operator_enclosure(camera_intervals,fx,fy,cx,cy,near=F(1,100)):
    if camera_intervals is None or len(camera_intervals)!=3:
        return dict(status='UNRESOLVED_missing_camera_intervals')
    x,y,z=[interval(*v)for v in camera_intervals]
    if z[0]<=near:
        return dict(status='UNRESOLVED_original_strict_near_plane',depth_interval=z)
    steps=[]
    try:
        for coordinate,focal,center in [(x,fx,cx),(y,fy,cy)]:
            product=mul(interval(F(float(focal))),coordinate)
            ratio=div_positive(product,z)
            uv=add(ratio,interval(F(float(center))))
            steps.append(dict(focal_product=product,division=ratio,uv=uv))
    except OverflowError:
        return dict(status='UNRESOLVED_projection_overflow',steps=steps)
    return dict(status='CONDITIONAL_original_multiply_divide_add_projection',
                uv_intervals=[s['uv']for s in steps],steps=steps,
                finite_RN_operator_graph_required=True,candidate_tolerances_UNCHANGED=True)

def common_dyadic(a):
    """Lossless common-denominator integer encoding of finite saved doubles."""
    a=np.asarray(a,float)
    if not np.isfinite(a).all():raise ValueError('finite saved inputs required')
    values=[float(x).as_integer_ratio()for x in a.ravel()]
    exponent=max((den.bit_length()-1 for num,den in values),default=0)
    integers=np.array([num<<(exponent-(den.bit_length()-1))for num,den in values],dtype=object).reshape(a.shape)
    return integers,exponent

def cached_camera_alias(Vint,EV,R,t,C0):
    """EXACT C0 - (V R^T+t) in a shared dyadic denominator, no source replay."""
    Rint,ER=common_dyadic(R);tint,ET=common_dyadic(t);Cint,EC=common_dyadic(C0)
    E=max(EV+ER,ET,EC)
    ideal=(Vint@Rint.T)*(1<<(E-EV-ER))+tint*(1<<(E-ET))
    defect=Cint*(1<<(E-EC))-ideal
    maxima=[max(abs(int(n))for n in defect[:,k])for k in range(3)]
    witnesses=[next(i for i,n in enumerate(defect[:,k])if abs(int(n))==maxima[k])for k in range(3)]
    return dict(defect_numerators=defect,denominator_exponent=E,
                maximum_absolute_defects=[F(n,1<<E)for n in maxima],witness_indices=witnesses,
                original_arrays_UNALTERED=True,actual_numeric_pose_NOT_assigned=True)

def cache_error_affine_coefficients(C0_absolute,alias_absolute,Rcw):
    """Error bound coefficients for ANY later declared finite box |delta_j|<=H_j.

    H remains UNASSIGNED. A declared algorithmic box would be a construction
    restriction, never an original object or robot acceptance box.
    """
    out=[]
    for i in range(3):
        constant=F(alias_absolute[i])+T5+U*(F(C0_absolute[i])+T5)+ETA
        slopes=[(G5+U*(1+G5))*abs(F(float(v)))for v in Rcw[i]]
        out.append([constant,*slopes])
    return out

def typed_query_return(distance,index,vertex_count,query_k,scope):
    if scope not in ['EVEN_training_k2','ORIGINAL_fullpixel_k1']:
        return dict(status='UNRESOLVED_wrong_operator_scope')
    required=2 if scope=='EVEN_training_k2'else 1
    if query_k!=required:return dict(status='UNRESOLVED_query_k_scope_mismatch')
    if not math.isfinite(float(distance))or index<0 or index>=vertex_count:
        return dict(status='UNRESOLVED_missing_or_nonfinite_neighbor',
                    raw_distance=distance,raw_index=index,scope=scope)
    if distance<0:return dict(status='UNRESOLVED_negative_distance',raw_distance=distance)
    return dict(status='actual_numeric_return_ONLY',raw_distance=distance,raw_index=index,
                scope=scope,query_k=query_k,global_ideal_or_source_truth_NOT_certified=True)
