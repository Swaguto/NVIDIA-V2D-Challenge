"""Free-rank/nearest-union source logic and LINE_8 triangle branch interpreter.

No episode evaluation is performed by importing this module. Exact rational
distance logic is separate from the original floating KD/norm operator. That
operator remains an explicit verification obligation at uncertain boundaries.

The clipping/line/scan conversion interpreter is adapted from OpenCV 4.11.0
drawing.cpp / imgproc.hpp, with this original license retained:

Copyright (C) 2000, Intel Corporation, all rights reserved.
Third party copyrights are property of their respective owners.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
* Redistribution's of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistribution's in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* The name of Intel Corporation may not be used to endorse or promote products
  derived from this software without specific prior written permission.
This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the Intel Corporation or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
"""
from fractions import Fraction as F
import math
import numpy as np


def free_rank_formula(n, baseline_median):
    if n < 1 or baseline_median is None:
        return {'status': 'UNRESOLVED_missing_or_empty'}
    m = F(baseline_median)
    if m < 0:
        raise ValueError('nonnegative distances require nonnegative median')
    lo, hi = (n - 1) // 2, n // 2
    return dict(status='FREE_threshold_cardinality_formula', n=n,
                ranks_zero_based=[lo, hi], counts=[lo + 1, hi + 1],
                threshold_symbols=['a', 'b'] if lo != hi else ['a'],
                constraints=['0<=a<=b', 'a+b<=2*m0'] if lo != hi else ['0<=a<=m0'],
                subsets_FREE=True, nearest_vertices_FREE=True,
                rational_ideal_distance_equivalence=True,
                original_binary64_median_requires_actual_rounded_average=True)


def rank_event(distances, baseline_median, a, b=None, floating=False):
    """Evaluate a DECLARED free-threshold witness, never choose a candidate.

    floating=True binds median = fl(fl(a+b)/2), including subnormal/overflow
    behavior. Thresholds must be finite nonnegative doubles in that scope.
    """
    if not distances or any(d is None for d in distances):
        return dict(status='UNRESOLVED_missing_distance', accepted=False)
    n = len(distances); lo, hi = (n - 1) // 2, n // 2
    a = float(a) if floating else F(a)
    b = a if lo == hi else (float(b) if floating else F(b))
    ds = [float(v) if floating else F(v) for v in distances]
    if floating and not all(math.isfinite(v) and v >= 0 for v in [a, b, *ds]):
        return dict(status='UNRESOLVED_nonfinite_or_negative', accepted=False)
    if min(ds) < 0 or a < 0 or b < a:
        return dict(status='invalid_ordered_nonnegative_thresholds', accepted=False)
    # Python double arithmetic is deliberately separate from rational arithmetic.
    average = (a + b) / 2
    counts = [sum(d <= a for d in ds), sum(d <= b for d in ds)]
    return dict(status='declared_threshold_witness', counts=counts,
                required=[lo+1, hi+1], average=average,
                accepted=bool(counts[0] >= lo+1 and counts[1] >= hi+1
                              and average <= baseline_median))


def cardinality_truth(values, required):
    if required < 0 or required > len(values):
        raise ValueError('invalid cardinality')
    if any(v is not True and v is not False and v is not None for v in values):
        raise TypeError('three-valued events only')
    certain = sum(v is True for v in values)
    possible = sum(v is not False for v in values)
    return dict(certain=certain, possible=possible, required=required,
                status='PASS' if certain >= required else
                ('FAIL' if possible < required else 'UNRESOLVED'))


def nearest_union_event(local_point, vertices, radius):
    """Rational ideal-model union; does not call or certify a KD operator."""
    if local_point is None or vertices is None or radius is None or not vertices:
        return None
    r = F(radius)
    if r < 0:
        return False
    return any(sum((F(x)-F(y))**2 for x, y in zip(local_point, v, strict=True)) <= r*r
               for v in vertices)


def quantized_interval(lo, hi, clip=20000):
    """Closed rational UV enclosure; an unresolved integer boundary stays free."""
    if lo is None or hi is None:
        return dict(status='UNRESOLVED_missing_projection')
    lo, hi = F(lo), F(hi)
    if lo > hi:
        raise ValueError('ordered outward interval required')
    qlo = int(min(F(clip), max(F(-clip), lo)))
    qhi = int(min(F(clip), max(F(-clip), hi)))
    return dict(status='CERTAIN_integer' if qlo == qhi else 'UNRESOLVED_integer_boundary',
                minimum_integer=qlo, maximum_integer=qhi)


def positive_projection(point, fx, fy, cx, cy, near=F(1,100)):
    if point is None:
        return dict(status='UNRESOLVED_missing_camera_point')
    x, y, z = map(F, point)
    if z <= near:
        return dict(status='UNRESOLVED_original_strict_near_plane')
    return dict(status='ideal_positive_depth_projection',
                uv=[F(fx)*x/z+F(cx), F(fy)*y/z+F(cy)],
                binary_operator_error_NOT_assumed_zero=True)


def silhouette_count_envelope(certain, possible, mask, domain, baseline_I, baseline_U):
    C, P, M, D = [np.asarray(v, bool) for v in (certain, possible, mask, domain)]
    if not (C.shape == P.shape == M.shape == D.shape) or np.any(C & ~P):
        raise ValueError('certain coverage must be contained in possible coverage')
    m = int(np.count_nonzero(M & D))
    il = int(np.count_nonzero(C & M & D)); iu = int(np.count_nonzero(P & M & D))
    ol = int(np.count_nonzero(C & ~M & D)); ou = int(np.count_nonzero(P & ~M & D))
    ul, uu = m+ol, m+ou
    if baseline_U <= 0 or ul <= 0:
        return dict(status='UNRESOLVED_empty_union', I=[il, iu], U=[ul, uu])
    lower = il*baseline_U-baseline_I*uu
    upper = iu*baseline_U-baseline_I*ul
    return dict(status='PASS' if lower >= 0 else ('FAIL' if upper < 0 else 'UNRESOLVED'),
                I=[il, iu], U=[ul, uu], determinant=[lower, upper],
                occlusion_is_full_triangle_OR_NOT_z_buffer=True)


def cdiv(a, b):
    """C++ signed integer division, truncating toward zero."""
    if not b:
        raise ZeroDivisionError
    return (1 if (a >= 0) == (b >= 0) else -1) * (abs(a)//abs(b))


def clipped_line(p1, p2, width, height):
    # The reference uses double, then int64 cast, for clipping intersections.
    # Python's evaluated double sequence is retained; binary build equivalence
    # is independently tested only in declared synthetic cases.
    if width <= 0 or height <= 0:
        return None
    x1, y1 = map(int, p1); x2, y2 = map(int, p2)
    right, bottom = width-1, height-1
    code = lambda x,y: int(x<0)+2*int(x>right)+4*int(y<0)+8*int(y>bottom)
    c1, c2 = code(x1,y1), code(x2,y2)
    if not (c1 & c2) and (c1 | c2):
        if c1 & 12:
            a = 0 if c1 < 8 else bottom
            x1 += int(float(a-y1)*(x2-x1)/(y2-y1)); y1 = a
            c1 = int(x1<0)+2*int(x1>right)
        if c2 & 12:
            a = 0 if c2 < 8 else bottom
            x2 += int(float(a-y2)*(x2-x1)/(y2-y1)); y2 = a
            c2 = int(x2<0)+2*int(x2>right)
        if not (c1 & c2) and (c1 | c2):
            if c1:
                a = 0 if c1 == 1 else right
                y1 += int(float(a-x1)*(y2-y1)/(x2-x1)); x1 = a; c1 = 0
            if c2:
                a = 0 if c2 == 1 else right
                y2 += int(float(a-x2)*(y2-y1)/(x2-x1)); x2 = a; c2 = 0
    return ((x1,y1),(x2,y2)) if not (c1 | c2) else None


def line8_pixels(p1, p2, width, height):
    clipped = clipped_line(p1,p2,width,height)
    if clipped is None:
        return []
    (x,y),(x2,y2) = clipped
    dx,dy = x2-x,y2-y; sx,sy = 1,1
    if dx < 0:
        dx,dy = -dx,-dy; x,y = x2,y2
    if dy < 0:
        dy,sy = -dy,-1
    vert = dy > dx
    if vert:
        dx,dy = dy,dx; sx,sy = sy,sx
    err = dx-2*dy
    minor = (0,sy); major = (sx,0)
    if vert:
        minor = minor[::-1]; major = major[::-1]
    out=[]
    for _ in range(dx+1):
        out.append((x,y)); take = err < 0
        err += -2*dy+(2*dx if take else 0)
        x += major[0]+(minor[0] if take else 0)
        y += major[1]+(minor[1] if take else 0)
    return out


def triangle_line8_integer(vertices, shape):
    """Reference branch interpreter, bounded to ORIGINAL clipped int32 inputs."""
    v=[tuple(map(int,p)) for p in vertices]
    if len(v) != 3 or any(abs(q)>20000 for p in v for q in p):
        raise ValueError('three original clip-then-truncate vertices required')
    h,w=map(int,shape)
    if h <= 0 or w <= 0:
        raise ValueError('positive raster dimensions required')
    out=np.zeros((h,w),bool)
    for p,q in zip(v[-1:]+v[:-1],v,strict=True):
        for x,y in line8_pixels(p,q,w,h):
            out[y,x]=True
    imin=min(range(3),key=lambda i:v[i][1]); ymin=v[imin][1]
    ymax=max(y for x,y in v)
    if max(x for x,y in v)<0 or ymax<0 or min(x for x,y in v)>=w or ymin>=h:
        return out
    ymax=min(ymax,h-1); y=ymin; edges=3; ONE=1<<16
    edge=[dict(idx=imin,di=1,x=-ONE,dx=0,ye=y),
          dict(idx=imin,di=2,x=-ONE,dx=0,ye=y)]
    while True:
        for e in edge:
            if y >= e['ye']:
                idx0=e['idx']; idx=(idx0+e['di'])%3
                while True:
                    prev=edges; edges-=1
                    if prev <= 0:
                        break
                    ty=v[idx][1]
                    if ty > y:
                        xs=v[idx0][0]*ONE; xe=v[idx][0]*ONE
                        e.update(ye=ty,dx=cdiv((xe-xs)*2+ty-y,2*(ty-y)),x=xs,idx=idx)
                        break
                    idx0=idx; idx=(idx+e['di'])%3
        if edges < 0:
            break
        if y >= 0:
            left,right=sorted([e['x'] for e in edge])
            xx1=(left+(ONE>>1))>>16; xx2=(right+(ONE>>1))>>16
            if xx2 >= 0 and xx1 < w:
                out[y,max(0,xx1):min(w-1,xx2)+1]=True
        for e in edge:
            e['x']+=e['dx']
        y+=1
        if y>ymax:
            break
    return out
