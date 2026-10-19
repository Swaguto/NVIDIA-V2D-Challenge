"""Numerical interval bounds for coupled Rodrigues roots and planar fingers.

Outward padding is defensive floating-point arithmetic, not a formal libm
directed-rounding proof. Boxes that cannot be excluded remain unresolved.
This module makes no collision, source-truth or physical-success claim.
"""
import math
import numpy as np


def exact(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Finite interval values required')
    return np.stack((x, x), axis=-1)


def pad(x):
    x = np.asarray(x, dtype=float)
    e = 32 * np.finfo(float).eps * (1 + np.max(np.abs(x), axis=-1))
    return np.stack((np.nextafter(x[..., 0] - e, -np.inf),
                     np.nextafter(x[..., 1] + e, np.inf)), axis=-1)


def add(a, b):
    return pad(np.asarray(a) + np.asarray(b))


def neg(a):
    return -np.asarray(a)[..., ::-1]


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    a, b = np.broadcast_arrays(a, b)
    v = np.stack((a[..., 0]*b[..., 0], a[..., 0]*b[..., 1],
                  a[..., 1]*b[..., 0], a[..., 1]*b[..., 1]), axis=-1)
    return pad(np.stack((v.min(axis=-1), v.max(axis=-1)), axis=-1))


def square(a):
    a = np.asarray(a)
    lo = np.where((a[..., 0] <= 0) & (a[..., 1] >= 0), 0.,
                  np.minimum(a[..., 0]**2, a[..., 1]**2))
    hi = np.maximum(a[..., 0]**2, a[..., 1]**2)
    return pad(np.stack((lo, hi), axis=-1))


def divide(a, b):
    b = np.asarray(b)
    if np.any((b[..., 0] <= 0) & (b[..., 1] >= 0)):
        raise ValueError('Denominator interval contains zero')
    return mul(a, pad(1 / b[..., ::-1]))


def sine(a):
    a = np.asarray(a, dtype=float)
    lo, hi = a[..., 0], a[..., 1]
    if np.any(hi < lo) or not np.isfinite(a).all():
        raise ValueError('Finite ordered angle interval required')
    lower = np.minimum(np.sin(lo), np.sin(hi))
    upper = np.maximum(np.sin(lo), np.sin(hi))
    lower = np.where(np.ceil((lo + math.pi/2)/(2*math.pi)) <=
                     np.floor((hi + math.pi/2)/(2*math.pi)), -1., lower)
    upper = np.where(np.ceil((lo - math.pi/2)/(2*math.pi)) <=
                     np.floor((hi - math.pi/2)/(2*math.pi)), 1., upper)
    return pad(np.stack((lower, upper), axis=-1))


def cosine(a):
    return sine(add(a, exact(math.pi/2)))


def sum_intervals(a, axis=0):
    return pad(np.asarray(a).sum(axis=axis))


def matvec(a, b):
    return sum_intervals(mul(a, np.asarray(b)[None, :, :]), axis=1)


def rotation_matrix_box(rodrigues_box):
    r = np.asarray(rodrigues_box, dtype=float)
    if r.shape != (3, 2) or not np.isfinite(r).all() or (r[:, 1] < r[:, 0]).any():
        raise ValueError('Three finite ordered Rodrigues intervals required')
    r2 = sum_intervals(square(r)); denominator = add(exact(1.), r2)
    result = np.empty((3, 3, 2))
    for i in range(3):
        result[i, i] = divide(add(sub(exact(1.), r2), mul(exact(2.), square(r[i]))), denominator)
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        p = mul(r[i], r[j])
        result[i, j] = divide(mul(exact(2.), sub(p, r[k])), denominator)
        result[j, i] = divide(mul(exact(2.), add(p, r[k])), denominator)
    return result


def root_rotvec_box(rodrigues_box, source_to_aligned_quaternion_xyzw):
    """Bound log(delta(r)*Rsource*Raligned.T), principal positive hemisphere."""
    r = np.asarray(rodrigues_box); q = np.asarray(source_to_aligned_quaternion_xyzw, dtype=float)
    if q.shape != (4,) or not np.isclose(q@q, 1., atol=1e-12):
        raise ValueError('Unit quaternion required')
    if q[3] < 0:
        q = -q
    den2 = add(exact(1.), sum_intervals(square(r)))
    den = pad(np.sqrt(np.maximum(den2, 0)))
    scalar = divide(sub(exact(q[3]), sum_intervals(mul(r, exact(q[:3])))), den)
    if scalar[0] <= 0:
        return None  # A cut locus/hemisphere crossing is unresolved, never rejected.
    cross = np.array([sub(mul(r[1], exact(q[2])), mul(r[2], exact(q[1]))),
                      sub(mul(r[2], exact(q[0])), mul(r[0], exact(q[2]))),
                      sub(mul(r[0], exact(q[1])), mul(r[1], exact(q[0])))])
    vector = divide(add(add(exact(q[:3]), mul(exact(q[3]), r)), cross), den)
    def scale(w):
        w = float(np.clip(w, 0., 1.))
        return 2. if 1-w < 1e-10 else 2*math.acos(w)/math.sqrt(1-w*w)
    factor = pad(np.array([scale(scalar[1]), scale(scalar[0])]))
    return mul(vector, factor)


def planar_point_box(joint_box, z):
    a, b = np.asarray(joint_box)
    ab = add(a, b)
    x = add(exact(.0777), add(mul(exact(.0458), cosine(a)), mul(exact(.035), cosine(ab))))
    y = add(exact(-.0016), add(mul(exact(.0458), sine(a)), mul(exact(.035), sine(ab))))
    return np.array([x, y, exact(z)])


def thumb_point_box(upstream, distal_box):
    a, b = np.asarray(upstream, dtype=float)
    bc = add(exact(b), distal_box)
    x = sub(exact(-.0025-.0458*math.sin(b)), mul(exact(.035), sine(bc)))
    y = add(exact(.0193+.0458*math.cos(b)), mul(exact(.035), cosine(bc)))
    return np.array([add(exact(.0255), mul(exact(math.cos(a)), x)),
                     y, mul(exact(-math.sin(a)), x)])


def camera_point_box(root_box, local_box, source_rotation, translation, camera_linear, camera_offset):
    source_local = matvec(exact(source_rotation), local_box)
    world = add(exact(translation), matvec(rotation_matrix_box(root_box), source_local))
    return sub(matvec(exact(camera_linear), world), exact(camera_offset))


def image_box(camera_box, intrinsics):
    if camera_box[2, 0] <= .01:
        return None  # Retain ambiguous depth; do not divide through zero.
    p = divide(camera_box[:2], camera_box[2])
    return add(mul(exact(np.diag(intrinsics)[:2]), p), exact(intrinsics[:2, 2]))


def disk_lower_bound(pixel_box, center):
    if pixel_box is None:
        return 0.
    d = np.maximum(np.maximum(pixel_box[:, 0]-center, center-pixel_box[:, 1]), 0.)
    return float(np.linalg.norm(d))
