"""Source-orientation domains for the remaining common-root twist.

The one-dimensional twist preserves an already mapped local point. Closed-form
orientation domains and numerical root-component envelopes do not establish
measured source orientation, finger reachability, collision or a trajectory.
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation


def _rotation(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (3, 3) or not np.isfinite(value).all()
            or not np.allclose(value.T @ value, np.eye(3), atol=1e-9, rtol=0)
            or not np.isclose(np.linalg.det(value), 1, atol=1e-9, rtol=0)):
        raise ValueError("A finite proper rotation is required")
    return value


def _axis(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value), 1, atol=1e-9, rtol=0):
        raise ValueError("The world twist axis must be a finite unit vector")
    return value / np.linalg.norm(value)


def _interval(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all() or value[1] < value[0] or value[1] - value[0] > 2 * math.pi + 1e-12:
        raise ValueError("A finite ordered interval of at most one turn is required")
    return float(value[0]), float(value[1])


def _multiply(a, b):
    return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]),
                 a[3] * b[3] - np.dot(a[:3], b[:3])]


def rotation_at_twist(base_rotation, axis_world, angle_rad):
    base = _rotation(base_rotation);axis = _axis(axis_world)
    if not np.isfinite(angle_rad):
        raise ValueError("Twist angle must be finite")
    return Rotation.from_rotvec(axis * angle_rad).as_matrix() @ base


@dataclass(frozen=True)
class OrientationTwistDomain:
    intervals: tuple[tuple[float, float], ...]
    dot_cosine_coefficient: float
    dot_sine_coefficient: float
    required_absolute_dot: float


def source_orientation_twist_domain(base_rotation, axis_world, source_rotation,
                                    orientation_cap_rad, search_interval=(-math.pi, math.pi)):
    """Solve |a cos(theta/2)+b sin(theta/2)| >= cos(cap/2).

    Source orientation is an explicitly supplied model hypothesis. The result
    is a floating-point closed-form domain for its geodesic constraint only;
    root, source-image, joint and native checks are separate.
    """
    base = Rotation.from_matrix(_rotation(base_rotation)).as_quat()
    source = Rotation.from_matrix(_rotation(source_rotation)).as_quat()
    axis = _axis(axis_world);lower, upper = _interval(search_interval)
    if not np.isfinite(orientation_cap_rad) or not 0 <= orientation_cap_rad <= math.pi:
        raise ValueError("Orientation cap must be finite and in [0,pi]")
    tangent = _multiply(np.r_[axis, 0.], base)
    a = float(np.dot(source, base));b = float(np.dot(source, tangent))
    threshold = math.cos(orientation_cap_rad / 2)
    if orientation_cap_rad == math.pi:
        return OrientationTwistDomain(((lower, upper),), a, b, 0.)
    amplitude = math.hypot(a, b)
    if amplitude < threshold - 2e-14 or amplitude == 0:
        return OrientationTwistDomain((), a, b, threshold)
    half_width = math.acos(min(1., threshold / amplitude))
    center = math.atan2(b, a)
    first = math.ceil((lower / 2 - center - half_width) / math.pi)
    last = math.floor((upper / 2 - center + half_width) / math.pi)
    intervals = []
    for k in range(first, last + 1):
        lo = max(lower, 2 * (center + k * math.pi - half_width))
        hi = min(upper, 2 * (center + k * math.pi + half_width))
        if lo <= hi:
            intervals.append((lo, hi))
    return OrientationTwistDomain(tuple(intervals), a, b, threshold)


def _sinusoid_range(a, b, interval):
    lower, upper = interval;phase = math.atan2(b, a)
    points = [lower, upper]
    points.extend(phase + k * math.pi for k in range(
        math.ceil((lower - phase) / math.pi), math.floor((upper - phase) / math.pi) + 1))
    values = [a * math.cos(x) + b * math.sin(x) for x in points]
    padding = 64 * np.finfo(float).eps * (1 + abs(a) + abs(b))
    return float(np.nextafter(min(values) - padding, -math.inf)), float(np.nextafter(max(values) + padding, math.inf))


def principal_rotvec_enclosure(base_rotation, axis_world, aligned_rotation, twist_interval):
    """Conservative numerical component envelope, with an explicit cut locus.

    It bounds the quaternion sinusoid and the monotone principal-log scale
    2*acos(w)/sqrt(1-w*w), using outward numerical padding. This is not a formal
    directed-rounding proof for libm. A hemisphere crossing remains unresolved.
    """
    base = _rotation(base_rotation);aligned = _rotation(aligned_rotation)
    axis = _axis(axis_world);interval = _interval(twist_interval)
    q0 = Rotation.from_matrix(base @ aligned.T).as_quat()
    q1 = _multiply(np.r_[axis, 0.], q0)
    half = (interval[0] / 2, interval[1] / 2)
    wrange = _sinusoid_range(q0[3], q1[3], half)
    if wrange[1] < 0:
        q0 = -q0;q1 = -q1;wrange = (-wrange[1], -wrange[0])
    if wrange[0] <= 0:
        return dict(component_bounds=[[-math.pi, math.pi]] * 3,
                    scalar_quaternion_bounds=list(wrange), hemisphere_crossing=True)
    wlo, whi = max(0., min(1., wrange[0])), max(0., min(1., wrange[1]))
    def log_scale(w):
        u = 1 - w
        if u < 1e-8:
            return 2 * (1 + u / 3 + 2 * u * u / 15)
        return 2 * math.acos(w) / math.sqrt(1 - w * w)
    hlo, hhi = log_scale(whi), log_scale(wlo)
    bounds = []
    for i in range(3):
        lo, hi = _sinusoid_range(q0[i], q1[i], half)
        values = [lo * hlo, lo * hhi, hi * hlo, hi * hhi]
        padding = 64 * np.finfo(float).eps * (1 + max(map(abs, values)))
        bounds.append([float(np.nextafter(min(values) - padding, -math.inf)),
                       float(np.nextafter(max(values) + padding, math.inf))])
    return dict(component_bounds=bounds, scalar_quaternion_bounds=list(wrange), hemisphere_crossing=False)


def partition_root_twist_domain(base_rotation, axis_world, source_rotation,
                                orientation_cap_rad, aligned_rotation,
                                component_caps, *, search_interval=(-math.pi, math.pi),
                                minimum_width_rad=1e-6, maximum_boxes=4096):
    """Keep every split/rejected/inside/unresolved interval and its envelope.

    Classification concerns the original component box within the supplied
    source-orientation domain, not fingers, images or native contacts. Budget
    exhaustion produces unresolved leaves, never a fabricated exclusion.
    """
    caps = np.asarray(component_caps, dtype=float)
    if caps.shape != (3,) or not np.isfinite(caps).all() or (caps < 0).any():
        raise ValueError("Three finite nonnegative component caps are required")
    if not np.isfinite(minimum_width_rad) or minimum_width_rad <= 0 or isinstance(maximum_boxes, bool) or not isinstance(maximum_boxes, int) or maximum_boxes < 2:
        raise ValueError("A positive width and integer box budget >=2 are required")
    domain = source_orientation_twist_domain(base_rotation, axis_world, source_rotation,
                                            orientation_cap_rad, search_interval)
    queue = deque((i, None, interval) for i, interval in enumerate(domain.intervals))
    next_id = len(queue);nodes = [];leaves = [];budget_hit = False
    while queue:
        ident, parent, interval = queue.popleft()
        enclosure = principal_rotvec_enclosure(base_rotation, axis_world, aligned_rotation, interval)
        bounds = np.array(enclosure['component_bounds'])
        if (bounds[:, 0] >= -caps).all() and (bounds[:, 1] <= caps).all():
            status = 'within_component_box'
        elif ((bounds[:, 1] < -caps) | (bounds[:, 0] > caps)).any():
            status = 'outside_component_box'
        else:
            status = 'unresolved'
        node = dict(id=ident, parent_id=parent, interval=list(interval),
                    classification=status, enclosure=enclosure)
        if status == 'unresolved' and interval[1] - interval[0] > minimum_width_rad:
            if len(nodes) + len(queue) + 3 <= maximum_boxes:
                middle = (interval[0] + interval[1]) / 2
                node['classification'] = 'split';node['children'] = [next_id, next_id + 1]
                queue.extend([(next_id, ident, (interval[0], middle)),
                              (next_id + 1, ident, (middle, interval[1]))])
                next_id += 2
            else:
                budget_hit = True;node['box_budget_exhausted'] = True
        nodes.append(node)
        if node['classification'] != 'split':
            leaves.append(ident)
    return dict(orientation_intervals=[list(x) for x in domain.intervals],
                orientation_dot_cosine=domain.dot_cosine_coefficient,
                orientation_dot_sine=domain.dot_sine_coefficient,
                required_absolute_dot=domain.required_absolute_dot,
                nodes=nodes, leaf_ids=leaves, box_budget_exhausted=budget_hit,
                numerical_enclosures_not_formal_global_proof=True)
