"""Construct a common bounded root translation for fixed contact facets.

This is a geometric feasibility problem. Finger configurations and root
orientation are fixed by the caller. Barycentric points on both facets may
move; proximity to a single preselected triangle point is not required.
No native contact, friction, effort or physical success is certified here.
"""
import numpy as np
from scipy.optimize import linprog


def _finite(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'Need finite {name}{shape}')
    return array


def _facets(triangles, normals, count, name):
    triangles = _finite(triangles, (count, 3, 3), name + ' triangles')
    normals = _finite(normals, (count, 3), name + ' normals')
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=1)
    if np.any(area < 1e-14):
        raise ValueError('Degenerate contact facet')
    if (not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-8, rtol=0)
            or np.any(abs(np.einsum('ni,ni->n', cross / area[:, None], normals)) < 1 - 1e-7)):
        raise ValueError('Normals must be unit and perpendicular to facets')
    return triangles, normals


def common_translation_contact_seed(hand_triangles_world_m, hand_outward_normals_world,
                                    object_triangles_world_m, object_outward_normals_world,
                                    translation_bounds_m, *, signed_gap_m=-0.00025):
    """Find the smallest L-infinity common root translation for 2/3 facets.

    For each pair, ``hand_point + translation = object_point + gap * normal``.
    The first object normal must oppose at least one other object normal;
    every hand normal must face its corresponding object surface. Winding is
    not used to invent outward signs. Source authority and external-union
    membership remain the caller's responsibility.

    Returns the solver's final result with independently checked barycentric,
    bound and closure residuals. Infeasibility concerns these fixed facets and
    configurations, not the continuous hand workspace or task as a whole.
    """
    raw = np.asarray(hand_triangles_world_m)
    if raw.ndim != 3 or len(raw) not in (2, 3):
        raise ValueError('Need two or three contact facet pairs')
    count = len(raw)
    hand, hn = _facets(raw, hand_outward_normals_world, count, 'hand')
    obj, on = _facets(object_triangles_world_m, object_outward_normals_world, count, 'object')
    limits = _finite(translation_bounds_m, (3, 2), 'translation bounds')
    if np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError('Translation lower bounds exceed upper bounds')
    if isinstance(signed_gap_m, (bool, np.bool_)) or not np.isscalar(signed_gap_m) or not np.isfinite(signed_gap_m):
        raise ValueError('Need a finite signed gap')
    if np.any(np.einsum('ni,ni->n', hn, on) > -0.5):
        raise ValueError('Hand facets must face their object facets')
    if np.min(on[1:] @ on[0]) > -0.5:
        raise ValueError('First object contact must oppose another contact')

    # Three root components, six barycentric weights/contact, one epigraph.
    size = 4 + 6*count
    eq, rhs = [], []
    for i in range(count):
        row = np.zeros((3, size)); row[:, :3] = np.eye(3)
        row[:, 3 + 6*i:6 + 6*i] = hand[i].T
        row[:, 6 + 6*i:9 + 6*i] = -obj[i].T
        eq.extend(row); rhs.extend(float(signed_gap_m)*on[i])
        for start in [3 + 6*i, 6 + 6*i]:
            row = np.zeros(size); row[start:start + 3] = 1
            eq.append(row); rhs.append(1.)
    inequality = np.zeros((6, size))
    inequality[:3, :3] = np.eye(3); inequality[3:, :3] = -np.eye(3)
    inequality[:, -1] = -1
    objective = np.zeros(size); objective[-1] = 1
    result = linprog(objective, A_ub=inequality, b_ub=np.zeros(6),
        A_eq=np.asarray(eq), b_eq=np.asarray(rhs),
        bounds=[tuple(row) for row in limits] + [(0, 1)]*(6*count) + [(0, None)],
        method='highs')
    output = dict(geometrically_feasible=False, solver_status=int(result.status),
        solver_message=str(result.message), fixed_facets_only_not_global_reachability=True,
        native_contacts_certified=False, forces_measured=False,
        motor_or_physical_success_certified=False)
    if not result.success:
        return output
    x = np.asarray(result.x)
    hb = np.stack([x[3+6*i:6+6*i] for i in range(count)])
    ob = np.stack([x[6+6*i:9+6*i] for i in range(count)])
    hp = np.einsum('ni,nij->nj', hb, hand)
    op = np.einsum('ni,nij->nj', ob, obj)
    closure = np.max(abs(hp + x[:3] - op - float(signed_gap_m)*on))
    bound_error = max(0., float(np.max(limits[:, 0]-x[:3])),
                      float(np.max(x[:3]-limits[:, 1])),
                      float(-hb.min()), float(-ob.min()),
                      float(abs(hb.sum(1)-1).max()), float(abs(ob.sum(1)-1).max()),
                      float(abs(x[:3]).max()-x[-1]))
    if closure > 1e-8 or bound_error > 1e-8:
        output.update(independent_verification_failed=True,
                      maximum_closure_error_m=float(closure), maximum_bound_error=float(bound_error))
        return output
    output.update(geometrically_feasible=True, translation_m=x[:3].tolist(),
        hand_barycentric=hb.tolist(), object_barycentric=ob.tolist(),
        hand_points_before_translation_world_m=hp.tolist(), object_points_world_m=op.tolist(),
        maximum_closure_error_m=float(closure), maximum_bound_error=float(bound_error),
        translation_Linf_m=float(abs(x[:3]).max()))
    return output
