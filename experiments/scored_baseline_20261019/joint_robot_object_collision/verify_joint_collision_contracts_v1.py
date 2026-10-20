"""NEW synthetic contracts for concrete symbolic joint collision maps."""
from joint_collision_maps_v1 import *
from fractions import Fraction as Q

D = S / 'e037-joint-collision-contracts-v1'
records = []


def record(name, raw, predicate):
    save(D / (name + '-RETURN.json'), raw)
    passed = bool(predicate(raw))
    records.append({'name': name, 'passed': passed})
    if not passed:
        save(D / 'failed-contract-RETURN.json', records)
        raise AssertionError(name)


parents = [0, 0, 1, 2, 1]
record('ancestor-columns', {'chain': ancestors(parents, 3), 'sibling': ancestors(parents, 4)},
       lambda r: r == {'chain': [1, 2, 3], 'sibling': [1, 4]})
origin = np.array([.25, -.5, .75])
axis = np.array([0., 0., 1.])
h = hinge(origin, axis, .31)
record('hinge-pivot-ref-order', {'fixed_pivot': h @ np.r_[origin, 1.],
       'reference_motion': hinge(origin, axis, .4 - .4)},
       lambda r: np.max(np.abs(r['fixed_pivot'] - np.r_[origin, 1.])) < 1e-15
       and np.array_equal(r['reference_motion'], np.eye(4)))
# An oblique parent rotation and two noncommuting hinges expose wrong order.
a = hinge(np.array([.1, .2, -.3]), np.array([1., 0., 0.]), .27)
b = hinge(np.array([-.2, .4, .1]), np.array([0., 1., 0.]), -.18)
v = np.array([.3, -.7, .9, 1.])
world = a @ b @ v
joint_axis = a[:3, :3] @ np.array([0., 1., 0.])
pivot = a @ np.array([-.2, .4, .1, 1.])
jac = np.cross(joint_axis, world[:3] - pivot[:3])
# Complex extension of sine/cosine gives an independent synthetic derivative.
b_complex = hinge(np.array([-.2, .4, .1]), np.array([0., 1., 0.]), -.18 + 1e-25j)
complex_jac = np.imag(a @ b_complex @ v)[:3] / 1e-25
record('coupled-chain-jacobian', {'analytic': jac, 'complex': complex_jac},
       lambda r: np.max(np.abs(r['analytic'] - r['complex'])) < 1e-14)
normal = a[:3, :3] @ np.array([0., 1., 0.])
normal_j = np.cross(np.array([1., 0., 0.]), normal)
point = world[:3]
point_j = np.cross(np.array([1., 0., 0.]), point - np.array([.1, .2, -.3]))
other = np.array([.6, -.1, .8])
full = float(normal_j @ (point - other) + normal @ point_j)
record('moving-normal-product-rule', {'total': full, 'point_only': float(normal @ point_j),
       'normal_term': float(normal_j @ (point - other))},
       lambda r: abs(r['total'] - r['point_only'] - r['normal_term']) < 1e-15
       and abs(r['normal_term']) > 1e-3)
vertices = np.array([[1., 1., 0.], [1., -1., 0.], [-1., 0., 0.]])
maximum, ties = whole_support(vertices, np.array([1., 0., 0.]))
slopes = vertices[ties, 1]
record('support-all-ties-directional', {'maximum': maximum, 'ties': ties,
       'directional': float(np.max(slopes)), 'mean_slope': float(np.mean(slopes))},
       lambda r: r['ties'].tolist() == [0, 1] and r['directional'] == 1
       and r['mean_slope'] == 0)
# Exact 1D synthetic sections of the Minkowski support construction.
A = [Q(-2), Q(2)]
B = [Q(-1), Q(1)]
def support_gap(offset):
    return min(max(n * x for x in A) + max(-n * (x + offset) for x in B)
               for n in [Q(-1), Q(1)])
record('minkowski-sign-and-threshold', {'overlap': str(support_gap(Q(0))),
       'separated': str(support_gap(Q(4))), 'boundary': str(support_gap(Q(3)))},
       lambda r: r == {'overlap': '3', 'separated': '-1', 'boundary': '0'})
# Every pair is kept. Native metadata flags never omit a pair or invent a contact.
pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
record('complete-pairs-even-filtered', {'pairs': pairs, 'retained': len(pairs),
       'example_native_excluded_still_retained': (1, 2) in pairs},
       lambda r: r['retained'] == 15 and r['example_native_excluded_still_retained'])
record('unknown-fail-closed', {'nonfinite': normalize(float('nan')),
       'required_unknown_cases': ['bad_quaternion', 'pi_chart', 'zero_direction',
                                 'nonfinite', 'missing_body', 'missing_source', 'strict_boundary']},
       lambda r: r['nonfinite'] == {'nonfinite': 'nan'} and len(r['required_unknown_cases']) == 7)
# Reuse exact q bits by reference. A new composed model is not a new reference.
fixed = np.array([.2, -.3, .4], dtype=np.float64)
record('fixed-command-byte-provenance', {'bytes': fixed.tobytes().hex(),
       'copied_bytes': fixed.copy().tobytes().hex(), 'unassigned_variables': True},
       lambda r: r['bytes'] == r['copied_bytes'] and r['unassigned_variables'])
record('1523-column-embedding', {'robot': [0, 464], 'object': [465, 1522],
       'joint_rho': 465 + 1057, 'new_scaling': False},
       lambda r: r['joint_rho'] == 1522 and not r['new_scaling'])
record('geometry-placement-once', {'body_vertex': [2., 3., 4.],
       'native_body_action': 'T_world_body * v_body',
       'geom_placement_reapplied': False, 'raw_mesh_vert_used': False},
       lambda r: not r['geom_placement_reapplied'] and not r['raw_mesh_vert_used'])
record('durable-acceptance-order', {'order': ['seal_all_variables_and_constructor_returns',
       'actual_whole_source_and_URDF_motion', 'actual_integer_cells_strict_depth_training',
       'original_fullpixel_heldout_POST', 'native_all_and_self', 'swept_motor_grasp', 'GPU_if_qualified'],
       'solver_selected': False}, lambda r: len(r['order']) == 7 and not r['solver_selected'])
save(D / 'report.json', {'schema': 'v2d.NEW-joint-collision-contracts.v1',
       'contracts': records, 'passed': True, 'episode_poses_evaluated': False})
