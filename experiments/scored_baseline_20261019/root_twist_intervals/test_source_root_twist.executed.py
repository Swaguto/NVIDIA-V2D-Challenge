import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from v2d_dex3_flashchord.source_root_twist import (
    partition_root_twist_domain, principal_rotvec_enclosure,
    rotation_at_twist, source_orientation_twist_domain,
)


def domain(source=None, cap=.1, **overrides):
    args = dict(base_rotation=np.eye(3), axis_world=[0, 0, 1],
                source_rotation=np.eye(3) if source is None else source,
                orientation_cap_rad=cap)
    args.update(overrides)
    return source_orientation_twist_domain(**args)


def test_identity_axis_orientation_domain():
    np.testing.assert_allclose(domain().intervals, [[-.1, .1]], atol=1e-12)


def test_shifted_domain_and_tangent_point():
    np.testing.assert_allclose(domain(Rotation.from_rotvec([0, 0, .05]).as_matrix()).intervals,
                               [[-.05, .15]], atol=1e-12)
    np.testing.assert_allclose(domain(cap=0).intervals, [[0, 0]], atol=1e-12)


def test_incompatible_perpendicular_orientation_is_empty():
    assert not domain(Rotation.from_rotvec([.3, 0, 0]).as_matrix(), cap=.2).intervals


def test_principal_rotation_seam_has_both_interval_pieces():
    result = domain(Rotation.from_rotvec([0, 0, math.pi]).as_matrix())
    np.testing.assert_allclose(result.intervals, [[-math.pi, -math.pi + .1],
                                                [math.pi - .1, math.pi]], atol=1e-12)


def test_pi_cap_accepts_full_declared_domain():
    assert domain(cap=math.pi, search_interval=(-.4, .3)).intervals == ((-.4, .3),)


def test_closed_form_agrees_with_independent_rotation_grid():
    rng = np.random.default_rng(14)
    grid = np.linspace(-math.pi, math.pi, 1201)
    for _ in range(8):
        base = Rotation.from_rotvec(rng.normal(size=3) * .3).as_matrix()
        axis = rng.normal(size=3);axis /= np.linalg.norm(axis)
        source = Rotation.from_rotvec(rng.normal(size=3) * .2).as_matrix() @ base
        result = domain(source, cap=.4, base_rotation=base, axis_world=axis)
        rotations = Rotation.from_rotvec(grid[:, None] * axis).as_matrix() @ base
        error = Rotation.from_matrix(source @ rotations.transpose(0, 2, 1)).magnitude()
        predicted = np.array([any(lo <= x <= hi for lo, hi in result.intervals) for x in grid])
        np.testing.assert_array_equal(predicted[abs(error - .4) > 1e-9], error[abs(error - .4) > 1e-9] <= .4)


def test_twist_preserves_the_mapped_point_and_world_transform():
    base = Rotation.from_rotvec([.2, -.3, .1]).as_matrix();point = np.array([.1, .05, -.02])
    axis = base @ point;axis /= np.linalg.norm(axis)
    for angle in [-.4, 0, .25]:
        np.testing.assert_allclose(rotation_at_twist(base, axis, angle) @ point,
                                   base @ point, atol=1e-12)
    Q = Rotation.from_rotvec([-.2, .1, .4]).as_matrix()
    a = source_orientation_twist_domain(base, axis, base, .3)
    b = source_orientation_twist_domain(Q @ base, Q @ axis, Q @ base, .3)
    np.testing.assert_allclose(a.intervals, b.intervals, atol=1e-12)


def test_component_envelopes_contain_independent_principal_logs():
    rng = np.random.default_rng(73)
    for _ in range(35):
        base = Rotation.from_rotvec(rng.normal(size=3) * .1).as_matrix()
        aligned = Rotation.from_rotvec(rng.normal(size=3) * .1).as_matrix()
        axis = rng.normal(size=3);axis /= np.linalg.norm(axis)
        lo = rng.uniform(-.2, .2);hi = lo + rng.uniform(.0001, .15)
        enclosure = principal_rotvec_enclosure(base, axis, aligned, [lo, hi])
        bounds = np.array(enclosure['component_bounds'])
        rotations = Rotation.from_rotvec(np.linspace(lo, hi, 81)[:, None] * axis).as_matrix() @ base
        values = Rotation.from_matrix(rotations @ aligned.T).as_rotvec()
        assert (values >= bounds[:, 0] - 1e-12).all()
        assert (values <= bounds[:, 1] + 1e-12).all()


def test_cut_locus_and_budget_exhaustion_remain_unresolved():
    e = principal_rotvec_enclosure(np.eye(3), [0, 0, 1], np.eye(3), [3., 3.3])
    assert e['hemisphere_crossing']
    p = partition_root_twist_domain(np.eye(3), [0, 0, 1], np.eye(3), .2,
                                    np.eye(3), [.08] * 3, maximum_boxes=2)
    assert p['box_budget_exhausted']
    assert p['nodes'][0]['classification'] == 'unresolved'


def test_partition_retains_full_cover_and_correct_numerical_classification():
    p = partition_root_twist_domain(np.eye(3), [0, 0, 1], np.eye(3), .2,
                                    np.eye(3), [.08] * 3, minimum_width_rad=1e-5)
    nodes = {n['id']: n for n in p['nodes']};leaves = [nodes[i] for i in p['leaf_ids']]
    assert not p['box_budget_exhausted']
    assert sum(n['interval'][1] - n['interval'][0] for n in leaves) == pytest.approx(.4)
    for n in leaves:
        values = np.linspace(*n['interval'], 9)
        if n['classification'] == 'within_component_box':
            assert (abs(values) <= .08 + 1e-12).all()
        elif n['classification'] == 'outside_component_box':
            assert (abs(values) >= .08 - 1e-12).all()
        else:
            assert n['interval'][1] - n['interval'][0] <= 1e-5
    for n in p['nodes']:
        if n['classification'] == 'split':
            left, right = [nodes[i] for i in n['children']]
            assert left['interval'][0] == n['interval'][0]
            assert left['interval'][1] == right['interval'][0]
            assert right['interval'][1] == n['interval'][1]


@pytest.mark.parametrize('overrides', [
    dict(axis_world=[0, 0, 2]), dict(axis_world=[0, 0, 0]),
    dict(base_rotation=np.diag([-1., 1., 1.])),
    dict(source_rotation=np.eye(3) * 2), dict(orientation_cap_rad=-.1),
    dict(orientation_cap_rad=np.inf), dict(search_interval=(1, -1)),
    dict(search_interval=(-4, 4)),
])
def test_domain_invalid_contracts_fail(overrides):
    with pytest.raises(ValueError):
        domain(**overrides)


@pytest.mark.parametrize('overrides', [
    dict(component_caps=[-1, .1, .1]), dict(minimum_width_rad=0),
    dict(maximum_boxes=1), dict(maximum_boxes=True),
])
def test_partition_invalid_contracts_fail(overrides):
    args = dict(base_rotation=np.eye(3), axis_world=[0, 0, 1],
                source_rotation=np.eye(3), orientation_cap_rad=.2,
                aligned_rotation=np.eye(3), component_caps=[.08] * 3)
    args.update(overrides)
    with pytest.raises(ValueError):
        partition_root_twist_domain(**args)
