from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.contact_translation import common_translation_contact_seed


def facets():
    # Opposed horizontal contacts. Each triangle has a free interior point.
    obj = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    [[0, 0, 1], [1, 0, 1], [0, 1, 1]]], float)
    on = np.array([[0, 0, 1], [0, 0, -1]], float)
    return obj, on


def solve(translation=(0, 0, .003), limits=None, gap=0):
    obj, on = facets()
    hand = obj - translation + gap*on[:, None, :]
    return common_translation_contact_seed(hand, -on, obj, on,
        np.tile([-.008, .008], (3, 1)) if limits is None else limits, signed_gap_m=gap)


def test_shared_root_translation_closes_both_opposing_facets():
    result = solve()
    assert result['geometrically_feasible']
    assert result['translation_m'][2] == pytest.approx(.003)
    assert result['translation_Linf_m'] == pytest.approx(.003)
    assert result['maximum_closure_error_m'] < 1e-12
    assert not result['native_contacts_certified']
    assert not result['motor_or_physical_success_certified']


def test_incompatible_contacts_are_not_solved_with_two_independent_roots():
    obj, on = facets(); hand = obj.copy()
    hand[0, :, 2] -= .003; hand[1, :, 2] += .003
    result = common_translation_contact_seed(hand, -on, obj, on,
        np.tile([-.008, .008], (3, 1)), signed_gap_m=0)
    assert not result['geometrically_feasible']
    assert result['solver_status'] == 2


def test_out_of_box_translation_does_not_enlarge_source_uncertainty():
    assert not solve(translation=(0, 0, .02))['geometrically_feasible']


def test_fixed_zero_root_and_signed_overlap_are_supported():
    obj, on = facets(); gap = -.00025
    hand = obj + gap*on[:, None, :]
    result = common_translation_contact_seed(hand, -on, obj, on, np.zeros((3, 2)), signed_gap_m=gap)
    assert result['geometrically_feasible']
    np.testing.assert_array_equal(result['translation_m'], [0, 0, 0])


def test_barycentric_contact_can_move_without_matching_triangle_centroids():
    obj, on = facets(); hand = obj.copy(); hand[:, 0, 0] += .4
    result = common_translation_contact_seed(hand, -on, obj, on, np.zeros((3, 2)), signed_gap_m=0)
    assert result['geometrically_feasible']
    assert result['maximum_closure_error_m'] < 1e-12


def test_whole_hull_plane_can_reject_an_otherwise_exact_facet_closure():
    obj, on = facets(); hand = obj - [0, 0, .003]
    # Meeting the chosen facets needs dz=.003, but another whole-hull support
    # plane allows only dz<=.001. Independent per-facet proximity is insufficient.
    result = common_translation_contact_seed(hand, -on, obj, on,
        np.tile([-.008, .008], (3, 1)), signed_gap_m=0,
        translation_halfspaces=(np.array([[0, 0, 1]]), np.array([.001])))
    assert not result['geometrically_feasible']


def test_halfspace_is_verified_without_inventing_native_contact_truth():
    obj, on = facets(); hand = obj - [0, 0, .003]
    result = common_translation_contact_seed(hand, -on, obj, on,
        np.tile([-.008, .008], (3, 1)), signed_gap_m=0,
        translation_halfspaces=(np.array([[0, 0, -1], [1, 0, 0]]), np.array([-.002, .001])))
    assert result['geometrically_feasible']
    assert result['maximum_bound_error'] < 1e-12
    assert not result['native_contacts_certified']


@pytest.mark.parametrize('halfspaces', [
    ([[0, 0, 0]], [1]), ([[1, 0]], [1]), ([[1, 0, 0]], []),
    ([[np.nan, 0, 0]], [1]), (np.eye(3), np.ones(2)),
])
def test_invalid_whole_hull_constraints_fail_closed(halfspaces):
    obj, on = facets()
    with pytest.raises(ValueError):
        common_translation_contact_seed(obj, -on, obj, on,
            np.tile([-.008, .008], (3, 1)), translation_halfspaces=halfspaces)


@pytest.mark.parametrize('kind', ['normal', 'facing', 'opposition', 'degenerate', 'bounds', 'nan'])
def test_invalid_geometry_fails_closed(kind):
    obj, on = facets(); hand = obj.copy(); hn = -on; limits = np.tile([-.008, .008], (3, 1))
    if kind == 'normal': on[0] *= 2
    elif kind == 'facing': hn[0] *= -1
    elif kind == 'opposition': on[1] *= -1; hn[1] *= -1
    elif kind == 'degenerate': hand[0] = 0
    elif kind == 'bounds': limits[0] = [1, -1]
    elif kind == 'nan': hand[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        common_translation_contact_seed(hand, hn, obj, on, limits)


def test_three_facets_share_one_root_and_independent_barycentric_points():
    obj, on = facets()
    obj = np.concatenate([obj, [[[0, 0, 0], [0, 1, 0], [0, 0, 1]]]])
    on = np.concatenate([on, [[1, 0, 0]]])
    translation = np.array([.002, -.001, .003])
    gap = -.00025
    hand = obj - translation + gap*on[:, None, :]
    result = common_translation_contact_seed(hand, -on, obj, on,
        np.tile([-.008, .008], (3, 1)), signed_gap_m=gap)
    assert result['geometrically_feasible']
    assert len(result['hand_barycentric']) == 3
    assert result['maximum_closure_error_m'] < 1e-12
    np.testing.assert_allclose(np.asarray(result['translation_m'])[[0, 2]], [.002, .003])
