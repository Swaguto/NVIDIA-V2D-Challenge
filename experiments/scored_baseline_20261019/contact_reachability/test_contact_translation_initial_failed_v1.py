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
    hand = obj - translation + gap*on
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
    hand = obj + gap*on
    result = common_translation_contact_seed(hand, -on, obj, on, np.zeros((3, 2)), signed_gap_m=gap)
    assert result['geometrically_feasible']
    np.testing.assert_array_equal(result['translation_m'], [0, 0, 0])


def test_barycentric_contact_can_move_without_matching_triangle_centroids():
    obj, on = facets(); hand = obj.copy(); hand[:, 0, 0] += .4
    result = common_translation_contact_seed(hand, -on, obj, on, np.zeros((3, 2)), signed_gap_m=0)
    assert result['geometrically_feasible']
    assert result['maximum_closure_error_m'] < 1e-12


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
