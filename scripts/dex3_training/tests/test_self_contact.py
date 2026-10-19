from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.self_contact import summarize_self_contacts

BODIES = {'left': ['L_palm', 'L_thumb'], 'right': ['R_palm', 'R_thumb']}


def contact(hand, gap):
    return dict(hand=hand, native_bodies=BODIES[hand], signed_gap_m=gap)


def summarize(rows, bodies=BODIES, count=2):
    return summarize_self_contacts(rows, bodies, reference_frame_count=count)


def test_both_hands_and_empty_knots_are_assessed():
    rows = [dict(reference_frame=0, contacts=[contact('left', -.002), contact('right', -.0005)]),
            dict(reference_frame=1, contacts=[])]
    result = summarize(rows)
    assert not result['hand_self_contact_gate_passed']
    assert result['summary_by_hand']['left']['frames_over_tolerance'] == 1
    assert result['summary_by_hand']['right']['frames_over_tolerance'] == 0
    assert not result['forces_measured'] and not result['dynamic_or_grasp_success_certified']


def test_multiple_contacts_count_one_bad_frame_and_strict_one_mm_boundary():
    rows = [dict(reference_frame=0, contacts=[contact('right', -.001), contact('right', .01)]),
            dict(reference_frame=1, contacts=[contact('right', -.002), contact('right', -.003)])]
    result = summarize(rows)
    assert result['summary_by_hand']['right']['frames_over_tolerance'] == 1
    assert result['summary_by_pair'][0]['maximum_penetration_m'] == .003
    assert result['summary_by_pair'][0]['worst_reference_frame'] == 1


@pytest.mark.parametrize('rows', [
    [], [dict(reference_frame=0, contacts=[])],
    [dict(reference_frame=1, contacts=[]), dict(reference_frame=0, contacts=[])],
    [dict(reference_frame=0, contacts=[]), dict(reference_frame=0, contacts=[])],
    [dict(reference_frame=False, contacts=[]), dict(reference_frame=1, contacts=[])],
])
def test_missing_truncated_duplicate_or_reordered_clock_is_rejected(rows):
    with pytest.raises(ValueError):
        summarize(rows)


@pytest.mark.parametrize('bad', [
    dict(hand='left', native_bodies=['L_palm', 'R_thumb'], signed_gap_m=-.002),
    dict(hand='right', native_bodies=['R_palm', 'unknown'], signed_gap_m=-.002),
    dict(hand='right', native_bodies=['R_palm', 'R_palm'], signed_gap_m=-.002),
    contact('right', np.nan), contact('right', np.inf), contact('right', True),
])
def test_bad_body_or_gap_cannot_be_silently_ignored(bad):
    with pytest.raises(ValueError):
        summarize([dict(reference_frame=0, contacts=[bad]), dict(reference_frame=1, contacts=[])])


def test_ambiguous_body_binding_is_rejected():
    with pytest.raises(ValueError):
        summarize([dict(reference_frame=0, contacts=[])]*2,
            bodies={'left': ['same', 'L'], 'right': ['same', 'R']})
