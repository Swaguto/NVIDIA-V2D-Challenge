"""Analytic virtual-work and effort-budget tests; no simulator dependency."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.contact_load_feasibility import (
    ContactGeometry, contact_effort_map,
    outer_actuated_contact_load_feasibility as check,
    outer_contact_load_feasibility,
)


def vertical_contact():
    return ContactGeometry([.1, 0, 0], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], 1, [0]*5)


def solve(load=2, cap=.3, gravity=.05):
    c = vertical_contact()
    # Rotation about -Y moves the contact in +Z with a 0.1 m moment arm.
    effort = contact_effort_map(c, [[0], [0], [.1]], [[0], [-1], [0]])
    return check([c], [.1, 0, 0], [0, 0, load, 0, 0, 0],
                 effort_maps=[effort], free_effort_Nm=[gravity],
                 effort_bounds_Nm=[[-cap, cap]])


def test_actual_lever_arm_and_gravity_consume_joint_effort():
    result = solve()
    assert result['feasible_in_outer_model']
    assert result['solved_required_joint_effort_Nm'] == pytest.approx([.25])
    assert result['motor_limits_applied']
    assert result['force_origin'] == 'solved_not_measured'
    assert not solve(cap=.24)['feasible_in_outer_model']
    assert solve(load=2, cap=.15, gravity=-.1)['feasible_in_outer_model']


def test_native_cap_cannot_be_replaced_by_larger_asset_rating():
    # Required 0.8 Nm is below an asset's 1.4 Nm rating, above runtime 0.76 Nm.
    assert solve(load=8, gravity=0, cap=1.4)['feasible_in_outer_model']
    assert not solve(load=8, gravity=0, cap=.76)['feasible_in_outer_model']
    c = vertical_contact()
    assert outer_contact_load_feasibility([c], [.1, 0, 0], [0, 0, 8, 0, 0, 0])['feasible_in_outer_model']


def test_spin_and_rolling_virtual_work_use_angular_jacobian():
    c = ContactGeometry([0]*3, np.eye(3), 6, [1, 1, .02, .03, .04])
    linear = np.array([[0, .1], [.2, 0], [0, 0]])
    angular = np.array([[1, 0], [0, 2], [3, 0]])
    m = contact_effort_map(c, linear, angular)
    components = np.array([3, .1, -.2, .01, -.02, .03])
    qdot = np.array([.7, -.4])
    # Joint virtual power equals Cartesian force/moment power.
    assert qdot @ m @ components == pytest.approx(
        (linear @ qdot) @ components[:3] + (angular @ qdot) @ components[3:])


def test_redistribution_can_satisfy_joint_caps_instead_of_rejecting_one_force_solution():
    c = ContactGeometry([0]*3, np.eye(3), 1, [0]*5)
    # Two colocated assumed contacts can share 4 N, limited to 1 N and 3 N.
    result = check([c, c], [0]*3, [4, 0, 0, 0, 0, 0],
        effort_maps=[[[1], [0]], [[0], [1]]], free_effort_Nm=[0, 0],
        effort_bounds_Nm=[[0, 1], [0, 3]])
    assert result['feasible_in_outer_model']
    assert result['solved_required_joint_effort_Nm'] == pytest.approx([1, 3])
    assert not check([c, c], [0]*3, [4, 0, 0, 0, 0, 0],
        effort_maps=[[[1], [0]], [[0], [1]]], free_effort_Nm=[0, 0],
        effort_bounds_Nm=[[0, 1], [0, 2.9]])['feasible_in_outer_model']


def test_free_effort_matters_even_without_object_load_or_contacts():
    args = dict(effort_maps=[], free_effort_Nm=[.2], effort_bounds_Nm=[[-.1, .1]])
    assert not check([], [0]*3, [0]*6, **args)['feasible_in_outer_model']
    args['free_effort_Nm'] = [.05]
    assert check([], [0]*3, [0]*6, **args)['feasible_in_outer_model']


@pytest.mark.parametrize('field,value', [
    ('effort_maps', []), ('effort_maps', [[[np.nan]]]),
    ('effort_maps', [[[1, 2]]]), ('free_effort_Nm', []),
    ('free_effort_Nm', [np.inf]), ('free_effort_Nm', [[0]]),
    ('effort_bounds_Nm', [[1, -1]]), ('effort_bounds_Nm', [[0, np.inf]]),
])
def test_invalid_effort_contracts_fail_closed(field, value):
    args = dict(effort_maps=[[[.1]]], free_effort_Nm=[0], effort_bounds_Nm=[[-1, 1]])
    args[field] = value
    with pytest.raises(ValueError):
        check([vertical_contact()], [.1, 0, 0], [0, 0, 1, 0, 0, 0], **args)


@pytest.mark.parametrize('linear,angular', [
    ([[0], [0]], [[0]]), (np.empty((3, 0)), np.empty((3, 0))),
    ([[0], [0], [np.inf]], [[0], [0], [0]]),
    ([[0], [0], [0]], np.zeros((3, 2))),
])
def test_bad_contact_jacobians_are_rejected(linear, angular):
    with pytest.raises(ValueError):
        contact_effort_map(vertical_contact(), linear, angular)
