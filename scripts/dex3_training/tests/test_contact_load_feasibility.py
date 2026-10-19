"""Mechanics and contracts for the optimistic contact-load diagnostic."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
import v2d_dex3_flashchord.contact_load_feasibility as module
from v2d_dex3_flashchord.contact_load_feasibility import (
    ContactGeometry, outer_contact_load_feasibility as check,
)


def contact(point, normal, dim=3, friction=(1, 1, 0, 0, 0)):
    normal = np.asarray(normal, float)
    tangent = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
    tangent /= np.linalg.norm(tangent)
    return ContactGeometry(point, np.stack([normal, tangent, np.cross(normal, tangent)]), dim, friction)


def opposed(dim=3, friction=(1, 1, 0, 0, 0)):
    return [contact([-.1, 0, 0], [1, 0, 0], dim, friction),
            contact([.1, 0, 0], [-1, 0, 0], dim, friction)]


def test_opposed_pair_supports_gravity_but_one_side_cannot_balance_it():
    pair = opposed(friction=(.5, .5, 0, 0, 0))
    load = [0, 0, 3, 0, 0, 0]
    result = check(pair, [0, 0, 0], load)
    assert result['feasible_in_outer_model']
    assert result['minimum_total_normal_force_N'] == pytest.approx(6)
    assert result['force_origin'] == 'solved_not_measured'
    assert not result['motor_limits_applied']
    assert not check(pair[:1], [0, 0, 0], load)['feasible_in_outer_model']


def test_point_pair_cannot_balance_axial_torque_but_spin_contacts_can():
    load = [0, 0, 0, 1, 0, 0]
    assert not check(opposed(), [0, 0, 0], load)['feasible_in_outer_model']
    # Spin coefficient has units of metres: a .02 m radius needs total 50 N.
    result = check(opposed(4, (0, 0, .02, 0, 0)), [0, 0, 0], load)
    assert result['feasible_in_outer_model']
    assert result['minimum_total_normal_force_N'] == pytest.approx(50)
    # Rolling cannot replace spin about the normal axis.
    assert not check(opposed(6, (0, 0, 0, 1, 1)), [0, 0, 0], load)['feasible_in_outer_model']


def test_rolling_coefficients_are_about_tangent_axes_and_sliding_is_anisotropic():
    # In this basis tangent1 is +Z, tangent2 is -Y. Suppress sliding entirely.
    result = check(opposed(6, (0, 0, 0, .01, 0)), [0, 0, 0], [0, 0, 0, 0, 0, 1])
    assert result['feasible_in_outer_model']
    assert result['minimum_total_normal_force_N'] == pytest.approx(100)
    assert not check(opposed(6, (0, 0, 0, .01, 0)), [0, 0, 0], [0, 0, 0, 0, 1, 0])['feasible_in_outer_model']
    assert check(opposed(3, (.5, 0, 0, 0, 0)), [0, 0, 0], [0, 0, 1, 0, 0, 0])['feasible_in_outer_model']
    assert not check(opposed(3, (.5, 0, 0, 0, 0)), [0, 0, 0], [0, 1, 0, 0, 0, 0])['feasible_in_outer_model']


def test_com_moment_and_rigid_world_transform_are_respected():
    pair, load = opposed(), np.array([0, 0, 1, 0, 0, 0])
    # A COM displaced along Y needs an X torque this point pair cannot supply.
    assert not check(pair, [0, .1, 0], load)['feasible_in_outer_model']
    rot = Rotation.from_rotvec([.3, -.4, .2]).as_matrix()
    shift = np.array([1, -2, 3])
    transformed = [ContactGeometry(rot @ c.point_world_m + shift,
                   c.frame_into_object_world @ rot.T, c.dimension, c.friction) for c in pair]
    result = check(transformed, shift, np.r_[rot @ load[:3], rot @ load[3:]], moment_length_m=.23)
    assert result['feasible_in_outer_model']
    assert result['minimum_total_normal_force_N'] == pytest.approx(1)


def test_native_object_binding_reverses_force_axes_including_left_handed_basis():
    native = np.eye(3)
    into_geom1 = ContactGeometry.from_mujoco_geometry(
        point_world_m=[0, 0, 0], frame_geom1_to_geom2_world=native,
        dimension=1, friction=[0]*5, object_is_geom2=False)
    assert np.linalg.det(into_geom1.frame_into_object_world) == pytest.approx(-1)
    assert check([into_geom1], [0, 0, 0], [-1, 0, 0, 0, 0, 0])['feasible_in_outer_model']
    assert not check([into_geom1], [0, 0, 0], [1, 0, 0, 0, 0, 0])['feasible_in_outer_model']
    with pytest.raises(ValueError, match='boolean'):
        ContactGeometry.from_mujoco_geometry(point_world_m=[0]*3,
            frame_geom1_to_geom2_world=native, dimension=1, friction=[0]*5, object_is_geom2=1)


def test_zero_load_and_no_contact_are_not_ambiguous():
    assert check([], [0]*3, [0]*6)['feasible_in_outer_model']
    assert not check([], [0]*3, [0, 0, 1e-12, 0, 0, 0])['feasible_in_outer_model']
    result = check(opposed(), [0]*3, [0]*6)
    assert result['feasible_in_outer_model']
    assert result['minimum_total_normal_force_N'] == 0


def test_outer_box_feasibility_does_not_assert_exact_friction_cone_feasibility():
    c = ContactGeometry([0]*3, np.eye(3), 3, [1, 1, 0, 0, 0])
    result = check([c], [0]*3, [1, 1, 1, 0, 0, 0])
    assert result['feasible_in_outer_model']
    normal, t1, t2 = result['solved_contact_components'][0]
    # This accepted box corner lies outside the unit-coefficient elliptic cone.
    assert np.hypot(t1, t2) > normal
    assert not result['certifies_force_closure_or_dynamic_or_task_success']


@pytest.mark.parametrize('dimension', [True, 1.0, 0, 2, 5, 7])
def test_invalid_native_dimensions_are_rejected(dimension):
    with pytest.raises(ValueError, match='dimension'):
        ContactGeometry([0]*3, np.eye(3), dimension, [1]*5)


@pytest.mark.parametrize('field,value', [
    ('point_world_m', [[0, 0, 0]]), ('point_world_m', [np.nan, 0, 0]),
    ('point_world_m', [1j, 0, 0]),
    ('frame_into_object_world', np.ones((3, 3))),
    ('frame_into_object_world', np.eye(3).reshape(-1)),
    ('friction', [1, 1]), ('friction', [1, 1, -1, 0, 0]),
    ('friction', [1, np.inf, 0, 0, 0]),
])
def test_invalid_geometry_is_rejected(field, value):
    args = dict(point_world_m=[0]*3, frame_into_object_world=np.eye(3), dimension=3, friction=[1]*5)
    args[field] = value
    with pytest.raises(ValueError):
        ContactGeometry(**args)


@pytest.mark.parametrize('com,load,length', [
    ([0]*2, [0]*6, .1), ([0]*3, [0]*5, .1),
    ([0]*3, [0, 0, np.nan, 0, 0, 0], .1), ([0]*3, [0]*6, 0),
    ([0]*3, [0]*6, np.inf), ([0]*3, [0]*6, True),
    ([0]*3, [0]*6, '0.1'), ([0]*3, [0]*6, .1j),
])
def test_invalid_load_and_conditioning_are_rejected(com, load, length):
    with pytest.raises(ValueError):
        check([], com, load, moment_length_m=length)


def test_unresolved_solver_or_bad_equilibrium_is_not_a_physics_rejection(monkeypatch):
    for solved in [SimpleNamespace(status=4, success=False, message='numerical failure'),
                   SimpleNamespace(status=0, success=True, message='optimal', x=np.zeros(6))]:
        monkeypatch.setattr(module, 'linprog', lambda *a, **kw: solved)
        with pytest.raises(RuntimeError):
            check(opposed(), [0]*3, [0, 0, 1, 0, 0, 0])
