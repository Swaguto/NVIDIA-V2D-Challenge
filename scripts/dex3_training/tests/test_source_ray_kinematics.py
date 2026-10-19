import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from v2d_dex3_flashchord.source_ray_kinematics import ray_bound_root_rotations


def solve(**overrides):
    args = dict(point_local=[.2, 0, 0], root_translation=[0, 0, 1],
                ray_origin=[0, 0, 0], ray_direction=[0, 0, 1],
                reference_rotation=np.eye(3))
    args.update(overrides)
    return ray_bound_root_rotations(**args)


def test_two_sphere_intersections_and_point_closure():
    result = solve()
    assert [b.parameter for b in result.branches] == pytest.approx([.8, 1.2])
    for branch in result.branches:
        np.testing.assert_allclose(branch.rotation @ [.2, 0, 0] + [0, 0, 1],
                                   [0, 0, branch.parameter], atol=1e-12)


def test_parameter_bounds_keep_only_declared_branch():
    assert [b.parameter for b in solve(parameter_bounds=(1., 2.)).branches] == pytest.approx([1.2])


def test_direction_parameter_is_not_silently_normalized():
    assert [b.parameter for b in solve(ray_direction=[0, 0, 2]).branches] == pytest.approx([.4, .6])


def test_tangency_deduplicates_and_preserves_reference_twist():
    result = solve(root_translation=[0, 0, 0], ray_origin=[.2, 0, -1])
    assert len(result.branches) == 1
    assert result.branches[0].parameter == pytest.approx(1.)
    np.testing.assert_allclose(result.branches[0].rotation, np.eye(3), atol=1e-12)


def test_missed_sphere_and_behind_ray_are_empty():
    assert not solve(ray_origin=[.5, 0, 0]).branches
    assert not solve(ray_direction=[0, 0, -1]).branches


def test_antiparallel_axis_is_flagged_not_claimed_unreachable():
    result = solve(root_translation=[0, 0, 0], ray_origin=[-2, 0, 0],
                   ray_direction=[1, 0, 0], parameter_bounds=(0., 2.))
    assert not result.branches
    assert result.antiparallel_alignment_count == 1


def test_rigid_change_of_world_coordinates_commutes():
    Q = Rotation.from_rotvec([.2, -.4, .1]).as_matrix()
    t = np.array([.3, -.5, .7])
    old = solve()
    new = solve(root_translation=Q @ [0, 0, 1] + t, ray_origin=t,
                ray_direction=Q @ [0, 0, 1], reference_rotation=Q)
    for a, b in zip(old.branches, new.branches, strict=True):
        assert a.parameter == pytest.approx(b.parameter)
        np.testing.assert_allclose(Q @ a.rotation, b.rotation, atol=1e-12)


def test_rotation_minimizes_change_under_the_declared_twist_convention():
    prior = Rotation.from_rotvec([.1, -.2, .3]).as_matrix()
    for b in solve(reference_rotation=prior).branches:
        best = Rotation.from_matrix(b.rotation @ prior.T).magnitude()
        axis = b.rotation @ np.array([1., 0, 0])
        for angle in np.linspace(-np.pi, np.pi, 31):
            other = Rotation.from_rotvec(angle * axis).as_matrix() @ b.rotation
            assert Rotation.from_matrix(other @ prior.T).magnitude() >= best - 1e-12


@pytest.mark.parametrize('overrides', [
    dict(point_local=[0, 0, 0]), dict(ray_direction=[0, 0, 0]),
    dict(point_local=[np.nan, 0, 0]), dict(ray_origin=[0, 0]),
    dict(reference_rotation=np.diag([-1., 1., 1.])),
    dict(reference_rotation=np.eye(3) * 2), dict(parameter_bounds=(-1., 2.)),
    dict(parameter_bounds=(2., 1.)), dict(parameter_bounds=(0., np.nan)),
])
def test_invalid_contracts_fail(overrides):
    with pytest.raises(ValueError):
        solve(**overrides)


def test_inputs_unchanged_and_returned_rotation_readonly():
    R = np.eye(3);point = np.array([.2, 0, 0])
    branch = solve(reference_rotation=R, point_local=point).branches[0]
    np.testing.assert_array_equal(R, np.eye(3))
    np.testing.assert_array_equal(point, [.2, 0, 0])
    with pytest.raises(ValueError):
        branch.rotation[0, 0] = 2
