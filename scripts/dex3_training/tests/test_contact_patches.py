from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.contact_patches import ObjectContactPatch, RobotContactMode

SHA = 'a' * 64
TRI = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)


def patch(**kwargs):
    args = dict(patch_id='face', object_id='object', geometry_sha256=SHA,
                triangles_local_m=TRI, outward_normals_local=[[0, 0, 1]])
    return ObjectContactPatch(**(args | kwargs))


def test_closest_surface_interior_does_not_use_nearest_vertex():
    p = patch()
    result = p.closest_local([[.25, .25, 2], [2, 0, 0]])
    np.testing.assert_allclose(result.points, [[.25, .25, 0], [1, 0, 0]])
    np.testing.assert_allclose(result.distances_m, [2, 1])
    np.testing.assert_array_equal(result.outward_normals, [[0, 0, 1]]*2)
    assert p.closest_local(np.empty((0, 3))).points.shape == (0, 3)


def test_patch_moves_with_object_pose_and_preserves_normal_distance():
    # 90 degrees about Y, then translation: local +Z becomes world +X.
    transform = np.array([[0, 0, 1, 2], [0, 1, 0, 3], [-1, 0, 0, 4], [0, 0, 0, 1]])
    result = patch().closest_world([[4, 3.25, 3.75]], transform)
    np.testing.assert_allclose(result.points, [[2, 3.25, 3.75]])
    np.testing.assert_allclose(result.outward_normals, [[1, 0, 0]])
    assert result.distances_m[0] == pytest.approx(2)


def test_patch_geometry_is_copied_and_outward_sign_is_explicit_not_winding():
    tri = TRI.copy()
    p = patch(triangles_local_m=tri, outward_normals_local=[[0, 0, -1]])
    tri[:] = 99
    assert p.triangles_local_m[0, 0, 0] == 0
    assert not p.triangles_local_m.flags.writeable
    np.testing.assert_array_equal(p.closest_local([[.1, .1, .2]]).outward_normals, [[0, 0, -1]])


def test_two_digit_mode_does_not_fabricate_middle_contact_or_source_truth():
    mode = RobotContactMode('pinch', 'right', 'object', SHA, (('thumb', 'face'), ('index', 'face')))
    assert mode.required_digits == ('thumb', 'index')
    proof = mode.validate_patch_bindings({'face': patch()})
    assert proof['mode_is_hypothesis']
    assert not proof['contacts_observed'] and not proof['physical_success_certified']
    three = RobotContactMode('three', 'left', 'object', SHA,
        (('thumb', 'face'), ('index', 'face'), ('middle', 'face')))
    three.validate_patch_bindings({'face': patch()})


@pytest.mark.parametrize('kwargs', [
    {'geometry_sha256': 'bad'}, {'object_id': ''}, {'patch_id': ''},
    {'triangles_local_m': np.zeros((1, 3, 3))},
    {'triangles_local_m': np.empty((0, 3, 3))},
    {'triangles_local_m': [[[0, 0, np.nan], [1, 0, 0], [0, 1, 0]]]},
    {'outward_normals_local': [[0, 0, 2]]},
    {'outward_normals_local': [[1, 0, 0]]},
])
def test_bad_patch_geometry_or_identity_fails_closed(kwargs):
    with pytest.raises(ValueError):
        patch(**kwargs)


@pytest.mark.parametrize('transform', [
    np.diag([-1, 1, 1, 1]), np.diag([2, 1, 1, 1]),
    np.eye(3), np.full((4, 4), np.nan),
])
def test_reflection_scale_and_bad_pose_are_rejected(transform):
    with pytest.raises(ValueError):
        patch().closest_world([[0, 0, 0]], transform)


@pytest.mark.parametrize('bindings', [(), (('thumb', 'face'),),
    (('thumb', 'face'), ('thumb', 'face')),
    (('thumb', 'face'), ('ring', 'face')),
    (('thumb', ''), ('index', 'face'))])
def test_bad_contact_mode_is_rejected(bindings):
    with pytest.raises(ValueError):
        RobotContactMode('mode', 'right', 'object', SHA, bindings)


@pytest.mark.parametrize('patches', [
    {}, {'face': patch(object_id='other')},
    {'face': patch(geometry_sha256='b'*64)}, {'face': patch(patch_id='other')},
])
def test_mode_cannot_reuse_stale_asset_or_wrong_object_patch(patches):
    mode = RobotContactMode('mode', 'right', 'object', SHA, (('thumb', 'face'), ('middle', 'face')))
    with pytest.raises(ValueError):
        mode.validate_patch_bindings(patches)
