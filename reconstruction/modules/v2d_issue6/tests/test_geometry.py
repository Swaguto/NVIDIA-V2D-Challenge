import numpy as np
import pytest
import trimesh

from issue6.geometry import (
    alignment_offset,
    bridge_contacts,
    camera,
    decode_depth,
    hand_geometry,
    reproject_depth,
    rigid,
    surface_contacts,
    transform,
)


def cam(w=4, h=3, focal=2):
    return dict(
        id="ego_cam_a",
        fx=focal,
        fy=focal,
        cx=0,
        cy=0,
        width=w,
        height=h,
        convention="rectified_pinhole",
    )


@pytest.mark.parametrize(
    "encoding,values,expected",
    [
        ("metres", [[1.0, 0, np.nan, np.inf]], [[1.0, np.nan, np.nan, np.nan]]),
        ("millimetres_u16", [[1000, 2000, 0]], [[1.0, 2.0, np.nan]]),
        ("inverse_u16", [[32767.5, 65535, 0]], [[1.0, np.nan, np.nan]]),
    ],
)
def test_depth_units_and_invalid(encoding, values, expected):
    np.testing.assert_allclose(decode_depth(values, encoding), expected, equal_nan=True)


def test_explicit_depth_mask():
    out = decode_depth([[1, 2]], "metres", [[True, False]])
    assert out[0, 0] == 1 and np.isnan(out[0, 1])
    with pytest.raises(ValueError):
        decode_depth([[1]], "metres", [[True, False]])


def test_depth_projection_identity_holes():
    z = np.array([[1, np.nan, 3, 4], [2, 2, 2, 2], [1, 1, 1, 1.0]])
    np.testing.assert_allclose(reproject_depth(z, cam(), cam(), np.eye(4)), z, equal_nan=True)


def test_depth_zbuffer_uses_nearest_not_last():
    src, target = cam(2, 1, 100), cam(1, 1, 1)
    out = reproject_depth(np.array([[1.0, 3.0]]), src, target, np.eye(4))
    assert out[0, 0] == 1


def test_depth_transform_direction_and_no_resize():
    t = np.eye(4)
    t[0, 3] = 1
    z = reproject_depth(np.ones((1, 1)), cam(1, 1, 1), cam(3, 1, 1), t)
    assert np.isnan(z[0, 0]) and z[0, 1] == 1
    with pytest.raises(ValueError, match="dimensions"):
        reproject_depth(np.ones((2, 2)), cam(), cam(), t)


def test_behind_camera_and_outside_image_are_discarded():
    t = np.eye(4)
    t[2, 3] = -2
    assert np.isnan(reproject_depth(np.ones((3, 4)), cam(), cam(), t)).all()


def test_transform_rejects_scale_reflection_and_distorted_camera():
    for value in (np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1])):
        with pytest.raises(ValueError):
            rigid(value)
    with pytest.raises(ValueError, match="rectified"):
        camera({**cam(), "convention": "raw_distorted"})


def test_left_hand_mirrors_and_joints_share_mesh_scale_pivot():
    vertices = np.array([[1, 0, 0], [3, 0, 0], [2, 3, 0.0]])
    joints = np.tile([1.0, 2.0, 3.0], (21, 1))
    v, j = hand_geometry(vertices, joints, False, 2, [0, 0, 4])
    center = np.array([-2, 1, 0])
    np.testing.assert_allclose(j[0], ([-1, 2, 3] - center) * 2 + center + [0, 0, 4])
    np.testing.assert_allclose(v.mean(0), center + [0, 0, 4])


def test_alignment_filters_invalid_depth_and_occlusion():
    rendered = np.ones((2, 3))
    measured = np.array([[2.0, np.nan, 99], [2, 0, np.inf]])
    occ = np.array([[False, False, True], [False, False, False]])
    offset, diag = alignment_offset(rendered, measured, np.ones((2, 3), bool), occ, [1, 0, 1], 2)
    np.testing.assert_allclose(offset, [1, 0, 1])
    assert diag["n_pixels"] == 2 and diag["metric_valid"]
    _, diag = alignment_offset(rendered, measured, np.ones((2, 3), bool), occ, [1, 0, 1], 3)
    assert not diag["metric_valid"]


def test_surface_distance_is_to_triangle_not_center_and_transform_is_inverted():
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    t = np.eye(4)
    t[:3, 3] = [3, 0, 5]
    pts = transform([[1.01, 0, 0], [0, 0, 0]], t)
    nearest, distance, normals, inside = surface_contacts(pts, mesh, t)
    np.testing.assert_allclose(distance, [0.01, 1], atol=1e-7)
    np.testing.assert_allclose(nearest[0], [1, 0, 0])
    np.testing.assert_allclose(normals[0], [1, 0, 0])
    assert inside == [False, True]


def test_open_mesh_has_unknown_penetration():
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    *_, inside = surface_contacts([[0.2, 0.2, 0.01]], mesh, np.eye(4))
    assert inside == [None]


def test_contact_threshold_is_strictly_below_two_cm():
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    _, distances, _, _ = surface_contacts([[0.2, 0.2, 0.019], [0.2, 0.2, 0.02], [0.2, 0.2, 0.021]], mesh, np.eye(4))
    assert (distances < 0.02).tolist() == [True, False, False]


@pytest.mark.parametrize(
    "raw,ids,maximum,expected",
    [
        ([True, False, True], [1, 2, 3], 1, [True, True, True]),
        ([True, None, True], [1, 2, 3], 15, [True, None, True]),
        ([True, False, True], [1, 2, 4], 15, [True, False, True]),
        ([True, False, False, True], [1, 2, 3, 4], 1, [True, False, False, True]),
        ([False, True, False], [1, 2, 3], 15, [False, True, False]),
        ([True] + [False] * 15 + [True], list(range(17)), 15, [True] * 17),
        (
            [True] + [False] * 16 + [True],
            list(range(18)),
            15,
            [True] + [False] * 16 + [True],
        ),
    ],
)
def test_contact_gap_boundaries(raw, ids, maximum, expected):
    assert bridge_contacts(raw, ids, maximum) == expected
