"""Occlusion and missing coverage must not become anatomical joint depth targets."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from source_hand_depth import compare_estimated_surface, front_surface_depth


def layers():
    vertices = np.array([[-.02, -.02, .3], [.02, -.02, .3], [0, .02, .3], [0, 0, .35], [-.01, -.01, .35], [.01, -.01, .35]])
    return vertices, np.array([[0, 1, 2], [3, 4, 5]]), np.array([[100., 0, 20], [0, 100., 20], [0, 0, 1]])


def test_exact_occluded_tip_geometry_matches_visible_skin_without_a_joint_offset():
    v, f, k = layers()
    result = compare_estimated_surface(v, f, [[20, 20]], k, [.3], [True], evidence_kind="estimated_visible_surface_depth")
    np.testing.assert_allclose(result.residual_m, 0, atol=1e-12)
    assert result.model_hit.all()
    # The old offset rejects this exact scene at its unchanged 10 mm screen.
    assert abs(v[3, 2] - (.3 + .005)) > .01


def test_missing_expected_ray_is_infinite_and_unknown_source_remains_unknown():
    v, f, k = layers()
    result = compare_estimated_surface(v, f, [[20, 20], [200, 200], [np.nan, np.nan]], k, [.3, .4, np.nan], [True, True, False], evidence_kind="estimated_visible_surface_depth")
    np.testing.assert_array_equal(result.model_hit, [True, False, False])
    assert np.isinf(result.residual_m[1]) and np.isnan(result.residual_m[2])
    assert not np.all(result.model_hit[result.observed_valid])


def test_depth_is_z_not_range_and_face_winding_or_chunk_boundaries_do_not_change_it():
    vertices = np.array([[-5., -5., 2.], [5., -5., 2.], [0., 5., 2.]])
    pixels = np.tile([.5, .2], (129, 1))
    a = front_surface_depth(vertices, [[0, 1, 2]], pixels, np.eye(3))
    b = front_surface_depth(vertices, [[2, 1, 0]], pixels[:1], np.eye(3))
    np.testing.assert_allclose(a, 2, atol=1e-12)
    np.testing.assert_array_equal(a[:1], b)


def test_behind_camera_and_empty_mesh_are_missing():
    v, f, k = layers()
    v[:, 2] *= -1
    assert np.isinf(front_surface_depth(v, f, [[20, 20]], k)).all()
    assert np.isinf(front_surface_depth(v, np.empty((0, 3), int), [[20, 20]], k)).all()


@pytest.mark.parametrize("failure", ["joint_target", "invalid_depth", "implicit_validity", "negative_focal", "float_faces"])
def test_rejects_wrong_observation_kind_or_invalid_contract(failure):
    v, f, k = layers()
    kind, depth, valid = "estimated_visible_surface_depth", [.3], [True]
    if failure == "joint_target": kind = "joint_center_depth"
    elif failure == "invalid_depth": depth = [np.nan]
    elif failure == "implicit_validity": valid = [1]
    elif failure == "negative_focal": k[0, 0] = -1
    elif failure == "float_faces": f = f.astype(float)
    with pytest.raises(ValueError):
        compare_estimated_surface(v, f, [[20, 20]], k, depth, valid, evidence_kind=kind)
