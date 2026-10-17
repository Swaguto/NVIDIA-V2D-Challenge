from pathlib import Path
import sys

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("rtree")
sys.path.insert(0, str(Path(__file__).parents[1]))
from filter_contacts_for_collision import exposed_projection


def test_internal_union_faces_and_deep_points_cannot_be_contact_targets():
    first = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    second = first.copy()
    second.apply_translation([0.001, 0, 0])
    points = np.array([[0.1005, 0, 0.01], [0.095, 0, 0.01]])
    force = np.array([[-1, 0, 0], [-1, 0, 0]])
    projected, normals, kept, distance, _ = exposed_projection(points, force, [first, second])
    assert kept.tolist() == [True, False]
    # x=0.100 is first's face buried inside second. Exposed boundary is0.101.
    np.testing.assert_allclose(projected[0], [0.101, 0, 0.01], atol=1e-9)
    np.testing.assert_allclose(normals[0], [-1, 0, 0])
    assert distance[0] == pytest.approx(0.0005)


def test_opposite_source_normal_is_rejected_instead_of_flipped():
    box = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    points = np.array([[0.101, 0, 0.01]])
    _, _, kept, _, _ = exposed_projection(points, np.array([[1, 0, 0]]), [box])
    assert kept.tolist() == [False]


def test_reference_uncertainty_correction_still_targets_exposed_zero_depth_surface():
    box = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    points = np.array([[0.09, 0, 0.01]])
    force = np.array([[-1, 0, 0]])
    _, _, narrow, _, _ = exposed_projection(points, force, [box], radius=0.002)
    projected, normals, kept, distance, depth = exposed_projection(points, force, [box], radius=0.012)
    assert not narrow[0] and kept[0]
    assert depth[0] == pytest.approx(0.01)
    assert distance[0] == pytest.approx(0.01)
    np.testing.assert_allclose(projected[0], [0.1, 0, 0.01], atol=1e-9)
    np.testing.assert_allclose(normals[0], [-1, 0, 0])


def test_abutting_internal_interface_is_not_exposed_contact_surface():
    first = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    second = first.copy()
    second.apply_translation([0.2, 0, 0])
    _, _, kept, _, _ = exposed_projection(np.array([[0.099, 0, 0.01]]), np.array([[-1, 0, 0]]), [first, second])
    assert not kept[0]
