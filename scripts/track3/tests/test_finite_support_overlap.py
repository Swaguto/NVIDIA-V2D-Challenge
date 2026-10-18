import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from measured_pose_refinement import finite_support_overlap, support_constraints


def surface():
    return np.array([0.0, 0.0, 1.0, 0.0]), np.array(
        [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]]
    )


def test_triangle_interior_is_checked_when_all_vertices_overhang():
    plane, footprint = surface()
    v = np.array([[-3, -3, -0.01], [3, -3, -0.01], [0, 3, -0.01]])
    assert support_constraints(v, np.eye(4), plane, footprint) is None
    r = finite_support_overlap(v, np.array([[0, 1, 2]]), np.eye(4), plane, footprint)
    assert r["minimum_height_m"] == pytest.approx(-0.01)
    assert r["maximum_height_m"] == pytest.approx(-0.01)
    assert np.max(np.abs(r["witness_body"][:2])) <= 0.5 + 1e-12


def test_low_overhanging_vertex_does_not_force_infinite_plane_clearance():
    plane = np.array([0.0, 0.0, 1.0, 0.0])
    footprint = np.array([[0, -0.5, 0], [1, -0.5, 0], [1, 0.5, 0], [0, 0.5, 0]])
    v = np.array([[-2, 0, -0.1], [0.1, -0.2, 0.02], [0.1, 0.2, 0.02]])
    r = finite_support_overlap(v, np.array([[0, 1, 2]]), np.eye(4), plane, footprint)
    assert r["minimum_height_m"] > 0.01
    assert r["maximum_height_m"] == pytest.approx(0.02)


def test_disjoint_body_has_no_support_overlap():
    plane, footprint = surface()
    v = np.array([[2, 0, -0.1], [3, 0, -0.1], [2, 1, -0.1]])
    assert (
        finite_support_overlap(v, np.array([[0, 1, 2]]), np.eye(4), plane, footprint)
        is None
    )


def test_rigid_coordinate_change_preserves_finite_contact_heights():
    plane, footprint = surface()
    v = np.array([[-3, -3, -0.01], [3, -3, 0.03], [0, 3, 0.02]])
    faces = np.array([[0, 1, 2]])
    a = finite_support_overlap(v, faces, np.eye(4), plane, footprint)
    g = np.eye(4)
    g[:3, :3] = Rotation.from_euler("xyz", [0.4, -0.3, 0.8]).as_matrix()
    g[:3, 3] = [0.3, -0.4, 0.7]
    moved = footprint @ g[:3, :3].T + g[:3, 3]
    normal = g[:3, :3] @ plane[:3]
    p = np.r_[normal, plane[3] - normal @ g[:3, 3]]
    b = finite_support_overlap(v, faces, g, p, moved)
    assert a["minimum_height_m"] == pytest.approx(b["minimum_height_m"], abs=1e-12)
    assert a["maximum_height_m"] == pytest.approx(b["maximum_height_m"], abs=1e-12)


@pytest.mark.parametrize("kind", ["bad_faces", "nonrigid", "off_plane_footprint"])
def test_invalid_geometry_is_rejected(kind):
    plane, footprint = surface()
    v = np.array([[-3, -3, -0.01], [3, -3, -0.01], [0, 3, -0.01]])
    pose, faces = np.eye(4), np.array([[0, 1, 2]])
    if kind == "bad_faces":
        faces[0, 0] = -1
    elif kind == "nonrigid":
        pose[0, 0] = 2
    else:
        footprint[0, 2] = 0.02
    with pytest.raises(ValueError, match="finite body triangles"):
        finite_support_overlap(v, faces, pose, plane, footprint)
