"""Analytic solids and URDF controls for grasp diagnostics."""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cooked_grasp_audit import convex_distance, convex_escape, world_convexes
from grasp_preload import URDFKinematics
from antipodal_seed import antipodal_pairs, exposed_faces


def cube(half=0.01):
    return np.array(list(product([-half, half], repeat=3)))


def test_distance_and_escape_for_analytic_solids():
    a, b = cube(), cube() + [0.023, 0, 0]
    distance, witness = convex_distance(a, b)
    assert distance == pytest.approx(0.003, abs=1e-8)
    assert np.linalg.norm(witness[0] - witness[1]) == pytest.approx(distance)
    assert convex_escape(a, b)[0] == pytest.approx(-0.003)
    b -= [0.005, 0, 0]
    assert convex_distance(a, b)[0] < 1e-7
    depth, axis = convex_escape(a, b)
    assert depth == pytest.approx(0.002)
    np.testing.assert_allclose(axis, [-1, 0, 0], atol=1e-9)
    assert convex_escape(a + (depth + 1e-6) * axis, b)[0] < 0


def test_containment_and_rigid_transform_equivariance():
    a, b = cube(0.002), cube(0.01)
    assert convex_distance(a, b)[0] < 1e-8
    assert convex_escape(a, b)[0] == pytest.approx(0.012)
    a = cube() + [0.018, 0.002, 0.001]
    depth, axis = convex_escape(a, b)
    r = Rotation.from_euler("xyz", [0.31, -0.52, 0.11])
    shifted = [r.apply(v) + [6, -3, 2] for v in (a, b)]
    new_depth, new_axis = convex_escape(*shifted)
    assert new_depth == pytest.approx(depth, abs=1e-9)
    np.testing.assert_allclose(new_axis, r.apply(axis), atol=1e-9)


def test_cooked_row_matrix_then_body_transform():
    matrix = np.eye(4)
    matrix[3, :3] = [0.1, 0.2, 0.3]
    collider = {
        "mesh_to_body_matrix_row_convention": matrix.tolist(),
        "convexes": [{"vertices_mesh_local_m": cube().tolist()}],
    }
    r = Rotation.from_euler("z", np.pi / 2)
    pose = np.r_[[1, 2, 3], r.as_quat()]
    actual = world_convexes(collider, pose)[0]
    np.testing.assert_allclose(actual, r.apply(cube() + [0.1, 0.2, 0.3]) + [1, 2, 3])


def test_urdf_joint_axis_and_fixed_offsets(tmp_path):
    p = tmp_path / "hand.urdf"
    p.write_text("""<robot name="test"><link name="root"/><link name="finger"/>
      <link name="tip"/><joint name="hinge" type="revolute">
      <parent link="root"/><child link="finger"/><origin xyz="1 0 0"/>
      <axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint>
      <joint name="fixed" type="fixed"><parent link="finger"/><child link="tip"/>
      <origin xyz="1 0 0"/></joint></robot>""")
    poses = URDFKinematics(p).poses([0, 0, 0, 0, 0, 0, 1], {"hinge": np.pi / 2})
    np.testing.assert_allclose(poses["tip"][:3, 3], [1, 1, 0], atol=1e-9)


def test_antipodal_pair_requires_opposed_normals_and_friction_cone():
    points = np.array([[-0.01, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
    normals = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0]])
    pairs = antipodal_pairs(points, normals, points[:2])
    assert any((a, b) == (0, 1) for _, a, b in pairs)
    bad = normals.copy()
    bad[1] = [-1, 0, 0]
    assert not antipodal_pairs(points, bad, points[:2])
    _, outward = exposed_faces([cube()])
    assert len(outward) == 12
