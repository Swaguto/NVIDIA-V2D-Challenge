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
from grip_force import open_pregrasp, point_jacobian


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


def test_grip_jacobian_transpose_matches_virtual_work_and_world_rotation(tmp_path):
    p = tmp_path / "hand.urdf"
    p.write_text("""<robot name="test"><link name="root"/><link name="pad"/>
      <joint name="hinge" type="revolute"><parent link="root"/><child link="pad"/>
      <axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint></robot>""")
    kin = URDFKinematics(p)
    root = np.array([0, 0, 0, 0, 0, 0, 1])
    jac = point_jacobian(kin, root, {"hinge": 0.0}, "pad", np.array([0.05, 0, 0]))
    np.testing.assert_allclose(jac[:, 0], [0, 0.05, 0], atol=1e-10)
    force = np.array([0, 6.0, 0])
    assert (jac.T @ force)[0] == pytest.approx(0.3)
    rot = Rotation.from_euler("xyz", [0.2, -0.5, 0.6])
    moved = np.r_[[2, -1, 3], rot.as_quat()]
    rotated = point_jacobian(kin, moved, {"hinge": 0.0}, "pad", np.array([0.05, 0, 0]))
    np.testing.assert_allclose(rotated.T @ rot.apply(force), jac.T @ force, atol=1e-9)


def test_pregrasp_opens_opposed_pads_without_moving_the_wrist(tmp_path):
    path = tmp_path / "pinch.urdf"
    path.write_text("""<robot name="pinch"><link name="root"/><link name="a"/><link name="b"/>
      <joint name="ja" type="revolute"><parent link="root"/><child link="a"/>
      <axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint>
      <joint name="jb" type="revolute"><parent link="root"/><child link="b"/>
      <axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint></robot>""")
    kin = URDFKinematics(path)
    anchors = [
        dict(link=link, point_local_m=[0.05, 0, 0], direction_world=[0, sign, 0])
        for link, sign in [("a", 1), ("b", -1)]
    ]
    opened = open_pregrasp(
        kin, [0, 0, 0, 0, 0, 0, 1], {"ja": 0, "jb": 0}, anchors, 0.004
    )
    assert opened["ja"] < 0 < opened["jb"]
    assert abs(0.05 * np.sin(opened["ja"])) == pytest.approx(0.004, abs=1e-5)
    # A pad corner needs more opening than its anchor point. Require the whole
    # pad to clear a 2 mm thick object by 2 mm, not just its point target.
    pad = cube(0.002) + [0.05, 0, 0]
    terms = [
        (link, pad, np.array([[0, 1, 0]]), np.array([-0.001]), np.array([0.001]))
        for link in ["a", "b"]
    ]
    opened = open_pregrasp(
        kin, [0, 0, 0, 0, 0, 0, 1], {"ja": 0, "jb": 0}, anchors, 0.004, terms
    )
    poses = kin.poses([0, 0, 0, 0, 0, 0, 1], opened)
    for link in ["a", "b"]:
        world = pad @ poses[link][:3, :3].T + poses[link][:3, 3]
        gap = max(world[:, 1].min() - 0.001, -0.001 - world[:, 1].max())
        assert gap >= 0.0018
