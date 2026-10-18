import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from measured_pose_refinement import refine_translation, support_constraints


def surface():
    vertices = np.array(
        [
            [(x - 15) * 0.5 / 100 + 0.3, (y - 15) * 0.5 / 100, 0.5]
            for y in range(6, 25)
            for x in range(6, 25)
        ]
    )
    faces = []
    for y in range(18):
        for x in range(18):
            i = y * 19 + x
            faces.extend([[i, i + 1, i + 19], [i + 1, i + 20, i + 19]])
    pose = np.eye(4)
    pose[0, 3] = -0.3
    pose[2, 3] = 0.025
    depth = np.ones((32, 32)) * 0.5
    mask = np.zeros_like(depth, bool)
    mask[6:25, 6:25] = True
    K = np.array([[100.0, 0, 15], [0, 100.0, 15], [0, 0, 1.0]])
    return vertices, np.array(faces), pose, depth, np.ones_like(mask), mask, K


def test_measured_depth_corrects_translation_without_changing_raw_body():
    args = surface()
    original_vertices = args[0].copy()
    original_pose = args[2].copy()
    pose, report = refine_translation(*args)
    assert report["accepted"]
    assert pose[2, 3] == pytest.approx(0, abs=0.001)
    assert report["after"]["median_measured_vertex_distance_m"] < 0.001
    np.testing.assert_array_equal(pose[:3, :3], original_pose[:3, :3])
    np.testing.assert_array_equal(args[0], original_vertices)
    np.testing.assert_array_equal(args[2], original_pose)


def test_missing_depth_remains_unknown_and_does_not_fill_a_pose():
    args = list(surface())
    args[4] = np.zeros_like(args[4])
    pose, report = refine_translation(*args)
    np.testing.assert_array_equal(pose, args[2])
    assert not report["accepted"] and not report["missing_observations_filled"]


def test_finite_table_does_not_push_up_an_overhanging_or_off_table_body():
    v = np.array([[-0.02, -0.02, -0.01], [0.02, -0.02, -0.01], [0, 0.02, 0.01]])
    table = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]])
    A, b = support_constraints(v, np.eye(4), [0, 0, 1, 0], table)
    assert b[0] == pytest.approx(0.012)
    assert np.min(A @ np.array([0, 0, 0.012]) - b) >= -1e-9
    pose = np.eye(4)
    pose[0, 3] = 0.2
    assert support_constraints(v, pose, [0, 0, 1, 0], table) is None
    pose[0, 3] = 0.09
    assert support_constraints(v, pose, [0, 0, 1, 0], table) is None


def test_plane_cannot_override_contradictory_measured_depth():
    args = surface()
    # This plane asks to move the measured surface another 25 mm away from the
    # pixels. A physically plausible constraint cannot qualify a worse fit.
    footprint = np.array([[-1, -1, 0.55], [1, -1, 0.55], [1, 1, 0.55], [-1, 1, 0.55]])
    pose, report = refine_translation(
        *args, plane=[0, 0, 1, -0.55], footprint=footprint
    )
    assert report["support_constraint_applied"] and not report["accepted"]
    np.testing.assert_array_equal(pose, args[2])
