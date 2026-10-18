"""Diagnostics must preserve unknowns, coordinate frames and named joint order."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_reference_geometry import body_coordinates, nearest, robot_tips, summary


def test_body_frame_distance_is_invariant_to_world_rigid_transform():
    local = np.array([[[0.1, 0.2, 0.3], [0.2, -0.1, 0.4]]])
    pose = np.eye(4)[None]
    pose[0, :3, :3] = Rotation.from_euler("xyz", [0.3, 1.2, -0.5]).as_matrix()
    pose[0, :3, 3] = [0.1, -0.3, 0.6]
    world = local @ pose[0, :3, :3].T + pose[0, :3, 3]
    np.testing.assert_allclose(body_coordinates(world, pose), local, atol=1e-15)


def test_absent_hand_stays_unknown_and_empty_summary_is_not_zero_error():
    tree = cKDTree([[0, 0, 0]])
    result = nearest(tree, [[np.nan, np.nan, np.nan], [1, 0, 0]])
    assert np.isnan(result[0]) and result[1] == 1
    assert summary(result[:1]) == dict(samples=0, median_m=None, p95_m=None)


def test_robot_diagnostic_binds_declared_joint_order():
    class Kinematics:
        names = ["thumb", "index", "middle"]

        def poses(self, root, positions):
            result = {}
            for digit in self.names:
                pose = np.eye(4)
                pose[0, 3] = positions[digit]
                result[f"right_{digit}_tip"] = pose
            return result

    q = np.array([[0, 0, 0, 0, 0, 0, 1, 11, 22, 33]])
    result = robot_tips(q, Kinematics(), ["middle", "thumb", "index"], "right")
    np.testing.assert_array_equal(result[0, :, 0], [22, 33, 11])
    with pytest.raises(ValueError, match="joint names"):
        robot_tips(q, Kinematics(), ["thumb", "index", "index"], "right")
