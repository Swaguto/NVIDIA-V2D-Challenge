from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temporal_pose_path import MotionLimits, motion_summary, select_path


def pose(x=0., angle=0.):
    p = np.eye(4)
    p[0, 3] = x
    p[:3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
    return p


def test_rejects_greedy_jump_and_chooses_feasible_lower_fit_alternative():
    pools = [np.array([pose(0)]), np.array([pose(.1), pose(2)]), np.array([pose(.2)])]
    selected, report = select_path(pools, [[0], [.2, 0], [0]], [0, 1, 2], MotionLimits(.5, 1, .5, 1))
    assert selected == [0, 0, 0]
    assert report['feasible'] is True
    assert report['source_fit_validation_performed'] is False
    assert report['physical_task_accepted'] is False


def test_acceleration_constraint_uses_actual_nonuniform_timestamps():
    pools = [np.array([pose(0)]), np.array([pose(.1)]), np.array([pose(.2), pose(.3)])]
    selected, report = select_path(pools, [[0], [0], [0, .1]], [0, 1, 3], MotionLimits(.2, 1, .01, 1))
    assert selected == [0, 0, 1]
    assert report['selected_motion']['maximum_linear_acceleration_m_s2'] < 1e-12


def test_rotation_limits_and_world_angular_acceleration_are_enforced():
    pools = [np.array([pose(angle=0)]), np.array([pose(angle=.1)]), np.array([pose(angle=-.1), pose(angle=.2)])]
    selected, report = select_path(pools, [[0], [0], [0, .1]], [0, 1, 2], MotionLimits(1, .25, 1, .05))
    assert selected == [0, 0, 1]
    assert report['selected_motion']['maximum_angular_acceleration_rad_s2'] < 1e-12


def test_missing_frame_is_rejected_without_interpolation():
    selected, report = select_path([np.array([pose()]), []], [[0], []], [0, 1], MotionLimits(1, 1, 1, 1))
    assert selected is None
    assert report['failed_frame'] == 1
    assert report['missing_observations_filled'] is False


def test_infeasible_path_and_invalid_rotation_remain_failures():
    selected, report = select_path([np.array([pose()]), np.array([pose(2)])], [[0], [0]], [0, 1], MotionLimits(1, 1, 1, 1))
    assert selected is None and report['feasible'] is False
    bad = pose(); bad[0, 0] = 2
    with pytest.raises(ValueError, match='proper rigid'):
        select_path([np.array([bad])], [[0]], [0], MotionLimits(1, 1, 1, 1))
    with pytest.raises(ValueError, match='ordered source clock'):
        select_path([np.array([pose()]), np.array([pose()])], [[0], [0]], [1, 1], MotionLimits(1, 1, 1, 1))
