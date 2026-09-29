# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for pose metrics and the CLI report/output boundary."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from v2d.world_calib.vm import fit_object_pose as fit


def _transform(degrees=0.0, translation=(0.0, 0.0, 0.0)):
    """Build an independent z-rotation fixture in metres."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[:3, 3] = translation
    return result


@pytest.mark.parametrize('angle', [0, 10, 90, 180])
def test_pose_error_units_and_world_frame_invariance(angle):
    """Known relative motion has the same error in a rotated world frame."""
    gt = _transform(37, (2, -3, 1))
    estimate = gt @ _transform(angle, (0.03, 0.04, 0))
    assert fit._pose_error(estimate, gt) == pytest.approx((angle, 0.05), abs=1e-6)


def test_chamfer_is_symmetric_unsquared_and_in_mm():
    """Unequal point sets distinguish symmetric Chamfer from directed distance."""
    a = np.array([[0, 0, 0], [0.004, 0, 0]])
    b = np.array([[0, 0, 0]])
    assert fit._chamfer(a, b) == pytest.approx(1.0)
    assert fit._chamfer(b, a) == pytest.approx(1.0)
    assert fit._chamfer(a, a) == 0
    assert np.isnan(fit._chamfer(a, np.empty((0, 3))))


def test_gt_aggregation_thresholds_and_coverage():
    """Pass@5cm/10deg includes boundaries and coverage counts missing solves."""
    score = fit._summarize_gt([0, 10, 10.01, 0], [0, 0.05, 0, 0.0501],
                              [0, 20, 30, 40], 8)
    assert score == pytest.approx({
        'n_solved_gt': 4, 'n_visible': 8, 'coverage_visible': 0.5,
        'median_rot_deg': 5, 'median_trans_m': 0.025,
        'median_chamfer_mm': 25, 'p90_rot_deg': 10.007,
        'p90_trans_m': 0.05007, 'pass_5cm_10deg': 0.5,
    })


@pytest.mark.parametrize('visible', [0, 4])
def test_gt_aggregation_without_solutions(visible):
    """An unscored episode retains its denominator without inventing errors."""
    score = fit._summarize_gt([], [], [], visible)
    assert score['n_visible'] == visible
    assert score['n_solved_gt'] == 0
    assert score['coverage_visible'] == (0.0 if visible else None)
    assert score['median_rot_deg'] is None
    assert score['median_trans_m'] is None
    assert score['median_chamfer_mm'] is None
    assert score['pass_5cm_10deg'] is None
    json.dumps(score, allow_nan=False)


@pytest.mark.parametrize('modes', [[], ['--score-gt', '--no-gt']])
def test_cli_requires_exactly_one_mode(monkeypatch, modes):
    """Invalid modes fail before touching missing dataset paths."""
    monkeypatch.setattr(sys, 'argv', ['fit_object_pose', '--data-root', '/missing',
                                    '--work', '/missing', *modes])
    with pytest.raises(SystemExit) as error:
        fit.main()
    assert error.value.code == 2


@pytest.fixture
def cli_case(tmp_path, monkeypatch):
    """Supply synthetic W1/GT inputs and intercept the expensive ICP solve."""
    data, work = tmp_path / 'public', tmp_path / 'work'
    (data / 'meta').mkdir(parents=True)
    (data / 'meta/camera_calibration.json').write_text('{}')
    cameras = [_transform(30, (0.3, -0.2, 0.5)),
               _transform(60, (-0.1, 0.4, 0.6))]
    world = _transform(45, (0.1, 0.2, 0.3))
    rig = _transform(0, (0.075, 0, 0))
    cam_root = work / 'ego_cam_poses/target/world_to_cam/left'
    cam_root.mkdir(parents=True)
    for frame, camera in enumerate(cameras):
        (cam_root / f'{frame:06d}.json').write_text(json.dumps(fit.mat_to_js(camera)))
    # Target is second in parquet, first in the requested object list.
    poses = np.zeros((3, 2, 7))
    poses[:, :, 6] = 1
    poses[:, 1, :3] = world[:3, 3]
    poses[:, 1, 5:] = [np.sin(np.pi / 8), np.cos(np.pi / 8)]
    ref = SimpleNamespace(object_names=('other', 'target'), steps=3,
                          visible=np.array([[False, True]] * 3), pose_xyzw=poses)
    loader = SimpleNamespace(load_reference=lambda *args: ref,
                             object_mesh_dir=lambda names, root: {n: n for n in names})
    mesh = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.02, 0.03]])
    monkeypatch.setattr(fit, 'load_reference_loader', lambda path: loader)
    monkeypatch.setattr(fit, 'ego_rig_matrix', lambda value: ('left', 'right', rig))
    monkeypatch.setattr(fit, 'mesh_points_and_normals', lambda *args: (mesh, mesh))
    monkeypatch.setattr(fit, 'frame_cloud', lambda *args, **kwargs: mesh + [0, 0, 1])
    initializations = []

    def solve(seed, *args):
        """Return known camera-space fits and capture propagation seeds."""
        initializations.append(seed.copy())
        frame = 0 if len(initializations) == 1 else 1
        return cameras[frame] @ world, {
            'cam_left': {'count': 3, 'inlier': 1.0},
            'cam_right': {'count': 0, 'inlier': 0.0},
        }

    monkeypatch.setattr(fit, 'solve_frame', solve)
    argv = ['fit_object_pose', '--data-root', str(data), '--work', str(work),
            '--episode', '2', '--objects', 'target', '--no-smooth']
    return SimpleNamespace(data=data, work=work, ref=ref, loader=loader,
                           cameras=cameras, world=world, rig=rig,
                           initializations=initializations, argv=argv)


@pytest.mark.parametrize('objects', ['target', 'target,other'])
def test_cli_scores_selected_objects_and_writes_world_poses(cli_case, monkeypatch, objects):
    """Subset/reordered GT, quaternion conversion and propagation stay aligned."""
    case = cli_case
    argv = case.argv.copy()
    argv[argv.index('--objects') + 1] = objects
    monkeypatch.setattr(sys, 'argv', argv + ['--score-gt'])
    assert fit.main() == 0
    root = case.work / 'ego_object_poses/e002'
    report = json.loads((root / 'report.json').read_text())
    target = report['objects']['target']
    assert target['solved'] == target['passed'] == 2
    assert target['gt_score']['n_visible'] == 3
    assert target['gt_score']['coverage_visible'] == 0.667
    assert target['gt_score']['median_rot_deg'] == pytest.approx(0, abs=1e-5)
    assert target['gt_score']['median_trans_m'] == pytest.approx(0, abs=1e-10)
    assert target['gt_score']['median_chamfer_mm'] == pytest.approx(0, abs=1e-5)
    assert target['gt_score']['pass_5cm_10deg'] == 1
    assert all(frame['gt_pass'] for frame in target['framedetails'].values())
    np.testing.assert_allclose(case.initializations[1], case.cameras[1] @ case.world,
                               atol=1e-12)
    np.testing.assert_allclose(case.initializations[2][:3, :3],
                               (case.cameras[1] @ case.world)[:3, :3], atol=1e-12)
    for frame, camera in enumerate(case.cameras):
        output = json.loads((root / f'target/object_to_world/{frame:06d}.json').read_text())
        np.testing.assert_allclose(fit.js_to_mat(output), case.world, atol=1e-12)
        output = json.loads((root / f'target/object_to_cam/right/{frame:06d}.json').read_text())
        np.testing.assert_allclose(fit.js_to_mat(output), case.rig @ camera @ case.world,
                                   atol=1e-12)


def test_cli_no_gt_uses_mask_frames(cli_case, monkeypatch):
    """No-GT mode never loads parquets and still exports poses and a report."""
    case = cli_case

    def forbidden(*args):
        """Fail if eval mode tries to read public ground truth."""
        pytest.fail('no-GT mode loaded ground truth')

    case.loader.load_reference = forbidden
    masks = case.work / 'masks/left/target/0'
    masks.mkdir(parents=True)
    for frame in range(2):
        (masks / f'{frame:06d}.png').touch()
    monkeypatch.setattr(sys, 'argv', case.argv + ['--no-gt'])
    assert fit.main() == 0
    report = json.loads((case.work / 'ego_object_poses/e002/report.json').read_text())
    target = report['objects']['target']
    assert target['frames'] == target['solved'] == 2
    assert target['gt_score'] == {}
    assert all('gt_pass' not in frame for frame in target['framedetails'].values())


def test_cli_frame_and_aggregate_gt_pass_agree(cli_case, monkeypatch):
    """Chamfer is reported separately from the 5cm/10deg success threshold."""
    monkeypatch.setattr(fit, '_pose_error', lambda *args: (10.0, 0.05))
    monkeypatch.setattr(fit, '_chamfer', lambda *args: 20.0)
    monkeypatch.setattr(sys, 'argv', cli_case.argv + ['--score-gt'])
    assert fit.main() == 0
    report = json.loads((cli_case.work / 'ego_object_poses/e002/report.json').read_text())
    target = report['objects']['target']
    assert all(frame['gt_pass'] for frame in target['framedetails'].values())
    assert target['gt_score']['pass_5cm_10deg'] == 1
    assert target['gt_score']['median_chamfer_mm'] == 20


def test_cli_reports_visible_frames_when_none_solve(cli_case, monkeypatch):
    """Missing depth/masks cannot erase an episode from coverage reporting."""
    monkeypatch.setattr(fit, 'frame_cloud', lambda *args, **kwargs: np.empty((0, 3)))
    monkeypatch.setattr(sys, 'argv', cli_case.argv + ['--score-gt'])
    assert fit.main() == 0
    report = json.loads((cli_case.work / 'ego_object_poses/e002/report.json').read_text())
    target = report['objects']['target']
    assert target['solved'] == 0
    assert target['framedetails'] == {}
    assert target['gt_score']['n_visible'] == 3
    assert target['gt_score']['coverage_visible'] == 0
    assert target['gt_score']['median_rot_deg'] is None
