"""Reject missing state coverage, failed controls, and unstable wrist trajectories."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_wrist_run import audit


def fixture(root, *, offset=0., controls=True, missing=False):
    root.mkdir()
    pose = [0, 0, 0, 0, 0, 0, 1]
    states = {'hands': [{'side': side, 'step': step,
        'actual_root_pose_stage_xyzw': [offset, *pose[1:]],
        'target_root_pose_stage_xyzw': pose} for side in ['left', 'right'] for step in range(3)],
        'rendered_source_frames': [{'source_frame': step, 'sim_time_s': step * .05} for step in range(3)]}
    if missing:
        states['hands'].pop()
    (root / 'states.json').write_text(json.dumps(states))
    # This test checks report/state integrity, not a mocked video decode claim.
    (root / 'supported-task.mp4').write_bytes(b'test media hash input')
    report = {'state_sha256': hashlib.sha256((root / 'states.json').read_bytes()).hexdigest(),
        'video_sha256': hashlib.sha256((root / 'supported-task.mp4').read_bytes()).hexdigest(),
        'simulation_steps': 2, 'source_frames': 3, 'inputs_manifest_sha256': 'a' * 64,
        'gravity_and_contact_controls_pass': controls, 'object_teleports_after_initialization': 0,
        'external_object_forces': False}
    (root / 'report.json').write_text(json.dumps(report))
    return root


def test_zero_error_is_controller_only_pass(tmp_path):
    report = audit(fixture(tmp_path / 'run'))
    assert report['controller_tracking_pass']
    assert not report['physical_task_accepted']
    assert not report['learned_policy_evaluated']


@pytest.mark.parametrize('options', [{'offset': .03}, {'controls': False}])
def test_motion_failure_and_bad_physics_controls_reject_acceptance(tmp_path, options):
    assert not audit(fixture(tmp_path / 'run', **options))['controller_tracking_pass']


def test_missing_physics_step_rejected(tmp_path):
    with pytest.raises(ValueError, match='physics records'):
        audit(fixture(tmp_path / 'run', missing=True))
