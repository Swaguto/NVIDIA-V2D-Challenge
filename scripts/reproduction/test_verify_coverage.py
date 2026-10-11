"""Meaningful failure controls for the frame-level reproduction contract."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from verify_coverage import sha, verify


class CoverageContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root/'inputs'; self.inputs.mkdir()
        self.raw = self.root/'raw'; self.raw.mkdir()
        (self.raw/'video.mp4').write_bytes(b'test video identity')
        self.reference = self.root/'reference.json'
        self.bundle = self.root/'bundle.npz'; self.robot = self.root/'robot.npz'
        self.frames = np.arange(3, dtype=np.int64)
        self.mask = np.array([[True, False], [False, True], [True, True]])
        metadata = {'episode':12,'frames':[0,1,2], 'hand_frames':{'left':[0,2],'right':[1,2]}}
        (self.inputs/'inputs.json').write_text(json.dumps(metadata))
        for side, frames in metadata['hand_frames'].items():
            folder = self.inputs/'hands'/side/'poses'; folder.mkdir(parents=True)
            for frame in frames:(folder/f'{frame:06d}.json').write_text('{}')
        lock = {p.relative_to(self.inputs).as_posix():sha(p) for p in self.inputs.rglob('*') if p.is_file()}
        (self.inputs/'input-files.json').write_text(json.dumps(lock))
        contract = dict(metadata, schema='v2d.coverage-reference.v1',
                        input_lock_sha256=sha(self.inputs/'input-files.json'),
                        raw_files={'video.mp4':sha(self.raw/'video.mp4')})
        self.reference.write_text(json.dumps(contract))
        self.write_arrays(self.mask)

    def write_arrays(self, mask, nan_valid=False):
        joints = np.full((3,2,21,3), np.nan); joints[mask] = 0.0
        if nan_valid:joints[0,0,0,0] = np.nan
        qpos = np.full((3,2,14), np.nan); qpos[mask] = 0.0
        np.savez(self.bundle, frame_indices=self.frames,timestamps_s=self.frames/20,
                 hand_valid=mask, hand_directly_observed=mask,hand_joints_world=joints)
        np.savez(self.robot, frame_indices=self.frames,timestamps_s=self.frames/20,
                 source_hand_valid=mask,robot_valid=mask,robot_qpos=qpos)

    def run_check(self):
        return verify(self.reference,self.inputs,self.raw,self.bundle,self.robot)

    def test_reference_matches(self):
        self.assertEqual(self.run_check()['status'], 'coverage_matches')

    def test_equal_counts_on_wrong_frames_fail(self):
        changed = self.mask.copy(); changed[0,0] = False; changed[1,0] = True
        self.write_arrays(changed)
        with self.assertRaisesRegex(ValueError,'coverage differs'):self.run_check()

    def test_changed_source_video_fails(self):
        (self.raw/'video.mp4').write_bytes(b'different video')
        with self.assertRaisesRegex(ValueError,'Changed input'):self.run_check()

    def test_changed_input_and_rewritten_lock_fail(self):
        (self.inputs/'hands/left/poses/000000.json').write_text('{"different":true}')
        lock = self.inputs/'input-files.json'
        entries = json.loads(lock.read_text())
        entries['hands/left/poses/000000.json'] = sha(self.inputs/'hands/left/poses/000000.json')
        lock.write_text(json.dumps(entries))
        with self.assertRaisesRegex(ValueError,'Input lock differs'):self.run_check()

    def test_claimed_valid_but_nan_fails(self):
        self.write_arrays(self.mask,nan_valid=True)
        with self.assertRaisesRegex(ValueError,'finite/unknown'):self.run_check()

    def test_unlisted_extra_pose_fails(self):
        (self.inputs/'hands/left/poses/000001.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'input pose frame IDs'):self.run_check()


if __name__ == '__main__':unittest.main()
