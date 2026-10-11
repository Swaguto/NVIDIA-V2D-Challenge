"""Contract tests for provenance and held-pose presentation."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from continuous_dex3 import build, digest, hold_rows, verify
from rebuild_motion import main as rebuild
from types import SimpleNamespace


class ContinuityTests(unittest.TestCase):
    def test_changed_refined_scene_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); trial=root/'refinement'; trial.mkdir()
            (trial/'scene.npz').write_bytes(b'changed scene')
            (trial/'run-receipt.json').write_text(json.dumps(dict(status='completed',scene_sha256='0'*64)))
            with self.assertRaisesRegex(ValueError,'Refined scene differs'):
                rebuild(SimpleNamespace(kit=root/'kit',refinement=trial,out=root/'out',inputs=None))

    def test_leading_internal_and_trailing_hold(self):
        valid = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 0]], dtype=bool)
        np.testing.assert_array_equal(hold_rows(valid), [[1, 0], [1, 0], [1, 2], [3, 2], [3, 2]])

    def test_no_source_pose_fails(self):
        with self.assertRaises(ValueError):
            hold_rows(np.zeros((4, 2), dtype=bool))

    def test_observed_targets_and_fk_preserved_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'retarget'; source.mkdir()
            valid = np.array([[0, 1], [1, 0], [1, 1]], dtype=bool)
            q = np.zeros((3, 2, 14)); q[:, :, 6] = 1
            q[:, :, 0] = np.arange(6).reshape(3, 2); q[~valid] = np.nan
            arrays = dict(frame_indices=np.arange(3), robot_valid=valid, robot_qpos=q)
            for s, side in enumerate(('left', 'right')):
                poses = np.zeros((3, 4, 7)); poses[:, :, 6] = 1
                poses[:, :, 0] = np.arange(3)[:, None]; poses[~valid[:, s]] = np.nan
                arrays[f'robot_{side}_frame_poses'] = poses
            np.savez_compressed(source / 'retarget.npz', **arrays)
            joints = {side: dict(lower_radians=[-1]*7, upper_radians=[1]*7) for side in ('left', 'right')}
            (source / 'manifest.json').write_text(json.dumps(dict(payload=dict(sha256=digest(source/'retarget.npz')), joints=joints)))
            build(source, root/'presentation')
            display, _ = verify(source, root/'presentation')
            np.testing.assert_array_equal(display['robot_qpos'][valid], q[valid])
            np.testing.assert_array_equal(display['source_robot_valid'], valid)
            display['robot_qpos'][0, 0, 0] += 1
            np.savez_compressed(root/'presentation/presentation.npz', **display)
            # Even an updated payload hash cannot disguise changed provenance.
            m = json.loads((root/'presentation/manifest.json').read_text())
            m['payload_sha256'] = digest(root/'presentation/presentation.npz')
            (root/'presentation/manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                verify(source, root/'presentation')


if __name__ == '__main__':
    unittest.main()
