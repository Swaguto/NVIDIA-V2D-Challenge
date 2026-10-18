"""Anchored tracking must retain original pixels, frame IDs, and raw mesh origins."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from foundationpose_track import centered_tracker_pose, foreground_depth, stereo_frames


class Capture:
    def __init__(self, count, offset=0):
        self.frames = [np.full((2, 3, 3), i + offset, np.uint8) for i in range(count)]
        self.index = 0

    def read(self):
        if self.index == len(self.frames):
            return False, None
        value = self.frames[self.index]
        self.index += 1
        return True, value.copy()


@pytest.mark.parametrize("anchor", [None, 0, 3, 6])
def test_each_decoded_source_frame_appears_once_with_its_original_pixels(anchor):
    rows = list(stereo_frames(Capture(7), Capture(7, 20), anchor))
    assert sorted(row[0] for row in rows) == list(range(7))
    for frame, left, right, _ in rows:
        np.testing.assert_array_equal(left, frame)
        np.testing.assert_array_equal(right, frame + 20)
    if anchor is not None:
        assert [r[0] for r in rows] == list(range(anchor, 7)) + list(
            range(anchor - 1, -1, -1)
        )
        assert [r[0] for r in rows if r[3]] == ([anchor, anchor - 1] if anchor else [0])


def test_mismatched_or_missing_anchor_frames_fail_explicitly():
    with pytest.raises(ValueError, match="mismatch"):
        list(stereo_frames(Capture(4), Capture(3), 1))
    with pytest.raises(ValueError, match="outside"):
        list(stereo_frames(Capture(4), Capture(4), 4))


def test_tracker_centering_does_not_change_the_reconstructed_body_frame():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.4, 1.2]).as_matrix()
    pose[:3, 3] = [0.1, 0.2, 0.7]
    center = np.array([-0.03, 0.04, 0.06])
    centered = centered_tracker_pose(pose, center)
    local = np.array([0.01, 0.02, 0.03])
    np.testing.assert_allclose(
        centered[:3, :3] @ local + centered[:3, 3],
        pose[:3, :3] @ (local + center) + pose[:3, 3],
    )


def test_foreground_refinement_preserves_measurements_and_excludes_other_surfaces():
    depth = np.array([[0.6, 0.7, np.nan], [0, -1, 0.8]], np.float32)
    original = depth.copy()
    valid = np.array([[True, True, True], [True, True, False]])
    mask = np.array([[255, 0, 255], [255, 255, 255]], np.uint8)
    np.testing.assert_array_equal(
        foreground_depth(depth, valid, mask),
        np.array([[0.6, 0, 0], [0, 0, 0]], np.float32),
    )
    np.testing.assert_array_equal(depth, original)
    with pytest.raises(ValueError, match="grid"):
        foreground_depth(depth, valid[:, :2], mask)
