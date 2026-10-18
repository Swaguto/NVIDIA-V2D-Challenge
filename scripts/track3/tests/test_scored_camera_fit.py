import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "reconstruction/validation/issue6"))
from fit_validation import verify_source_cameras


def fixture(tmp_path, offset=0):
    cameras = np.tile(np.eye(4), (4, 1, 1))
    cameras[:, 0, 3] = np.arange(4) * 0.01
    source = tmp_path / "source.npz"
    np.savez(source, w2c=cameras)
    ids = [0, 1, 2]
    final = tmp_path / "final.npz"
    np.savez(
        final,
        cam_R=np.tile(cameras[ids, :3, :3][None], (2, 1, 1, 1)),
        cam_t=np.tile(cameras[ids, :3, 3][None] + offset, (2, 1, 1)),
    )
    return source, final, ids


def test_calibrated_fixed_camera_fit_retains_exact_source_frame_axis(tmp_path):
    source, final, ids = fixture(tmp_path)
    report = verify_source_cameras(source, final, ids)
    assert (
        report["source_frames_checked"] == 3
        and report["hand_camera_tracks_checked"] == 2
    )


def test_random_upstream_camera_translation_is_rejected(tmp_path):
    source, final, ids = fixture(tmp_path, 0.1)
    with pytest.raises(AssertionError):
        verify_source_cameras(source, final, ids)


def test_a_shifted_camera_frame_axis_is_rejected(tmp_path):
    source, final, ids = fixture(tmp_path)
    with pytest.raises(AssertionError):
        verify_source_cameras(source, final, [1, 2, 3])
