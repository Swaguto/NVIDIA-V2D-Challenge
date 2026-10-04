"""Check stereo diagnostic evidence boundaries and invalid-match handling."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/track3/validate_stereo.py"
SPEC = importlib.util.spec_from_file_location("stereo_diagnostic", PATH)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


@pytest.mark.parametrize("reverse, expected_count", [(-97, 0), (-4, 720)])
def test_lr_excludes_reverse_invalid_sentinel(monkeypatch, reverse, expected_count):
    calls = []
    class Matcher:
        def __init__(self, disparity): self.disparity = disparity
        def compute(self, left, right):
            return np.full(left.shape, self.disparity * 16, np.int16)
    def create(**kwargs):
        calls.append(kwargs)
        return Matcher(4 if kwargs['minDisparity'] == 0 else reverse)
    monkeypatch.setattr(diagnostic.cv2, 'StereoSGBM_create', create)
    image = np.zeros((24, 64, 3), np.uint8)
    rates, count = diagnostic.lr_stats(image, image)
    assert count == expected_count
    assert all(np.isnan(value) if count == 0 else value == 0 for value in rates.values())
    assert all(call['P1'] == 8 * 25 and call['P2'] == 32 * 25 for call in calls)


def test_plane_diagnostic_does_not_require_camera_z_normal():
    # Plane x=.4 metres, nearly edge-on to the camera, remains a legitimate plane.
    depth = np.broadcast_to(16 / (np.arange(30) + 10), (30, 30)).copy()
    P = np.array([[40, 0, -10, 0], [0, 40, 0, 0], [0, 0, 1, 0]], float)
    ratio, rms, count = diagnostic.plane_inliers(depth, np.ones_like(depth, bool), P)
    assert count == 900
    assert ratio == 1
    assert rms < 1e-6
