"""Sparse stereo holes must never become synthetic finite surfaces."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

path = Path(__file__).resolve().parents[3] / "reconstruction/modules/v2d_gsplat_refinement/lib/depth.py"
spec = importlib.util.spec_from_file_location("refinement_depth", path)
depth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth)


def test_resize_keeps_unknown_separate_from_metric_samples(tmp_path):
    source = np.array([[0, 32768], [43690, 0]], dtype=np.uint16)
    file = tmp_path / "depth.png"
    Image.fromarray(source).save(file)
    observed = depth.load_inverse_depth(file, 6, 6)
    assert np.isinf(observed[:3, :3]).all()
    assert np.isinf(observed[3:, 3:]).all()
    assert observed[:3, 3:] == pytest.approx(np.full((3, 3), 65535 / 32768 - 1))
    assert observed[3:, :3] == pytest.approx(np.full((3, 3), 0.5))
    assert observed[np.isfinite(observed)].max() < 1.0


def test_downsample_retains_holes_and_does_not_average_inverse_depth(tmp_path):
    file = tmp_path / "depth.png"
    source = np.repeat(np.repeat(np.array([[0, 32768], [43690, 65535]], dtype=np.uint16), 2, 0), 2, 1)
    Image.fromarray(source).save(file)
    observed = depth.load_inverse_depth(file, 2, 2)
    assert np.isinf(observed[0, 0]) and np.isinf(observed[1, 1])
    assert observed[0, 1] == pytest.approx(65535 / 32768 - 1)
    assert observed[1, 0] == pytest.approx(0.5)


def test_color_image_cannot_masquerade_as_depth(tmp_path):
    file = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(file)
    with pytest.raises(ValueError, match="inverse-u16"):
        depth.load_inverse_depth(file, 2, 2)
