"""Combined RGB/segmentation rendering must use initialized CUDA channels."""
import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[3] / "reconstruction/modules/v2d_gsplat_refinement/lib/render_features.py"
spec = importlib.util.spec_from_file_location("render_features", path)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


@pytest.mark.parametrize("channels", [5, 7, 9, 17, 33])
def test_all_padding_is_zero_and_supervised_gradients_are_preserved(channels):
    original = torch.arange(3 * channels, dtype=torch.float32).reshape(3, channels).requires_grad_(True)
    padded = features.pad_render_features(original)
    assert padded.shape[-1] in features.SUPPORTED_CHANNELS
    assert torch.equal(padded[:, :channels], original)
    assert torch.count_nonzero(padded[:, channels:]) == 0
    padded.square().sum().backward()
    assert torch.equal(original.grad, 2 * original.detach())


@pytest.mark.parametrize("channels", [2, 3, 4, 8])
def test_supported_channel_count_needs_no_padding(channels):
    original = torch.ones(3, channels)
    assert features.pad_render_features(original) is original


@pytest.mark.parametrize("channels", [0, 513])
def test_invalid_channel_count_is_rejected(channels):
    with pytest.raises(ValueError, match="render features"):
        features.pad_render_features(torch.empty(3, channels))
