"""Numerical controls for actual optimizer losses, including observation gaps."""
import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[3] / "reconstruction/modules/v2d_gsplat_refinement/lib/losses.py"
spec = importlib.util.spec_from_file_location("refinement_losses", path)
losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses)


def test_temporal_loss_does_not_bridge_missing_observations():
    p = torch.tensor([[0.0], [1.0], [100.0], [101.0]], requires_grad=True)
    loss = losses.temporal_smoothness(p, [0, 1, 10, 11])
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    assert torch.allclose(p.grad[:, 0], torch.tensor([-2.0, 2.0, -2.0, 2.0]))


def test_rotation_loss_does_not_pull_reacquired_hand_through_gap():
    aa = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 3.14], [0., 0., 3.14]])
    assert losses.rotation_smoothness(aa, [0, 1, 10, 11]).item() == 0.0
    assert losses.rotation_smoothness(aa).item() > 1.0


def test_unknown_stereo_depth_has_finite_loss_and_zero_gradient():
    rendered = torch.tensor([[0.8, 0.9], [1.1, 0.7]], requires_grad=True)
    target = torch.tensor([[0.8, float('inf')], [float('nan'), 0.8]])
    loss = losses.depth_loss(rendered, target, torch.ones_like(rendered))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(rendered.grad).all()
    assert rendered.grad[0, 1] == 0 and rendered.grad[1, 0] == 0
