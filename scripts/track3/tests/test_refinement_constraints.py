"""Physical controls for bounded refinement rather than unconstrained drift."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

path = Path(__file__).resolve().parents[3] / "reconstruction/modules/v2d_gsplat_refinement/lib/constraints.py"
spec = importlib.util.spec_from_file_location("refinement_constraints", path)
constraints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constraints)


def test_translation_changes_stay_within_metric_radius():
    p = torch.nn.Parameter(torch.tensor([[0., 0., 1.], [1., 1., 1.]]))
    initial = p.detach().clone()
    guard = constraints.PoseTrustRegion()
    guard.add("translation", p, 0.003)
    with torch.no_grad():
        p.add_(torch.tensor([[1., 2., 3.], [0.001, 0., 0.]]))
    result = guard.project()
    assert result['translation'] == 1
    assert torch.all((p - initial).norm(dim=-1) <= 0.003001)
    assert torch.allclose(p[1], initial[1] + torch.tensor([0.001, 0., 0.]))


def test_physical_rotation_stays_bounded_near_axis_angle_pi():
    p = torch.nn.Parameter(torch.tensor([[0., 0., 0.], [3.13, 0., 0.], [1., -2., 0.5]]))
    initial = p.detach().numpy().copy()
    guard = constraints.PoseTrustRegion()
    guard.add("rotation", p, np.deg2rad(0.5))
    with torch.no_grad():
        p.add_(torch.tensor([[3., 1., -2.], [0.3, 0.2, 0.1], [-1., 4., -2.]]))
    guard.project()
    change = Rotation.from_rotvec(p.detach().numpy()) * Rotation.from_rotvec(initial).inv()
    assert np.all(change.magnitude() <= np.deg2rad(0.50001))


def test_each_finger_joint_is_bounded_independently():
    p = torch.nn.Parameter(torch.zeros((2, 45)))
    guard = constraints.PoseTrustRegion()
    guard.add("fingers", p, 0.01)
    with torch.no_grad():
        p[:, :3].fill_(1.)
        p[:, 3:6].fill_(0.001)
    guard.project()
    assert torch.all(p.reshape(2, 15, 3).norm(dim=-1) <= 0.010001)
    assert torch.all(p[:, 3:6] == 0.001)


def test_nonfinite_updates_are_rejected_not_hidden_by_projection():
    p = torch.nn.Parameter(torch.zeros((1, 3)))
    guard = constraints.PoseTrustRegion()
    guard.add("root", p, 0.01)
    with torch.no_grad():
        p[0, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        guard.project()


@pytest.mark.parametrize('radius', [-1, float('inf'), float('nan')])
def test_invalid_trust_radius_is_rejected(radius):
    with pytest.raises(ValueError):
        constraints.PoseTrustRegion().add("root", torch.zeros((1, 3)), radius)
