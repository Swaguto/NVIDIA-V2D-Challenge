"""Run the optimizer's temporal objective on CPU without importing the renderer."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

LIB = Path(__file__).resolve().parents[3] / "reconstruction/modules/v2d_gsplat_refinement/lib"


def load_temporal_objective():
    spec = importlib.util.spec_from_file_location("temporal_losses", LIB / "losses.py")
    losses = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses)
    namespace = {"torch": torch, **vars(losses)}
    # Compile the actual CPU-only definitions, leaving CUDA/asset imports out.
    selected = {
        "gaussians.py": {"axis_angle_to_quat", "quat_to_rotmat"},
        "background.py": {"BackgroundPoseField"},
        "refine_simple.py": {"_normalized_smooth", "_observed_smooth", "_pose_smoothness"},
    }
    namespace["nn"] = torch.nn
    for filename, names in selected.items():
        tree = ast.parse((LIB / filename).read_text(encoding="utf-8"))
        body = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        assert {node.name for node in body} == names
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                 *body], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(LIB / filename), "exec"), namespace)  # noqa: S102 -- trusted repository definitions
    return namespace


@pytest.mark.parametrize("term", ["translation", "rotation", "acceleration_translation", "acceleration_rotation"])
@pytest.mark.parametrize("gapped", [False, True])
def test_camera_temporal_objective_does_not_bridge_reacquisition(term, gapped):
    functions = load_temporal_objective()
    frames = [0, 1, 2, 10, 11, 12] if gapped else list(range(6))
    camera = functions["BackgroundPoseField"](6, device="cpu")
    # Each segment is stationary, separated by a translation and turn.
    with torch.no_grad():
        camera.translation[3:, 0] = 10
        camera.axis_angle[3:, 2] = 1
    obj = SimpleNamespace(pose_field=SimpleNamespace(
        axis_angle=torch.zeros(6, 3), translation=torch.zeros(6, 3),
        frame_indices=frames, num_frames=lambda: 6,
    ))
    weights = dict.fromkeys([
        "w_smooth_object_rot", "w_smooth_object_trans", "w_smooth_hand_rot",
        "w_smooth_hand_articulation", "w_smooth_hand_trans", "w_smooth_hand_object_relative_rot",
        "w_smooth_hand_object_relative_trans", "w_smooth_camera_rot", "w_smooth_camera_trans",
        "w_camera_acceleration_trans", "w_camera_acceleration_rot",
    ], 0)
    weight = {
        "translation": "w_smooth_camera_trans", "rotation": "w_smooth_camera_rot",
        "acceleration_translation": "w_camera_acceleration_trans",
        "acceleration_rotation": "w_camera_acceleration_rot",
    }[term]
    weights[weight] = 1
    # Deliberately reverse insertion order: row indices determine temporal order.
    frame_to_t = dict(reversed(list(zip(frames, range(6)))))
    loss = functions["_pose_smoothness"]([obj], [], camera, frame_to_t=frame_to_t, **weights)
    if gapped:
        assert loss.item() == pytest.approx(0, abs=1e-12)
    else:
        assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(camera.translation.grad).all()
    assert torch.isfinite(camera.axis_angle.grad).all()
    if gapped:
        assert camera.translation.grad.abs().max().item() < 1e-12
        assert camera.axis_angle.grad.abs().max().item() < 1e-12
