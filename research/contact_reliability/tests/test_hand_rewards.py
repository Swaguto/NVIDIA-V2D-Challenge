"""Compare hand-level adapters with actual upstream function bodies on CPU."""
import ast
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import pytest
torch = pytest.importorskip("torch")
from v2d_reliability.isaac_adapter import weighted_hand_reward
from upstream_source import upstream_root


def load_function(path, name, namespace):
    if not path.exists():
        pytest.skip("requires pinned upstream source")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("term", ["force_closure_reward", "motion_contact_tracking_gaussian_exp"])
def test_hand_term_parity_and_attenuation(monkeypatch, term):
    root = upstream_root() / "robotic_grounding/source/robotic_grounding/robotic_grounding/tasks"
    namespace = {"torch": torch, "ManagerBasedEnv": object}
    chamfer = load_function(root / "v2d/mdp/utils.py", "chamfer_distance", namespace)
    module = ModuleType("robotic_grounding.tasks.v2d.mdp.utils")
    module.chamfer_distance = chamfer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace["chamfer_distance"] = chamfer
    source = "contact_rewards.py" if term == "force_closure_reward" else "tracking_rewards.py"
    baseline = load_function(root / "v2d_whole_body/mdp/rewards" / source, term, namespace)
    command = SimpleNamespace(timestep_counter=torch.tensor([0, 1, 2]))
    torch.manual_seed(23)
    for side in ("left", "right"):
        setattr(command, f"{side}_hand_contact_wrench_supports", torch.rand(3, 2, 4))
        setattr(command, f"{side}_hand_contact_active_command", torch.tensor([1., 0., 1.]))
        points = torch.rand(3, 2, 3, 3)
        setattr(command, f"{side}_hand_object_contact_positions_e", points)
        setattr(command, f"{side}_hand_object_contact_positions_w", points + 1)
        setattr(command, f"{side}_hand_object_contact_command_positions_e", points + .01)
        setattr(command, f"retargeted_{side}_object_contact_is_valid", torch.ones(3, 2, 3, dtype=torch.bool))
    command._reliability_cache = ("fixture", torch.ones(3, 2, 2))
    env = SimpleNamespace(num_envs=3, device="cpu", command_manager=SimpleNamespace(get_term=lambda _: command))
    params = {"command_name": "motion"}
    if term == "motion_contact_tracking_gaussian_exp": params["std"] = .05
    expected = baseline(env, **params)
    actual = weighted_hand_reward(env, term, "fixture", **params)
    torch.testing.assert_close(actual, expected)
    command._reliability_cache = ("fixture", torch.full((3, 2, 2), .25))
    torch.testing.assert_close(weighted_hand_reward(env, term, "fixture", **params), expected * .25)
