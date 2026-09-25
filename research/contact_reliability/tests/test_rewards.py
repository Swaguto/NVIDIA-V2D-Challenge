import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from v2d_reliability.rewards import contact_terms
from v2d_reliability.isaac_adapter import configure, _weights
from v2d_reliability.core import fit_normalizer, score
from v2d_reliability.io import write_json
from test_core import observation
from upstream_source import upstream_root


def upstream_functions():
    path = upstream_root() / "robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp/utils_jit.py"
    if not path.exists():
        pytest.skip("run scripts/bootstrap_upstream.py for upstream parity checks")
    names = {"contact_wrench_support_reward_jit", "missed_contact_penalty_jit", "unintended_contact_penalty_jit"}
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in functions:
        node.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("empty", [False, True])
def test_all_ones_match_actual_upstream(empty):
    upstream = upstream_functions()
    torch.manual_seed(17)
    cmd, cur = torch.rand(2, 7, 3, 8), torch.rand(2, 7, 3, 8)
    ca, ua = cmd > .4, cur > .5
    if empty:
        ca[:] = False
    result = contact_terms(ca, ua, cmd, cur, torch.ones(2, 7, 3))
    kwargs = {"right_cmd_active": ca[0], "left_cmd_active": ca[1],
              "right_cur_active": ua[0], "left_cur_active": ua[1],
              "right_cmd_active_per_body": ca[0].any(-1), "left_cmd_active_per_body": ca[1].any(-1),
              "right_cur_active_per_body": ua[0].any(-1), "left_cur_active_per_body": ua[1].any(-1),
              "right_cmd_supports": cmd[0], "left_cmd_supports": cmd[1],
              "right_cur_supports": cur[0], "left_cur_supports": cur[1],
              "tolerance": .1, "var": .1, "num_bodies": 3}
    import inspect
    for name, actual in result.items():
        func = upstream[name + "_jit"]
        expected = func(**{k: kwargs[k] for k in inspect.signature(func).parameters})
        torch.testing.assert_close(actual, expected)


def test_attenuation_does_not_renormalize_away():
    support = torch.ones(2, 1, 2, 4)
    active = support.bool()
    baseline = contact_terms(active, active, support, support, torch.ones(2, 1, 2))
    quarter = contact_terms(active, active, support, support, torch.full((2, 1, 2), .25))
    assert baseline["contact_wrench_support_reward"].item() == 1
    assert quarter["contact_wrench_support_reward"].item() == .25
    weight = torch.ones(2, 1, 2); weight[0, 0, 0] = 0
    assert contact_terms(active, active, support, support, weight)["contact_wrench_support_reward"].item() == .75


def test_disabled_adapter_is_exact_noop():
    cfg = SimpleNamespace()
    assert configure(cfg, enabled=False) == []
    assert vars(cfg) == {}


def test_adapter_alignment_and_reset_frame_lookup(tmp_path):
    doc = observation(); norm = fit_normalizer([doc], ["dev"])
    doc["signals"][3][0][1] = [0, 0, 100, 100]
    sidecar = score(doc, norm); path = tmp_path / "sidecar.json"; write_json(path, sidecar)
    md = SimpleNamespace(sequence_id="dev", fps=30, hand_sides=["left", "right"])
    command = SimpleNamespace(_motion_data=md, num_timesteps=8, retargeted_object_body_names=["base", "lid"],
                              device="cpu", timestep_counter=torch.tensor([3, 0]))
    weights = _weights(command, str(path))
    assert weights.shape == (2, 2, 2)
    assert weights[1, 0, 1] < weights[0, 0, 1]
    command.timestep_counter = torch.tensor([0, 3])
    assert _weights(command, str(path))[1, 1, 1] == weights[1, 0, 1]


def test_config_replaces_contact_terms_even_before_curriculum_activation():
    def contact_wrench_support_reward(): pass
    def force_closure_reward(): pass
    first = SimpleNamespace(func=contact_wrench_support_reward, weight=1, params={})
    other = SimpleNamespace(func=force_closure_reward, weight=0, params={})
    cfg = SimpleNamespace(rewards=SimpleNamespace(a=first, b=other))
    assert configure(cfg, "test.json", True) == ["a", "b"]
    assert first.func.__name__ == "weighted_contact_reward"
    assert other.func.__name__ == "weighted_hand_reward"
