from types import SimpleNamespace, ModuleType
import sys
import pytest
pytest.importorskip("torch")
from v2d_reliability.isaac_adapter import configure_variant
from v2d_reliability.io import write_json, sha256


def config(monkeypatch):
    def contact_wrench_support_reward(): pass
    def force_closure_reward(): pass
    name = "robotic_grounding.tasks.v2d_whole_body.mdp.rewards.contact_rewards"
    module = ModuleType(name); module.force_closure_reward = force_closure_reward
    monkeypatch.setitem(sys.modules, name, module)
    term = SimpleNamespace(func=contact_wrench_support_reward, params={}, weight=5.)
    rewards = SimpleNamespace(contact_wrench_support_reward=term,
        missed_contact_penalty=SimpleNamespace(weight=-1.), unintended_contact_penalty=SimpleNamespace(weight=-2.))
    schedules = {"contact_wrench_support_reward": [1, 5], "missed_contact_penalty": [-1, -2]}
    curriculum = SimpleNamespace(fixed_timestep_curriculum=SimpleNamespace(params={"reward_weight_schedules": schedules}))
    return SimpleNamespace(rewards=rewards, curriculum=curriculum)


def test_fallback_preserves_schedule_and_disables_penalties(monkeypatch):
    cfg = config(monkeypatch)
    configure_variant(cfg, "force_closure")
    assert cfg.rewards.contact_wrench_support_reward.func.__name__ == "force_closure_reward"
    assert cfg.rewards.contact_wrench_support_reward.weight == 5
    schedules = cfg.curriculum.fixed_timestep_curriculum.params["reward_weight_schedules"]
    assert schedules["contact_wrench_support_reward"] == [1, 5]
    assert schedules["missed_contact_penalty"] == [0, 0]
    assert cfg.rewards.unintended_contact_penalty.weight == 0


def test_combination_requires_reference_binding(monkeypatch, tmp_path):
    cfg = config(monkeypatch); ref = tmp_path / "motion.parquet"; ref.write_bytes(b"fixture")
    cfg.motion_file = str(ref); sidecar = tmp_path / "sidecar.json"
    write_json(sidecar, {"reference_sha256": "wrong"})
    with pytest.raises(ValueError): configure_variant(cfg, "reliability_force_closure", sidecar)
    assert cfg.rewards.contact_wrench_support_reward.func.__name__ == "contact_wrench_support_reward"
    write_json(sidecar, {"reference_sha256": sha256(ref)})
    configure_variant(cfg, "reliability_force_closure", sidecar)
    assert cfg.rewards.contact_wrench_support_reward.func.__name__ == "weighted_hand_reward"


def test_baseline_is_noop():
    cfg = SimpleNamespace()
    assert configure_variant(cfg) == [] and not vars(cfg)
