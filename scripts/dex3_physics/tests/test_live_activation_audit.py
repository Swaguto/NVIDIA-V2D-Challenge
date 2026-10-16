import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_coupled_grip import live_allocation_summary  # noqa: E402


def test_fallback_is_never_reported_as_exercised_live_control():
    fallback = {
        "live_contact_wrench_control": {
            "active": False,
            "fallback_to_frozen_position_and_effort": "missing pad",
        }
    }
    result = live_allocation_summary([fallback, fallback])
    assert not result["exercised_at_least_once"]
    assert result["active_fraction_of_load_steps"] == 0
    assert result["fallback_reasons"] == {"missing pad": 2}
    active = {
        "live_contact_wrench_control": {
            "active": True,
            "input_patch_readback": [{"signed_scalar_min_N": -3}],
        }
    }
    mixed = live_allocation_summary([fallback, active, {}])
    assert mixed["active_steps"] == 1
    assert mixed["attempted_steps"] == 2
    assert mixed["active_fraction_of_load_steps"] == 1 / 3
    assert mixed["patches_with_negative_force_scalar"] == 1
    assert live_allocation_summary([{}, {}]) is None
