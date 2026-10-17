"""Isolate torsional contact from changes to softness, priority or friction."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.contact_profiles import parameters


def test_patch_only_preserves_native_material_parameters():
    assert parameters("default") == {}
    assert parameters("default_patch") == {"condim": 4}


def test_previous_profiles_keep_their_recorded_parameters():
    point = parameters("constraint_point")
    patch = parameters("constraint_patch")
    assert point | {"condim": 4} == patch
    assert "ke" not in point
    assert parameters("firm_point") == point | {"ke": 40000.0, "kd": 400.0}
    with pytest.raises(ValueError):
        parameters("unknown")
