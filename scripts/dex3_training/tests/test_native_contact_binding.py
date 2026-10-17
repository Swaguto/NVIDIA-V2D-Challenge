import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "native_contact_binding",
    Path(__file__).parents[1] / "v2d_dex3_flashchord/native_contact_binding.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_inverse_preserves_permutations_unmapped_bodies_and_worlds():
    original = np.array([[-1, 2, 0], [-1, 3, 5]], dtype=np.int32)
    saved = original.copy()
    actual = module.invert_body_mapping(original, 3)
    np.testing.assert_array_equal(actual, [2, -1, 1, 1, -1, 2])
    np.testing.assert_array_equal(original, saved)


@pytest.mark.parametrize(
    "mapping,bodies",
    [([[0, 0]], 2), ([[0], [0]], 2), ([[2]], 2), ([[-2]], 2),
     ([[0.0]], 2), ([0, 1], 2), ([[0]], 0), (np.empty((0, 2), int), 2)],
)
def test_invalid_or_ambiguous_mapping_fails_closed(mapping, bodies):
    with pytest.raises(ValueError):
        module.invert_body_mapping(mapping, bodies)
