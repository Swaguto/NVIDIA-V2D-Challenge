from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_native_pad_geometry import weighted_normal


def test_normal_load_ignores_tangential_friction_and_inactive_contacts():
    n, load = weighted_normal([[1, 0, 0], [0, 1, 0]], [[2, 7, 0], [0, 0, 0]])
    np.testing.assert_array_equal(n, [1, 0, 0])
    assert load == 2


def test_reversing_native_contact_shape_order_reverses_the_normal_only():
    n, load = weighted_normal([[0, 0, -1]], [[0, 0, 3]])
    other, other_load = weighted_normal([[0, 0, 1]], [[0, 0, -3]])
    np.testing.assert_array_equal(n, -other)
    assert load == other_load == 3
    with pytest.raises(ValueError):
        weighted_normal([[0, 0, np.nan]], [[0, 0, 3]])
