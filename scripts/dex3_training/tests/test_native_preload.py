"""The native preload cannot add an opening component or exceed its torque cap."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from prepare_native_preload import compressive_projection


def test_opening_torque_is_replaced_with_bounded_compression():
    jac = np.zeros((2, 7))
    jac[0, 0], jac[1, 4] = -0.04, 0.03
    old = np.array([0.2, 0, 0, 0, 0.15, 0, 0])
    target, force = compressive_projection(old, jac, 2)
    np.testing.assert_allclose(target[[0, 4]], [-0.08, 0.15])
    np.testing.assert_allclose(force, [2, 5])


def test_motor_limit_and_invalid_preload_fail_closed():
    jac = np.ones((1, 7))
    for floor in (0, 7, 2):
        with pytest.raises(ValueError):
            compressive_projection(np.zeros(7), jac, floor)
