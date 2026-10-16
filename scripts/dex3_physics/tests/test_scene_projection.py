import sys
from pathlib import Path
from itertools import product

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from scene_projection import upward_clearance, rate_bounded_majorant  # noqa: E402
from cooked_grasp_audit import convex_escape  # noqa: E402


def test_upward_clearance_removes_overlap_without_moving_disjoint_objects():
    cube = np.array(list(product([-0.01, 0.01], repeat=3)))
    upper = cube + [0, 0, 0.015]
    shift = upward_clearance([cube], [upper])
    assert shift == pytest.approx(0.0052001)
    assert convex_escape(cube, upper + [0, 0, shift])[0] < -0.0002
    assert upward_clearance([cube], [upper + [1, 0, 0]]) == 0
    assert upward_clearance([cube], [upper + [0, 0, 0.02]]) == 0
    with pytest.raises(ValueError, match="above"):
        upward_clearance([cube], [upper], cap=0.001)


def test_reference_smoothing_preserves_clearance_and_limits_speed_in_both_directions():
    values = np.array([0, 0.01, 0, 0.02, 0])
    times = np.array([0, 0.1, 0.3, 0.35, 0.5])
    actual = rate_bounded_majorant(values, times)
    assert np.all(actual >= values)
    assert actual.max() == values.max()
    assert np.max(abs(np.diff(actual)) / np.diff(times)) <= 0.050000001
    np.testing.assert_array_equal(values, [0, 0.01, 0, 0.02, 0])
