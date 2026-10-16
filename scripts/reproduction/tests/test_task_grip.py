"""Force-buffer and phase-boundary mechanics for the coupled grasp diagnostic."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from _dex3_task_grip import (  # noqa: E402
    TaskGripReference,
    contact_wrench,
    filtered_reaction_compensation,
)


def test_contact_wrench_uses_active_offsets_and_both_normal_and_friction():
    # Two populated entries deliberately start after unused garbage capacity.
    points = np.array([[99, 99, 99], [0.1, 0, 0], [0, 0.2, 0], [99, 99, 99]])
    normals = np.array([[1, 0, 0], [0, 0, -1], [1, 0, 0], [1, 0, 0]])
    normal = [
        np.array([[999], [2], [3], [999]]),
        points,
        normals,
        np.zeros((4, 1)),
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 99], [99, 2]]),
    ]
    friction = [
        np.array([[999, 999, 999], [0, 0, -0.943]]),
        np.array([[99, 99, 99], [0.1, 0, 0]]),
        np.array([[1]]),
        np.array([[1]]),
    ]
    result = contact_wrench(normal, friction, [0, 0, 0])
    np.testing.assert_allclose(result, [3, 0, -2.943, 0, 0.2943, -0.6])
    shift = np.array([4, -2, 3])
    translated = normal.copy()
    translated[1] = points + shift
    translated_friction = friction.copy()
    translated_friction[1] = friction[1] + shift
    np.testing.assert_allclose(
        contact_wrench(translated, translated_friction, shift), result
    )
    invalid = normal.copy()
    invalid[-1] = np.array([[4, 99], [99, 2]])
    with pytest.raises(ValueError, match="overflow"):
        contact_wrench(invalid, friction, [0, 0, 0])


def test_reaction_feedback_has_correct_sign_bounds_and_release_decay():
    previous = np.zeros(6)
    for _ in range(100):
        previous = filtered_reaction_compensation(
            [0, 0, -2.943, 0, 0.3, 0], previous, 0.01, True
        )
    np.testing.assert_allclose(previous, [0, 0, 2.943, 0, -0.3, 0], atol=1e-8)
    for _ in range(100):
        previous = filtered_reaction_compensation(
            np.ones(6) * 999, previous, 0.01, True
        )
    np.testing.assert_allclose(previous, [-10, -10, -10, -2, -2, -2], atol=1e-8)
    for _ in range(100):
        previous = filtered_reaction_compensation(
            np.ones(6) * 999, previous, 0.01, False
        )
    np.testing.assert_allclose(previous, 0, atol=1e-8)


def test_grasp_schedule_closes_before_loading_and_opens_before_retreat():
    phase = dict(
        sample="fixture",
        approach_start_s=1.0,
        close_s=2.0,
        release_start_s=3.0,
        hand_in_object_xyzw=[0, 0, 0.1, 0, 0, 0, 1],
        open_rad=[0] * 7,
        closed_rad=[0.2] * 7,
        effort_times_s=[2, 3],
        effort_Nm=[[0.3] * 7] * 2,
        required_digits=["thumb", "index"],
    )
    spec = dict(
        schema="v2d.dex3-task-grips.v1",
        physical_task_accepted=False,
        park_height_m=0.12,
        phases={"left": [phase]},
    )
    ref = TaskGripReference(spec)
    obj = [0, 0, 0, 0, 0, 0, 1]
    def command(t):
        return ref.command("left", t, obj)
    assert command(1)["pose"][2] == pytest.approx(0.22)
    assert command(1.75)["pose"][2] == pytest.approx(0.1)
    np.testing.assert_allclose(command(1.875)["fingers"], 0.2)
    np.testing.assert_allclose(command(1.875)["effort"], 0)
    np.testing.assert_allclose(command(2)["effort"], 0.3)
    np.testing.assert_allclose(command(3.25)["fingers"], 0)
    np.testing.assert_allclose(command(3.25)["effort"], 0)
    assert command(3.25)["pose"][2] == pytest.approx(0.1)
    assert command(3.75)["pose"][2] == pytest.approx(0.22)
    # Numerical one-sided limits across every schedule boundary.
    for t in [1, 1.75, 1.875, 2, 3, 3.25, 3.75]:
        a, b = command(t - 1e-7), command(t + 1e-7)
        np.testing.assert_allclose(a["pose"], b["pose"], atol=1e-6)
        np.testing.assert_allclose(a["fingers"], b["fingers"], atol=1e-6)
        np.testing.assert_allclose(a["effort"], b["effort"], atol=1e-6)
    settled = ref.command("left", 1.75, obj, [0, 0, -0.01, 0, 0, 0, 1])
    assert settled["pose"][2] == pytest.approx(0.09)
    carried = ref.command("left", 2.5, obj, [0, 0, -0.01, 0, 0, 0, 1])
    assert carried["pose"][2] == pytest.approx(0.1)
