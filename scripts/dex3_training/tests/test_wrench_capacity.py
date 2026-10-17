from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.wrench_capacity import capacity
from audit_native_wrench import audit


def test_opposed_contacts_can_lift_but_cannot_use_external_support():
    result = capacity(
        [[-0.05, 0, 0], [0.05, 0, 0]],
        [[1, 0, 0], [-1, 0, 0]],
        [4, 4],
        [0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        0.5,
    )
    assert result["load_multiplier"] == pytest.approx(2.0)
    empty = capacity([], [], [], [0, 0, 0], [0, 0, 2, 0, 0, 0], 0.5)
    assert empty["load_multiplier"] == 0


def test_offset_single_contact_cannot_balance_moment_even_with_large_force():
    result = capacity(
        [[0.1, 0, 0]], [[0, 0, 1]], [100], [0, 0, 0], [0, 0, 2, 0, 0, 0], 1.25
    )
    assert result["load_multiplier"] == pytest.approx(0)


def test_com_shift_and_wrong_normal_direction_do_not_create_free_wrench():
    args = ([[-0.05, 0, 0], [0.05, 0, 0]], [[1, 0, 0], [-1, 0, 0]], [4, 4])
    result = capacity(*args, [0, 0.1, 0], [0, 0, 2, 0, 0, 0], 0.5)
    assert result["load_multiplier"] == pytest.approx(0)
    with pytest.raises(ValueError):
        capacity([[0, 0, 0]], [[2, 0, 0]], [1], [0, 0, 0], np.ones(6), 0.5)


def test_recorded_body0_force_sign_and_empty_contact_frame(tmp_path):
    import json

    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(
        json.dumps({"metrics": {"world_count": 1}, "control_fps": 20})
    )
    (run / "body-labels.json").write_text(
        json.dumps(["dex3_left/pad", "white_pot/object", "support"])
    )
    np.savez(
        run / "native-state-readback.npz",
        body_poses=np.tile([0, 0, 0, 0, 0, 0, 1], (2, 3, 1)),
    )
    np.savez(
        run / "native-model-readback.npz",
        body_mass=[0, 0.1, 0.3, 1],
        body_ipos=np.zeros((4, 3)),
        geom_bodyid=[1, 2],
        geom_friction=[[1.25, 0, 0], [1.25, 0, 0]],
    )
    np.savez(
        run / "native-contacts-readback.npz",
        frame_offsets=[0, 1, 1],
        body0_ids=[0],
        body1_ids=[1],
        contact_force_w=[[0, 0, -2.943]],
        contact_torque_w=[[0, 0, 0]],
        contact_pos_w=[[0, 0, 0]],
        contact_normal_w=[[0, 0, 1]],
    )
    result = audit(run, tmp_path / "audit.json", stride_frames=1)
    loaded, empty = result["rows"]
    assert loaded["load_multiplier"] == pytest.approx(1)
    np.testing.assert_allclose(
        loaded["measured_hand_wrench_about_object_com"], [0, 0, 2.943, 0, 0, 0]
    )
    assert not loaded["sampled_support_bears_weight"]
    assert empty["load_multiplier"] == 0
