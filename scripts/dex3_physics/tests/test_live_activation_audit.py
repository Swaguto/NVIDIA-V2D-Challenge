import sys
import json
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_coupled_grip import live_allocation_summary, summarize  # noqa: E402


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


def test_stationary_aborted_grasp_cannot_pass_retention(tmp_path):
    run = tmp_path / "simulator"
    run.mkdir()
    inputs = tmp_path / "static-load-inputs"
    inputs.mkdir()
    spec = {
        "source_times_s": [0.0, 1.0],
        "phases": {
            "left": [
                {
                    "sample": "aborted",
                    "close_s": 0.0,
                    "release_start_s": 1.0,
                    "object": 0,
                }
            ]
        },
    }
    pose = [0, 0, 0, 0, 0, 0, 1]
    states = {
        "objects": [
            {
                "step": i,
                "object": oi,
                "actual_root_pose_stage_xyzw": pose,
                "source_object_observed": True,
                "position_error_m": 0.0,
            }
            for i in [1, 2]
            for oi in [0, 1]
        ],
        "hands": [
            {
                "step": i,
                "side": "left",
                "source_time_s": i * 0.1,
                "actual_root_pose_stage_xyzw": pose,
                "controller_details": {"contact_hold_remaining_s": 0.03},
            }
            for i in [1, 2]
        ],
    }
    spec_file = inputs / "task-grips.json"
    spec_file.write_text(json.dumps(spec))
    state_file = run / "states.json"
    state_file.write_text(json.dumps(states))
    report = {
        "state_sha256": hashlib.sha256(state_file.read_bytes()).hexdigest(),
        "task_grip_spec_sha256": hashlib.sha256(spec_file.read_bytes()).hexdigest(),
        "physics_dt_s": 0.1,
        "source_motion_speed": 1,
        "aborted_reacquisition_phases": {"aborted": 0.0},
        "gravity_and_contact_controls_pass": True,
        "initial_scene_clearance_check": {},
    }
    (run / "report.json").write_text(json.dumps(report))
    result = summarize(run)
    assert result["grasps"][0]["maximum_relative_position_drift_m"] == 0
    assert result["grasps"][0]["phase_aborted"]
    assert not result["all_four_grasps_retained_within_10mm"]
    report["aborted_reacquisition_phases"] = {}
    for row in states["hands"]:
        row["controller_details"]["contact_hold_remaining_s"] = 0
    state_file.write_text(json.dumps(states))
    report["state_sha256"] = hashlib.sha256(state_file.read_bytes()).hexdigest()
    (run / "report.json").write_text(json.dumps(report))
    assert not summarize(run)["grasps"][0]["retained_within_10mm_diagnostic"]
