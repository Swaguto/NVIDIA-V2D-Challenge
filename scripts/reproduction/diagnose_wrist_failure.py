"""Measure the onset of a saved physical wrist failure without rerunning Isaac."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def diagnose(states: Path, expected_sha256: str) -> dict:
    """Report oscillation and error onset separately from contact observations."""
    sha = hashlib.sha256(states.read_bytes()).hexdigest()
    if sha != expected_sha256:
        raise ValueError("Historical state hash differs")
    data = json.loads(states.read_text())
    report = {
        "schema": "v2d.wrist-failure-onset.v1",
        "state_sha256": sha,
        "physical_task_accepted": False,
        "causal_explanation_established": False,
        "per_side": {},
    }
    for side in ("left", "right"):
        rows = [r for r in data["hands"] if r["side"] == side]
        times = np.array([r["source_time_s"] for r in rows])
        actual = np.array([r["actual_root_pose_stage_xyzw"] for r in rows])
        target = np.array([r["target_root_pose_stage_xyzw"] for r in rows])
        error = np.linalg.norm(actual[:, :3] - target[:, :3], axis=-1)
        angle = (
            Rotation.from_quat(target[:, 3:]) * Rotation.from_quat(actual[:, 3:]).inv()
        ).magnitude()
        torque = np.array([r["applied_wrist_torque_local_Nm"] for r in rows])
        contact = np.array([max(r["object_contact_force_magnitudes_N"]) for r in rows])

        def first(mask):
            indices = np.flatnonzero(mask)
            return float(times[indices[0]]) if len(indices) else None

        early = times <= 0.05
        report["per_side"][side] = {
            "first_position_error_over_10cm_source_s": first(error > 0.1),
            "first_orientation_error_over_90deg_source_s": first(angle > np.pi / 2),
            "first_positive_sampled_object_contact_source_s": first(contact > 1e-8),
            "early_window_source_s": [0.0, 0.05],
            "early_torque_x_sign_changes": int(
                np.sum(torque[1:, 0][early[1:]] * torque[:-1, 0][early[1:]] < 0)
            ),
            "early_peak_component_torque_Nm": float(np.abs(torque[early]).max()),
            "position_median_m": float(np.median(error)),
            "orientation_median_deg": float(np.degrees(np.median(angle))),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.states, args.sha256)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
