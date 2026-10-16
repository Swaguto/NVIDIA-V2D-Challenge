"""Summarize coupled readback without claiming policy or official evaluation success."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def live_allocation_summary(rows):
    details = [r.get("live_contact_wrench_control") for r in rows]
    attempts = [d for d in details if d]
    if not attempts:
        return None
    active = [d for d in attempts if d.get("active")]
    patches = [p for d in attempts for p in d.get("input_patch_readback", [])]
    return dict(
        attempted_steps=len(attempts),
        active_steps=len(active),
        active_fraction_of_load_steps=len(active) / len(rows),
        exercised_at_least_once=bool(active),
        fallback_reasons=dict(
            Counter(
                d.get("fallback_to_frozen_position_and_effort", "unspecified")
                for d in attempts
                if not d.get("active")
            )
        ),
        patch_readbacks=len(patches),
        patches_with_negative_force_scalar=sum(
            p.get("signed_scalar_min_N", 0) < 0 for p in patches
        ),
        acceptance_scope="Activation diagnostic only; exercising a controller does not accept a grasp",
    )


def summarize(run):
    report = json.loads((run / "report.json").read_text())
    spec_file = run.parent / "static-load-inputs/task-grips.json"
    if (
        digest(run / "states.json") != report["state_sha256"]
        or digest(spec_file) != report["task_grip_spec_sha256"]
    ):
        raise ValueError("Coupled states or grasp specification changed")
    spec = json.loads(spec_file.read_text())
    states = json.loads((run / "states.json").read_text())
    end = spec["source_times_s"][-1]
    lookup = {(r["step"], r["object"]): r for r in states["objects"]}
    metrics, grasps = [], []
    for oi in [0, 1]:
        rows = [
            r
            for r in states["objects"]
            if r["object"] == oi
            and r["step"] * report["physics_dt_s"] * report["source_motion_speed"]
            <= end + 1e-8
            and r["source_object_observed"]
        ]
        pos = np.array([r["position_error_m"] for r in rows])
        metrics.append(
            dict(
                object=oi,
                observed_source_steps=len(rows),
                position_mean_m=float(pos.mean()),
                position_median_m=float(np.median(pos)),
                position_p90_m=float(np.quantile(pos, 0.9)),
                position_max_m=float(pos.max()),
            )
        )
    for side, phases in spec["phases"].items():
        hand_rows = [r for r in states["hands"] if r["side"] == side]
        for phase in phases:
            rows = [
                r
                for r in hand_rows
                if phase["close_s"] + 0.05
                <= r["source_time_s"]
                < phase["release_start_s"]
            ]
            obj = np.array(
                [
                    lookup[r["step"], phase["object"]]["actual_root_pose_stage_xyzw"]
                    for r in rows
                ]
            )
            hand = np.array([r["actual_root_pose_stage_xyzw"] for r in rows])
            inverse = Rotation.from_quat(hand[:, 3:]).inv()
            relative = inverse.apply(obj[:, :3] - hand[:, :3])
            drift = np.linalg.norm(relative - relative[0], axis=1)
            coverage = np.mean(
                [r["controller_details"]["contact_hold_remaining_s"] > 0 for r in rows]
            )
            item = dict(
                sample=phase["sample"],
                source_interval_s=[phase["close_s"] + 0.05, phase["release_start_s"]],
                load_steps=len(rows),
                maximum_relative_position_drift_m=float(drift.max()),
                maximum_relative_rotation_drift_deg=float(
                    np.rad2deg(
                        (
                            (inverse * Rotation.from_quat(obj[:, 3:]))
                            * (inverse[0] * Rotation.from_quat(obj[0, 3:])).inv()
                        ).magnitude()
                    ).max()
                ),
                contact_gate_coverage_fraction=float(coverage),
                retained_within_10mm_diagnostic=bool(drift.max() <= 0.01),
            )
            live = live_allocation_summary(rows)
            if live is not None:
                item["live_contact_allocation"] = live
            if "normal_force_by_hand_body_and_object_world_N" in rows[0]:
                binding = report["physical_contact_bindings"][
                    ("left", "right").index(side)
                ]
                forces = np.array(
                    [r["normal_force_by_hand_body_and_object_world_N"] for r in rows]
                )[:, :, phase["object"], :]
                magnitudes = np.linalg.norm(forces, axis=-1)
                digit_loads = np.array(
                    [
                        magnitudes[
                            :,
                            [
                                f"_{digit}_" in name
                                for name in binding["hand_body_paths"]
                            ],
                        ].sum(axis=1)
                        for digit in phase["required_digits"]
                    ]
                ).T
                item["primary_digit_load_p10_N"] = {
                    digit: float(np.quantile(digit_loads[:, i], 0.1))
                    for i, digit in enumerate(phase["required_digits"])
                }
                item["all_primary_digits_above_0_5N_fraction"] = float(
                    np.mean((digit_loads >= 0.5).all(axis=1))
                )
                item["maximum_actual_combined_finger_effort_Nm"] = float(
                    np.max(
                        np.abs([r["actual_combined_finger_effort_Nm"] for r in rows])
                    )
                )
            grasps.append(item)
    return dict(
        schema="v2d.coupled-grasp-readback-audit.v1",
        report_sha256=digest(run / "report.json"),
        states_sha256=report["state_sha256"],
        task_grip_spec_sha256=report["task_grip_spec_sha256"],
        source_object_metrics_exclude_inferred_postroll=True,
        objects=metrics,
        grasps=grasps,
        all_four_grasps_retained_within_10mm=all(
            r["retained_within_10mm_diagnostic"] for r in grasps
        ),
        physics_controls_pass=report["gravity_and_contact_controls_pass"],
        initial_scene_clearance=report["initial_scene_clearance_check"],
        physical_task_accepted=False,
        policy_trained=False,
        official_score_claimed=False,
        scope="Recorded simulated grasp retention and original inferred-source tracking only; no ground-truth contact, continuous collision, or official success certificate",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
