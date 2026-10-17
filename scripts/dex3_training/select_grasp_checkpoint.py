"""Rank audited development replays without changing physical acceptance gates.

The worst normalized violation is an engineering diagnostic, not an official
challenge metric. Selection does not certify hidden performance or a repeat.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def select(candidates):
    rows = []
    for folder in candidates:
        report_path = folder / "diagnostic/report.json"
        audit_path = folder / "audit.json"
        report = json.loads(report_path.read_text())
        audit = json.loads(audit_path.read_text())
        evaluation = json.loads((folder / "diagnostic/evaluation-report.json").read_text())
        report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
        if audit["source_report_sha256"] != report_hash:
            raise ValueError("Audit does not match the recorded replay")
        if (report["zero_assistance_control_steps_verified"] != 343
                or not report["virtual_object_actuator_caps_zero"]
                or report["evaluation_reset_stabilization_steps"] != 0):
            raise ValueError("Candidate is not a complete unassisted e002 replay")
        if (evaluation["start_frame"] != 0 or evaluation["settling_steps"] != 0
                or evaluation["non_finite_worlds"]):
            raise ValueError("Candidate did not finish a finite frame-zero evaluation")
        grasps = audit["grasps"]
        expected_worlds = set(range(evaluation["world_count"]))
        for sample in {g["sample"] for g in grasps}:
            if {g["world"] for g in grasps if g["sample"] == sample} != expected_worlds:
                raise ValueError("A grasp audit omits an evaluation world")
        ratios = []
        for grasp in grasps:
            if not grasp["sampled_penetration_verified"]:
                raise ValueError("Candidate lacks sampled penetration evidence")
            ratios.extend([
                grasp["maximum_relative_position_drift_m"] / 0.01,
                grasp["maximum_relative_rotation_drift_deg"] / 5,
                (1 - grasp["all_required_pad_sampled_support_fraction"]) / 0.05,
                max(0, grasp["maximum_reference_lift_m"] - 0.01)
                / max(grasp["maximum_actual_lift_m"], 1e-6),
                grasp["maximum_sampled_hand_object_penetration_m"] / 0.002,
                grasp["maximum_source_position_error_m"] / 0.03,
                grasp["maximum_source_orientation_error_deg"] / 15,
            ])
        if not ratios or not all(math.isfinite(x) and x >= 0 for x in ratios):
            raise ValueError("Candidate lacks finite grasp measurements")
        passed = all(
            g["supported_retention_with_sampled_depth_within_2mm"]
            and g["maximum_source_position_error_m"] <= 0.03
            and g["maximum_source_orientation_error_deg"] <= 15
            for g in grasps
        )
        rows.append(dict(
            label=folder.name, checkpoint_sha256=report["checkpoint_sha256"],
            checkpoint_environment_steps=report["environment_steps"],
            report_sha256=report_hash,
            audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            worst_normalized_grasp_violation=max(ratios),
            all_grasp_gates_passed=passed,
        ))
    if not rows:
        raise ValueError("At least one audited candidate is required")
    return dict(
        schema="v2d.development-grasp-checkpoint-selection.v1", candidates=rows,
        selected=min(rows, key=lambda x: (
            not x["all_grasp_gates_passed"], x["worst_normalized_grasp_violation"])),
        selection_is_not_an_official_metric=True, quality_gates_unchanged=True,
        physical_task_accepted=False, independent_success_repeat_required=True,
        scope="e002 development; sampled contacts; no hidden generalization claim",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.candidate)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
