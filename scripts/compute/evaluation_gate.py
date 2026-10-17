"""Allocate further GPU work only after comparable, unassisted evaluations improve.

This is an experiment allocation gate, not an official score or task acceptance.
It reads the existing readback artifacts and original checkpoint metadata without
changing the evaluator, simulator, training recipe or saved policies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Expected a finite numerical metric")
    return value


def metadata(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated checkpoint")
        size = struct.unpack("<Q", raw)[0]
        if not 2 <= size <= min(64 * 1024 * 1024, Path(path).stat().st_size - 8):
            raise ValueError("Invalid checkpoint header")
        return json.loads(stream.read(size))["__metadata__"]


def experiment(config):
    config = json.loads(config)
    for key in (
        "output_dir",
        "resume_context_manifest",
        "trace_training_transitions",
        "numerical_failure_policy",
    ):
        config.pop(key, None)
    for key in ("resume", "checkpoint", "max_iterations", "save_interval"):
        config["training"].pop(key, None)
    return config


def read_evaluation(directory, checkpoint, inputs):
    directory, inputs, checkpoint = Path(directory), Path(inputs), Path(checkpoint)
    report = json.loads((directory / "report.json").read_text())
    evaluation = json.loads((directory / "evaluation-report.json").read_text())
    source = json.loads((inputs / "manifest.json").read_text())
    meta = metadata(checkpoint)
    checkpoint_sha, input_sha = sha256(checkpoint), sha256(inputs / "manifest.json")
    frames = source["selected_frame_count"]
    if (
        report.get("schema") != "v2d.dex3-full-task-policy-readback.v1"
        or report.get("source_manifest_sha256") != input_sha
        or meta.get("user.dex3_input_manifest_sha256") != input_sha
        or report.get("checkpoint_sha256") != checkpoint_sha
        or evaluation.get("checkpoint_sha256") != checkpoint_sha
        or report.get("environment_steps") != int(meta["user.environment_steps"])
        or evaluation.get("environment_steps") != int(meta["user.environment_steps"])
        or report.get("frame_count") != frames
        or report.get("source_frame_count") != source["source_frame_count"]
        or evaluation.get("step_count") != frames
        or evaluation.get("world_count", 0) < 4
        or evaluation.get("non_finite_worlds") != []
        or evaluation.get("reset_mode") != "explicit"
        or evaluation.get("settling_steps") != 0
        or evaluation.get("start_frame") != 0
        or evaluation.get("motion_start_frame") != 0
        or evaluation.get("motion_end_frame") != -1
        or evaluation.get("sim_control_fps") != 20
        or report.get("virtual_object_assistance") != 0
        or report.get("virtual_object_actuator_caps_zero") is not True
        or report.get("evaluation_reset_stabilization_steps") != 0
        or report.get("zero_assistance_control_steps_verified", 0) < frames
        or report.get("metrics") != evaluation.get("metrics")
        or report.get("reference_tracking") != evaluation.get("reference_tracking")
        or report.get("diagnostic_source_sha256")
        != sha256(Path(__file__).parents[1] / "dex3_training/record_policy_readback.py")
    ):
        raise ValueError(
            "Evaluation evidence does not match the frozen full readback contract"
        )
    if frames != source["source_frame_count"]:
        raise ValueError("Allocation gate requires the complete development episode")
    if evaluation["reference_tracking"].get("reference_end_fraction") != 1:
        raise ValueError("Evaluation did not cover the complete reference")
    for key in ("world_count", "step_count"):
        if evaluation["metrics"][key] != evaluation[key]:
            raise ValueError("Evaluation metric coverage differs")
    if evaluation["metrics"]["object_body_names"] != [
        obj["name"] for obj in source["objects"]
    ]:
        raise ValueError("Evaluation object order differs from the inputs")
    for key in (
        "user.dex3_runtime_patches_sha256",
        "user.dex3_image_context_sha256",
        "user.policy_schema_json",
    ):
        if not meta.get(key):
            raise ValueError("Checkpoint lacks runtime provenance")
    return report, evaluation, meta


def compare(
    baseline,
    candidate,
    baseline_checkpoint,
    candidate_checkpoint,
    inputs,
    min_auc_gain=0.005,
):
    finite_number(min_auc_gain)
    if min_auc_gain <= 0:
        raise ValueError("A strictly positive improvement is required")
    inputs = Path(inputs).resolve()
    manifest = json.loads((inputs / "manifest.json").read_text())
    if not manifest.get("files") or not manifest.get("collision_cache", {}).get(
        "files"
    ):
        raise ValueError("Gate requires frozen input files and collision geometry")
    for name, expected in manifest["files"].items():
        path = inputs / name
        if (
            Path(name).is_absolute()
            or not path.resolve().is_relative_to(inputs)
            or sha256(path) != expected
        ):
            raise ValueError("Gate input bytes differ from their manifest")
    if any(
        name not in manifest["files"] for name in manifest["collision_cache"]["files"]
    ):
        raise ValueError("Unbound collision geometry")
    left = read_evaluation(baseline, baseline_checkpoint, inputs)
    right = read_evaluation(candidate, candidate_checkpoint, inputs)
    for key in (
        "user.dex3_runtime_patches_sha256",
        "user.dex3_image_context_sha256",
        "user.policy_schema_json",
    ):
        if left[2][key] != right[2][key]:
            raise ValueError("Compared checkpoint runtime contracts differ")
    if experiment(left[2]["user.resolved_config_json"]) != experiment(
        right[2]["user.resolved_config_json"]
    ):
        raise ValueError(
            "Compared experiments differ; establish a separate controlled baseline"
        )
    for key in (
        "world_count",
        "step_count",
        "sim_control_fps",
        "sim_physics_fps",
        "object_orientation_threshold_rad",
        "object_position_threshold_m",
    ):
        if key not in left[1] or left[1][key] != right[1].get(key):
            raise ValueError("Evaluation coverage or thresholds differ")
    if right[1]["environment_steps"] <= left[1]["environment_steps"]:
        raise ValueError("Candidate must contain additional training steps")
    # Preserve all of the object tracking metrics; a rise in reward alone never
    # authorizes another training chunk. These are local allocation thresholds.
    directions = {
        "add_auc": 1,
        "chord_sr": 1,
        "spider_sr_uncentered": 1,
        "maniptrans_sr": 1,
        "relative_position_error_cm": -1,
        "mppe_cm": -1,
    }
    changes, reasons = {}, []
    for metric, direction in directions.items():
        a, b = (
            finite_number(left[1]["metrics"][metric]),
            finite_number(right[1]["metrics"][metric]),
        )
        changes[metric] = {"baseline": a, "candidate": b, "delta": b - a}
        if direction * (b - a) < -1e-9:
            reasons.append(metric + " regressed")
    if changes["add_auc"]["delta"] < min_auc_gain:
        reasons.append("AUC improvement below the configured allocation threshold")
    return {
        "schema": "v2d.compute-evaluation-gate.v1",
        "continue_training": not reasons,
        "reasons": reasons,
        "min_auc_gain": min_auc_gain,
        "metrics": changes,
        "baseline_checkpoint_sha256": sha256(baseline_checkpoint),
        "candidate_checkpoint_sha256": sha256(candidate_checkpoint),
        "input_manifest_sha256": sha256(Path(inputs) / "manifest.json"),
        "baseline_report_sha256": sha256(Path(baseline) / "evaluation-report.json"),
        "candidate_report_sha256": sha256(Path(candidate) / "evaluation-report.json"),
        "independent_confirmation_required": True,
        "official_score_claimed": False,
        "physical_task_accepted": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline",
        "candidate",
        "baseline-checkpoint",
        "candidate-checkpoint",
        "inputs",
        "out",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--min-auc-gain", type=float, default=0.005)
    a = p.parse_args()
    result = compare(
        a.baseline,
        a.candidate,
        a.baseline_checkpoint,
        a.candidate_checkpoint,
        a.inputs,
        a.min_auc_gain,
    )
    with a.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["continue_training"] else 3)
