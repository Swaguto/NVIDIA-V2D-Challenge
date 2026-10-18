"""Screen repeated local readbacks of declared single-factor PPO experiments."""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "compute"))
from evaluation_gate import read_evaluation, sha256
from v2d_dex3_flashchord.controlled_experiment import normalize
from v2d_dex3_flashchord.resume_contract import experiment_contract

DIRECTIONS = {
    "add_auc": 1,
    "chord_sr": 1,
    "spider_sr_uncentered": 1,
    "maniptrans_sr": 1,
    "relative_position_error_cm": -1,
    "mppe_cm": -1,
}


def summarize(repeats):
    if len(repeats) < 3:
        raise ValueError("At least three complete repeated readbacks are required")
    import math

    result = {}
    for name in DIRECTIONS:
        values = [r[name] for r in repeats]
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in values
        ):
            raise ValueError("Nonfinite or missing repeated metric")
        result[name] = {
            "mean": statistics.mean(values),
            "minimum": min(values),
            "maximum": max(values),
            "repeats": values,
        }
    return result


def screen(parent, control, candidate):
    reasons = []
    # An empirical variability guard, not a confidence interval or significance test.
    noise = max(
        parent["add_auc"]["maximum"] - parent["add_auc"]["minimum"],
        control["add_auc"]["maximum"] - control["add_auc"]["minimum"],
        0.005,
    )
    if candidate["add_auc"]["mean"] - parent["add_auc"]["mean"] < noise:
        reasons.append(
            "AUC gain over the retained parent does not exceed observed control variability"
        )
    if candidate["add_auc"]["mean"] - control["add_auc"]["mean"] < 0.005:
        reasons.append("AUC gain over matched trained control is below 0.005")
    for metric, direction in DIRECTIONS.items():
        if direction * (candidate[metric]["mean"] - parent[metric]["mean"]) < -1e-9:
            reasons.append(metric + " regressed against parent repeat mean")
    return {
        "eligible_for_independent_confirmation": not reasons,
        "reasons": reasons,
        "empirical_auc_variability_guard": noise,
        "automatic_long_continuation": False,
    }


def analyze(root, inputs, parent_checkpoint):
    checkpoints = {"parent": parent_checkpoint}
    checkpoints.update(
        {
            name: root / name / "ppo/model_2100.safetensors"
            for name in ("control", "half-cws", "short-settling")
        }
    )
    results, parent_meta, coverage = {}, None, None
    for name, checkpoint in checkpoints.items():
        repeats, hashes = [], []
        for repeat in range(1, 4):
            path = root / name / f"readback-{repeat}"
            _, evaluation, meta = read_evaluation(path, checkpoint, inputs)
            contract = {
                key: evaluation[key]
                for key in (
                    "world_count",
                    "step_count",
                    "sim_control_fps",
                    "sim_physics_fps",
                    "object_orientation_threshold_rad",
                    "object_position_threshold_m",
                )
            }
            if coverage is not None and contract != coverage:
                raise ValueError("Repeated evaluation coverage/thresholds differ")
            coverage = contract
            if name == "parent":
                parent_meta = meta
            else:
                old = json.loads(parent_meta["user.resolved_config_json"])
                config = json.loads(meta["user.resolved_config_json"])
                normalized, factor = normalize(old, config)
                if experiment_contract(old) != experiment_contract(normalized):
                    raise ValueError(
                        "Trial contains an undeclared configuration change"
                    )
                lineage = json.loads(meta["user.dex3_resume_lineage_json"])
                if (
                    lineage["parent_checkpoint_sha256"] != sha256(parent_checkpoint)
                    or lineage["controlled_experiment"] != factor
                    or meta["user.dex3_runtime_patches_sha256"]
                    != parent_meta["user.dex3_runtime_patches_sha256"]
                    or meta["user.policy_schema_json"]
                    != parent_meta["user.policy_schema_json"]
                ):
                    raise ValueError("Trial lineage/runtime contracts differ")
            repeats.append(evaluation["metrics"])
            hashes.append(sha256(path / "evaluation-report.json"))
        results[name] = {
            "metrics": summarize(repeats),
            "checkpoint_sha256": sha256(checkpoint),
            "evaluation_report_sha256": hashes,
        }
        audit = json.loads((root / name / "audit.json").read_text())
        grasps = audit["grasps"]
        results[name]["all_first_readback_grasp_gates_passed"] = bool(grasps) and all(
            g["supported_retention_with_sampled_depth_within_2mm"]
            and g["maximum_source_position_error_m"] <= 0.03
            and g["maximum_source_orientation_error_deg"] <= 15
            for g in grasps
        )
    for name in ("half-cws", "short-settling"):
        results[name]["screen"] = screen(
            results["parent"]["metrics"],
            results["control"]["metrics"],
            results[name]["metrics"],
        )
    return {
        "schema": "v2d.repeated-controlled-experiments.v1",
        "trials": results,
        "evaluation_coverage": coverage,
        "official_scores_claimed": False,
        "physical_task_accepted": False,
        "automatic_long_continuation": False,
        "comparison_policy": "Descriptive repeated local readbacks, not hidden scores or statistical significance; independent confirmation still required",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "inputs", "parent-checkpoint", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.inputs, args.parent_checkpoint)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {name: trial.get("screen") for name, trial in result["trials"].items()}
        )
    )
