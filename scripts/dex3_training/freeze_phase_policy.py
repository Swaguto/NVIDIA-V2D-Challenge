"""Bind an accepted native phase-policy prior to a copy of its original inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(inputs, plan_path, replay_audit, out):
    manifest = json.loads((inputs / "manifest.json").read_text())
    plan = json.loads(plan_path.read_text())
    audit = json.loads(replay_audit.read_text())
    source_sha = sha(inputs / "manifest.json")
    if (
        plan["input_manifest_sha256"] != source_sha
        or audit["source_manifest_sha256"] != source_sha
    ):
        raise ValueError("Fitted policy and independent replay inputs differ")
    if (
        not plan.get("fit_reports")
        or set(plan["fit_reports"]) != set(plan["phase_actions"])
        or audit.get("fitted_phase_policy_sha256") != sha(plan_path)
        or audit.get("native_motor_caps_verified") is not True
    ):
        raise ValueError(
            "Need the exact fitted policy and independent native motor verification"
        )
    for name, row in plan["fit_reports"].items():
        measured = [g for g in audit["grasps"] if g["sample"] == name]
        if (
            not row["robust_phase_pass"]
            or len(measured) < 2
            or not all(
                g["supported_retention_with_sampled_depth_within_2mm"]
                and g["maximum_source_position_error_m"] <= 0.03
                and g["maximum_source_orientation_error_deg"] <= 15
                for g in measured
            )
        ):
            raise ValueError(
                "Fitted phase has not passed independent measured retention/depth checks: "
                + name
            )
    for name, expected in manifest["files"].items():
        path = (inputs / name).resolve()
        if not path.is_relative_to(inputs.resolve()) or sha(path) != expected:
            raise ValueError("Private source inputs changed")
    if out.exists():
        raise FileExistsError(out)
    shutil.copytree(inputs, out)
    shutil.copyfile(plan_path, out / "phase-policy.json")
    manifest["phase_policy_prior_base_manifest_sha256"] = source_sha
    manifest["phase_policy_prior_independent_audit_sha256"] = sha(replay_audit)
    manifest["source_reference_arrays_changed_by_prior"] = False
    manifest["files"] = {
        p.relative_to(out).as_posix(): sha(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p != out / "manifest.json"
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "plan", "replay-audit", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    freeze(a.inputs, a.plan, a.replay_audit, a.out)
