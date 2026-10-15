"""Check a completed wrist diagnostic against thresholds frozen before GPU runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

THRESHOLDS = {
    "position_p95_m": 0.02,
    "position_max_m": 0.10,
    "rotation_p95_deg": 10.0,
    "rotation_max_deg": 45.0,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run: Path) -> dict:
    """Verify complete numerical records and report controller-only acceptance."""
    report = json.loads((run / "report.json").read_text())
    states_file = run / "states.json"
    if digest(states_file) != report["state_sha256"]:
        raise ValueError("Simulator states changed")
    if digest(run / "supported-task.mp4") != report["video_sha256"]:
        raise ValueError("Simulator video changed")
    if (run.parent / (run.name + ".failure.json")).exists():
        raise ValueError("Simulator failure sidecar exists")
    states = json.loads(states_file.read_text())
    expected_steps = list(range(report["simulation_steps"] + 1))
    failures = []
    sides = {}
    if not report["gravity_and_contact_controls_pass"]:
        failures.append("gravity/contact controls failed")
    if (
        report["object_teleports_after_initialization"] != 0
        or report["external_object_forces"]
    ):
        failures.append("object motion was externally assisted")
    rendered = states["rendered_source_frames"]
    if len(rendered) != report["source_frames"] or len(rendered) < 2:
        raise ValueError("Missing source-frame coverage")
    if any(
        b["source_frame"] <= a["source_frame"] or b["sim_time_s"] <= a["sim_time_s"]
        for a, b in zip(rendered, rendered[1:])
    ):
        raise ValueError("Source or simulation frame axis is not increasing")
    for side in ("left", "right"):
        rows = [r for r in states["hands"] if r["side"] == side]
        if [r["step"] for r in rows] != expected_steps:
            raise ValueError("Missing/duplicate hand physics records")
        actual = np.array([r["actual_root_pose_stage_xyzw"] for r in rows])
        target = np.array([r["target_root_pose_stage_xyzw"] for r in rows])
        if not np.isfinite(actual).all() or not np.isfinite(target).all():
            raise ValueError("Nonfinite physical poses")
        if any(
            not np.allclose(np.linalg.norm(pose[:, 3:], axis=-1), 1, atol=1e-4, rtol=0)
            for pose in (actual, target)
        ):
            raise ValueError("Physical quaternion norm differs from unity")
        error = np.linalg.norm(actual[:, :3] - target[:, :3], axis=-1)
        angle = np.degrees(
            (
                Rotation.from_quat(target[:, 3:])
                * Rotation.from_quat(actual[:, 3:]).inv()
            ).magnitude()
        )
        measured = {
            "position_p95_m": float(np.quantile(error, 0.95)),
            "position_max_m": float(error.max()),
            "rotation_p95_deg": float(np.quantile(angle, 0.95)),
            "rotation_max_deg": float(angle.max()),
        }
        sides[side] = measured
        for key, threshold in THRESHOLDS.items():
            if measured[key] > threshold:
                failures.append(f"{side} {key} exceeds {threshold}")
    return {
        "schema": "v2d.wrist-tracking-gate.v1",
        "report_sha256": digest(run / "report.json"),
        "state_sha256": report["state_sha256"],
        "inputs_manifest_sha256": report["inputs_manifest_sha256"],
        "thresholds": THRESHOLDS,
        "wrist_diagnostic": report.get("wrist_diagnostic", {}),
        "complete_source_episode": report.get("complete_source_episode", False),
        "per_side": sides,
        "failures": failures,
        "controller_tracking_pass": not failures,
        "physical_task_accepted": False,
        "learned_policy_evaluated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    if not result["controller_tracking_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
