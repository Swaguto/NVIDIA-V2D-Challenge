"""Measure sampled native wrist motion alongside the independent grasp audit.

This uses control-rate readbacks. It cannot measure physics-step jitter above
the control Nyquist frequency, and a quiet hand does not establish a grasp.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dex3_physics"))
from audit_wrist_jitter import motion_residual


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run, inputs, grasp_audit, start_s=5.5, end_s=7.5):
    report = json.loads((run / "report.json").read_text())
    manifest = json.loads((inputs / "manifest.json").read_text())
    if digest(inputs / "manifest.json") != report["source_manifest_sha256"]:
        raise ValueError("Readback and input manifest differ")
    if digest(inputs / "targets.npz") != manifest["files"]["targets.npz"]:
        raise ValueError("Reference target hash mismatch")
    retention = json.loads(grasp_audit.read_text())
    if retention["source_report_sha256"] != digest(run / "report.json"):
        raise ValueError("Grasp audit belongs to another readback")
    with np.load(inputs / "targets.npz") as source:
        times = source["timestamps_s"].astype(float)
        target = source["robot_qpos"][:, :, :7].copy()
    with np.load(run / "native-state-readback.npz") as state:
        poses = state["body_poses"].copy()
    if len(poses) != len(times) or report["source_motion_speed"] != 1.0:
        raise ValueError("Need complete source-speed readback")
    if not np.isclose(1 / np.median(np.diff(times)), report["control_fps"]):
        raise ValueError("Readback and reference clocks differ")
    labels = json.loads((run / "body-labels.json").read_text())
    if len(labels) != poses.shape[1]:
        raise ValueError("Readback labels and body count differ")
    selected = (times >= start_s) & (times < end_s)
    hands = {}
    for hand, side in enumerate(["left", "right"]):
        indices = [
            i for i, name in enumerate(labels)
            if name == f"dex3_{side}/{side}_hand_palm_link"
        ]
        if not indices:
            raise ValueError(f"No native palm readback for {side}")
        hands[side] = dict(
            reference=motion_residual(times[selected], target[selected, hand]),
            worlds=[
                dict(world=world, **motion_residual(times[selected], poses[selected, index]))
                for world, index in enumerate(indices)
            ],
            complete_grasp_diagnostics=[
                g for g in retention["grasps"] if g["sample"].startswith(side + "-")
            ],
        )
    return dict(
        schema="v2d.native-policy-sampled-wrist-jitter.v1",
        source_report_sha256=digest(run / "report.json"),
        state_sha256=digest(run / "native-state-readback.npz"),
        labels_sha256=digest(run / "body-labels.json"),
        source_manifest_sha256=report["source_manifest_sha256"],
        grasp_audit_sha256=digest(grasp_audit),
        analysis_source_sha256=digest(Path(__file__)),
        source_interval_s=[start_s, end_s],
        sampled_control_fps=report["control_fps"],
        sampled_nyquist_hz=report["control_fps"] / 2,
        physics_step_jitter_certified=False,
        physical_task_accepted=False,
        hands=hands,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--grasp-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.inputs, args.grasp_audit)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
