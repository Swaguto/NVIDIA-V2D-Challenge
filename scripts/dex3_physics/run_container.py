"""Execute one explicit, hash-bound physics validation stage inside the review image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reproduction"))
from reproduce_frozen_dex3_replay import restore_methods
from audit_wrist_run import audit

PACK_SHA = "759b6e9fa642b47ec24306969688cc1ce15ead4cb0b80e4f8d086b727b93c680"
PLANE_SHA = "38a6bf97df7def496bed7773e265c894e77da6bb5521d3077bf43330cbd3166f"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["baseline-free", "candidate-free", "candidate-full"],
        required=True,
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = Path("/outputs") / args.stage
    if out.exists():
        raise FileExistsError("Choose a fresh output root for this experiment")
    candidate_file = ROOT / "scripts/reproduction/_dex3_wrist_control.py"
    if args.stage == "candidate-full":
        free = Path("/outputs/candidate-free")
        gate = json.loads((free / "wrist-gate.json").read_text())
        invocation = json.loads((free / "invocation.json").read_text())
        checked_gate = audit(free / "run/simulator")
        if (
            not gate["controller_tracking_pass"]
            or gate != checked_gate
            or invocation["executor_sha256"]
            != digest(ROOT / "scripts/reproduction/_dex3_static_load.py")
            or gate["report_sha256"] != digest(free / "run/simulator/report.json")
            or gate["wrist_diagnostic"]["candidate_controller_sha256"]
            != digest(candidate_file)
            or gate["wrist_diagnostic"]["physics_dt_s"] != 0.01
            or gate["wrist_diagnostic"]["controller"] != "implicit_pd"
            or not gate["wrist_diagnostic"]["free_space"]
        ):
            raise ValueError(
                "Full-task candidate requires a passing free-space gate with this controller"
            )
    out.mkdir()
    evidence = ROOT / "docs/team/reproduction/dex3-motion-replay"
    restore_methods(
        evidence / "supported-task-measurements-v35.tar.gz",
        evidence / "supported-task-fixtures-v35.json",
        out / "evidence",
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/reproduction/reproduce_frozen_dex3_replay.py"),
        "--archive",
        "/physics-inputs/input-pack.tar.gz",
        "--sha256",
        PACK_SHA,
        "--out",
        str(out / "run"),
        "--supported-task-check",
        "--task-motion-evaluator",
        str(out / "evidence/motion-evaluator.py"),
        "--support-plane",
        str(evidence / "support-inferred-support-plane-report-v1.json"),
        "--support-plane-sha256",
        PLANE_SHA,
        "--wrist-controller",
        "project_pd" if args.stage == "baseline-free" else "implicit_pd",
        "--physics-dt",
        "0.01",
    ]
    if args.stage != "candidate-full":
        command += ["--free-space", "--source-duration-s", "2"]
    if args.prepare_only:
        command += ["--check-only"]
    else:
        if (
            os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES"
            or os.environ.get("ACCEPT_EULA") != "Y"
        ):
            raise RuntimeError(
                "Read and accept the NVIDIA license before simulator execution"
            )
        command += ["--runtime-python", sys.executable, "--lab", "/opt/IsaacLab"]
    receipt = {
        "stage": args.stage,
        "prepare_only": args.prepare_only,
        "command": command,
        "candidate_controller_sha256": digest(candidate_file),
        "executor_sha256": digest(ROOT / "scripts/reproduction/_dex3_static_load.py"),
        "physical_task_accepted": False,
    }
    (out / "invocation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if not args.prepare_only:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/dex3_physics/check_gpu.py"),
                "--out",
                str(out / "gpu-preflight.json"),
            ],
            check=True,
            env=env,
        )
    started = time.time()
    try:
        subprocess.run(command, check=True, env=env)
    finally:
        log_root = Path("/opt/sim/lib/python3.12/site-packages/isaacsim/kit/logs")
        for log in log_root.rglob("*.log"):
            if log.stat().st_mtime >= started:
                target = out / "sdk-logs" / log.relative_to(log_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log, target)
    if not args.prepare_only:
        video = out / "run/simulator/supported-task.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True
        )
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "json",
                    str(video),
                ],
                text=True,
            )
        )
        report = json.loads((out / "run/simulator/report.json").read_text())
        if int(probe["streams"][0]["nb_read_frames"]) != report["source_frames"]:
            raise ValueError("Actual decoded physics video frame count differs")
        (out / "video-verification.json").write_text(json.dumps(probe, indent=2) + "\n")
        if args.stage == "candidate-full":
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/reproduction/audit_supported_dex3_task.py"),
                    "--case",
                    str(out / "run/case"),
                    "--states",
                    str(out / "run/simulator/states.json"),
                    "--states-sha256",
                    report["state_sha256"],
                    "--video",
                    str(video),
                    "--video-sha256",
                    report["video_sha256"],
                    "--annotation-archive",
                    str(evidence / "original-CAD-baseline-audits-v1.tar.gz"),
                    "--out",
                    str(out / "independent-full-task-audit"),
                ],
                check=True,
                env=env,
            )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/reproduction/audit_wrist_run.py"),
                "--run",
                str(out / "run/simulator"),
                "--out",
                str(out / "wrist-gate.json"),
            ],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    main()
