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
        choices=[
            "baseline-free",
            "candidate-free",
            "candidate-full",
            "static-load",
            "grasp-seed",
            "grasp-acquisition",
        ],
        required=True,
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--grasp-seed", type=Path)
    parser.add_argument(
        "--candidate-controller",
        choices=["implicit_pd", "implicit_pd_ff", "implicit_pid_ff"],
        default="implicit_pd",
    )
    args = parser.parse_args()
    static_stage = args.stage in {"static-load", "grasp-seed", "grasp-acquisition"}
    acquisition_stage = args.stage == "grasp-acquisition"
    if (args.stage in {"grasp-seed", "grasp-acquisition"}) != (
        args.grasp_seed is not None
    ):
        parser.error(
            "--stage grasp-seed requires --grasp-seed; other stages preserve source targets"
        )
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
            or gate["wrist_diagnostic"]["controller"] != args.candidate_controller
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
        "--support-plane",
        str(evidence / "support-inferred-support-plane-report-v1.json"),
        "--support-plane-sha256",
        PLANE_SHA,
    ]
    if static_stage:
        command += ["--static-load-check"]
        if args.grasp_seed:
            command += [
                "--static-grasp-seed",
                str(args.grasp_seed),
                "--static-grasp-seed-sha256",
                digest(args.grasp_seed),
            ]
            if acquisition_stage:
                command += ["--controlled-acquisition"]
    else:
        command += [
            "--supported-task-check",
            "--task-motion-evaluator",
            str(out / "evidence/motion-evaluator.py"),
            "--wrist-controller",
            "project_pd"
            if args.stage == "baseline-free"
            else args.candidate_controller,
            "--physics-dt",
            "0.01",
        ]
    if args.stage in {"baseline-free", "candidate-free"}:
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
        video = (
            out
            / "run/simulator"
            / ("static-load.mp4" if static_stage else "supported-task.mp4")
        )
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
        expected_frames = (
            (31 if acquisition_stage else 21)
            if static_stage
            else report["source_frames"]
        )
        if int(probe["streams"][0]["nb_read_frames"]) != expected_frames:
            raise ValueError("Actual decoded physics video frame count differs")
        (out / "video-verification.json").write_text(json.dumps(probe, indent=2) + "\n")
        if static_stage:
            run = out / "run/simulator"
            states = json.loads((run / "states.json").read_text())
            if (
                digest(run / "states.json") != report["states_sha256"]
                or digest(video) != report["video_sha256"]
            ):
                raise ValueError("Static load artifact hashes differ")
            if not report["gravity_and_contact_controls_pass"]:
                raise ValueError("Static load physics controls failed")
            if (
                len(states["records"]) != (604 if acquisition_stage else 404)
                or len(states["controls"]) != (353 if acquisition_stage else 202)
                or len(report["summaries"]) != 4
            ):
                raise ValueError("Static load record counts differ")
            for sample in range(4):
                if [
                    r["step"] for r in states["records"] if r["sample_index"] == sample
                ] != list(range(151 if acquisition_stage else 101)):
                    raise ValueError("Static load state coverage differs")
            for control in range(2):
                if [
                    r["step"] for r in states["controls"] if r["control"] == control
                ] != list(range(101)):
                    raise ValueError("Static load control coverage differs")
            if acquisition_stage and [
                r["step"] for r in states["controls"] if r["control"] == 2
            ] != list(range(151)):
                raise ValueError("Temporary support release control coverage differs")
            cooked = report["cooked_collider_summary"]
            if {r["asset"] for r in cooked if r["asset"].startswith("object-")} != {
                f"object-{i}" for i in range(4)
            } or any(
                r["convex_count"] <= 1
                for r in cooked
                if r["asset"].startswith("object-")
            ):
                raise ValueError(
                    "Static load object convex decomposition was not established"
                )
            gate = {
                "schema": "v2d.static-load-gate.v1",
                "report_sha256": digest(run / "report.json"),
                "states_sha256": report["states_sha256"],
                "video_sha256": report["video_sha256"],
                "gravity_and_contact_controls_pass": True,
                "all_four_samples_held_within_10mm_for_1s": all(
                    r["held_within_10mm_for_1s_diagnostic_only"]
                    for r in report["summaries"]
                ),
                "physical_task_accepted": False,
                "learned_policy_evaluated": False,
                "controlled_acquisition": acquisition_stage,
            }
            if acquisition_stage:
                releases = report["support_release_readback"]
                if (
                    not report["controlled_acquisition"]
                    or len(releases) != 5
                    or report["load_start_s"] != 0.5
                    or report["duration_s"] != 1.5
                    or any(
                        r["step"] != 50
                        or not r["before_enabled"]
                        or r["after_enabled"]
                        or r["visibility"] != "invisible"
                        for r in releases
                    )
                    or not report["controls"]["temporary_support_release_control_pass"]
                ):
                    raise ValueError(
                        "Controlled acquisition did not remove all temporary supports"
                    )
                gate["temporary_supports_removed_before_load"] = True
            if args.grasp_seed:
                changed = report["experimental_grasp_seed_samples"]
                selected = [r for r in report["summaries"] if r["sample"] in changed]
                if (
                    not changed
                    or len(selected) != len(changed)
                    or report["grasp_seed_sha256"] != digest(args.grasp_seed)
                ):
                    raise ValueError(
                        "Experimental grasp seed results are incomplete or mismatched"
                    )
                gate["experimental_grasp_seed_samples"] = changed
                gate["changed_seeds_held_within_10mm_for_1s"] = all(
                    r["held_within_10mm_for_1s_diagnostic_only"] for r in selected
                )
            (out / "static-load-gate.json").write_text(
                json.dumps(gate, indent=2) + "\n"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/dex3_physics/package_static_preview.py"),
                    "--run",
                    str(run),
                ],
                check=True,
            )
            print(json.dumps(gate, indent=2))
            if not gate.get(
                "changed_seeds_held_within_10mm_for_1s",
                gate["all_four_samples_held_within_10mm_for_1s"],
            ):
                raise SystemExit(2)
            return
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
