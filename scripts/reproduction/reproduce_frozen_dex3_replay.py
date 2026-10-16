"""Verify a frozen Dex3 case and optionally run the measured Isaac replay.

This reproduces a saved trajectory. It does not reconstruct the source video,
refit hands, certify continuous clearance, or train a policy.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore_methods(archive, manifest_path, out):
    manifest = json.loads(manifest_path.read_text())
    if digest(archive) != manifest["archive_sha256"]:
        raise ValueError("Method snapshot hash mismatch")
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(manifest["files_sha256"]):
            raise ValueError("Unexpected or duplicate method snapshot files")
        for member in members:
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
            ):
                raise ValueError("Unsafe method snapshot member")
        out.mkdir()
        tar.extractall(out, filter="data")
    for name, expected in manifest["files_sha256"].items():
        if digest(out / name) != expected:
            raise ValueError("Extracted method hash mismatch: " + name)


def run(command, **kwargs):
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--verified-case", type=Path)
    parser.add_argument("--case-manifest-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--lab", type=Path)
    parser.add_argument(
        "--static-load-check",
        action="store_true",
        help="Prepare/run a fixed-wrist physical load diagnostic, not an episode acceptance",
    )
    parser.add_argument("--support-plane", type=Path)
    parser.add_argument("--support-plane-sha256")
    parser.add_argument("--supported-task-check", action="store_true")
    parser.add_argument("--task-motion-evaluator", type=Path)
    parser.add_argument(
        "--wrist-controller",
        choices=["project_pd", "implicit_pd", "implicit_pd_ff", "implicit_pid_ff"],
        default="project_pd",
    )
    parser.add_argument(
        "--free-space",
        action="store_true",
        help="Disable hand collisions for controller isolation only",
    )
    parser.add_argument(
        "--source-duration-s",
        type=float,
        help="Explicit partial diagnostic duration; never full-task acceptance",
    )
    parser.add_argument("--physics-dt", type=float, default=0.01)
    parser.add_argument("--static-grasp-seed", type=Path)
    parser.add_argument("--static-grasp-seed-sha256")
    parser.add_argument("--controlled-acquisition", action="store_true")
    parser.add_argument("--controlled-grip-lift", action="store_true")
    args = parser.parse_args()
    if args.controlled_grip_lift and not args.controlled_acquisition:
        parser.error("Controlled grip lift requires controlled acquisition")
    if args.static_load_check and args.supported_task_check:
        parser.error("Choose one physical diagnostic mode")
    if args.static_grasp_seed and (
        not args.static_load_check or not args.static_grasp_seed_sha256
    ):
        parser.error(
            "Experimental grasp seeds require --static-load-check and --static-grasp-seed-sha256"
        )
    if args.controlled_acquisition and (
        not args.static_load_check or not args.static_grasp_seed
    ):
        parser.error(
            "Controlled acquisition requires --static-load-check and --static-grasp-seed"
        )
    if args.supported_task_check and args.task_motion_evaluator is None:
        parser.error(
            "--supported-task-check requires the frozen --task-motion-evaluator"
        )
    if not args.supported_task_check and (
        args.wrist_controller != "project_pd"
        or args.free_space
        or args.source_duration_s is not None
        or args.physics_dt != 0.01
    ):
        parser.error("Wrist controller options require --supported-task-check")
    if (args.archive is None) == (args.verified_case is None):
        parser.error("Provide exactly one --archive or --verified-case")
    if args.archive is not None and args.sha256 is None:
        parser.error("--archive requires --sha256")
    if args.verified_case is not None and args.case_manifest_sha256 is None:
        parser.error("--verified-case requires --case-manifest-sha256")
    if (args.static_load_check or args.supported_task_check) and (
        args.support_plane is None or args.support_plane_sha256 is None
    ):
        parser.error("--static-load-check requires a hash-bound --support-plane report")
    if args.out.exists():
        raise FileExistsError("Use a fresh output directory")
    if not args.check_only and (
        sys.platform != "linux" or args.runtime_python is None or args.lab is None
    ):
        parser.error(
            "Isaac replay requires Linux, --runtime-python and --lab; --check-only works on Windows"
        )
    repo = Path(__file__).resolve().parents[2]
    snapshot = repo / "docs/team/reproduction/dex3-motion-replay"
    args.out.mkdir(parents=True)
    methods = args.out.resolve() / "methods"
    restore_methods(
        snapshot / "methods.tar.gz", snapshot / "methods-manifest.json", methods
    )
    if args.verified_case is not None:
        case = args.verified_case.resolve()
        manifest_file = case / "file-manifest.json"
        if digest(manifest_file) != args.case_manifest_sha256:
            raise ValueError("Cached case manifest hash mismatch")
        for name, expected in json.loads(manifest_file.read_text())[
            "files_sha256"
        ].items():
            path = case / name
            if not path.resolve().is_relative_to(case) or digest(path) != expected:
                raise ValueError("Changed or escaping cached case input: " + name)
    else:
        case = args.out.resolve() / "case"
        run(
            [
                sys.executable,
                methods / "verify_generated_sim_package_v1.py",
                "--archive",
                args.archive.resolve(),
                "--sha256",
                args.sha256,
                "--out",
                case,
            ]
        )
    if args.static_load_check or args.supported_task_check:
        from _dex3_static_load import prepare

        prepared = prepare(
            case,
            repo,
            args.support_plane.resolve(),
            args.support_plane_sha256,
            args.out.resolve() / "static-load-inputs",
            task_motion=args.task_motion_evaluator.resolve()
            if args.supported_task_check
            else None,
            controller_mode=args.wrist_controller,
            free_space=args.free_space,
            source_duration_s=args.source_duration_s,
            physics_dt=args.physics_dt,
            grasp_seed=args.static_grasp_seed,
            grasp_seed_sha256=args.static_grasp_seed_sha256,
            controlled_acquisition=args.controlled_acquisition,
            controlled_grip_lift=args.controlled_grip_lift,
        )
        print("Prepared source-derived static grasps: " + str(len(prepared["samples"])))
    if args.check_only:
        print(
            "Frozen input and method bytes verified. Reconstruction and simulator execution were not run."
        )
        return
    runtime = args.runtime_python.resolve()
    lab = args.lab.resolve()
    env = dict(
        os.environ,
        OMNI_KIT_ACCEPT_EULA="YES",
        OMNI_KIT_ALLOW_ROOT="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    run(
        [
            runtime,
            methods / "setup_sim6.py",
            "--check",
            "--lock",
            methods / "sim6-runtime-lock.txt",
            "--lab",
            lab,
        ],
        env=env,
    )
    replay = args.out.resolve() / "simulator"
    if args.static_load_check or args.supported_task_check:
        physical_inputs = args.out.resolve() / "static-load-inputs"
        run(
            [
                runtime,
                repo / "scripts/reproduction/_dex3_static_load.py",
                "--case",
                case,
                "--repo",
                repo,
                "--inputs",
                physical_inputs,
                "--manifest-sha256",
                digest(physical_inputs / "manifest.json"),
                "--out",
                replay,
                "--headless",
                "--enable_cameras",
                "--device",
                "cpu",
                "--rendering_mode",
                "performance",
            ],
            cwd=lab,
            env=env,
        )
        if (replay.parent / (replay.name + ".failure.json")).exists() or not (
            replay / "report.json"
        ).is_file():
            raise RuntimeError(
                "Static physics execution failed or did not produce a result; inspect its failure report"
            )
        physical_report = json.loads((replay / "report.json").read_text())
        if not physical_report["gravity_and_contact_controls_pass"]:
            raise RuntimeError(
                "Static physics controls failed; grasp outcomes are not accepted"
            )
        print(
            "Static physical diagnostic complete. It does not accept the complete task or score a policy."
        )
        return
    run(
        [
            runtime,
            case / "scripts/replay_generated_dex3_source_camera_sim6_v1.py",
            "--source",
            case / "source/robotic_grounding",
            "--bundle",
            case / "bundle",
            "--retarget",
            case / "retarget",
            "--proposal",
            case / "proposal",
            "--scene-audit",
            case / "scene-audit",
            "--out",
            replay,
            "--width",
            "1280",
            "--height",
            "800",
            "--headless",
            "--enable_cameras",
            "--device",
            "cpu",
            "--rendering_mode",
            "performance",
        ],
        cwd=lab,
        env=env,
    )
    run(
        [
            runtime,
            methods / "audit_generated_dex3_sim_replay_v1.py",
            "--bundle",
            case / "bundle",
            "--baseline",
            case / "retarget",
            "--proposal",
            case / "proposal",
            "--replay",
            replay,
            "--out",
            args.out.resolve() / "independent-audit",
        ],
        env=env,
    )
    print(
        "Replay and state/camera/video audit complete. Inspect visual continuity separately; physical targets remain unaccepted."
    )


if __name__ == "__main__":
    main()
