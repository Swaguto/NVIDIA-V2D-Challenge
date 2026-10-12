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
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--lab", type=Path)
    args = parser.parse_args()
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
