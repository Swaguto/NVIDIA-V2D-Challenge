"""Create an allowlisted image context from a clean pinned upstream checkout."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

COMMIT = "8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef"
PREFIX = "robotic_grounding/flash_chord/"
ROOT = Path(__file__).resolve().parents[2]


def create(upstream, out):
    head = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != COMMIT:
        raise ValueError("Upstream checkout is not the official kit pin")
    paths = [
        PREFIX + name
        for name in (
            "src",
            "scripts",
            "tests",
            "pyproject.toml",
            "uv.lock",
            "README.md",
        )
    ]
    archive = subprocess.check_output(
        ["git", "-C", str(upstream), "archive", COMMIT, *paths]
    )
    if out.exists():
        raise FileExistsError(out)
    (out / "runtime").mkdir(parents=True)
    # Read committed Git objects, never untracked dataset or credential files.
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if not member.isfile() or not member.name.startswith(PREFIX):
                continue
            name = Path(member.name.removeprefix(PREFIX))
            if name.is_absolute() or ".." in name.parts or not name.parts:
                raise ValueError("Unsafe upstream archive path")
            if name.parts[0] not in {
                "src",
                "scripts",
                "tests",
            } and name.as_posix() not in {
                "pyproject.toml",
                "uv.lock",
                "README.md",
                "LICENSE",
            }:
                continue
            dest = out / "runtime" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(member).read())
    (out / "runtime/LICENSE").write_bytes(
        subprocess.check_output(
            ["git", "-C", str(upstream), "show", f"{COMMIT}:LICENSE"]
        )
    )
    shutil.copytree(
        ROOT / "scripts/dex3_training",
        out / "extension",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
    )
    shutil.copyfile(ROOT / "docker/dex3-training/Dockerfile", out / "Dockerfile")
    source_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--",
            "scripts/dex3_training",
            "docker/dex3-training",
        ],
        text=True,
    ).strip()
    manifest = dict(
        runtime_commit=COMMIT,
        source_code_commit=source_commit,
        selected_source_dirty=bool(dirty),
        private_episode_inputs_included=False,
        files={
            p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )
    (out / "context-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = create(args.upstream, args.out)
    print(
        json.dumps(
            dict(runtime_commit=result["runtime_commit"], files=len(result["files"]))
        )
    )
