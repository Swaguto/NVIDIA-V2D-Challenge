"""Freeze a controlled experiment's runtime to its checkpoint's original source.

Only committed training/lineage entry points are replaced. Physics, actions,
observations, rewards, the upstream runtime and dependency layers retain their
original hashes. The manifest records both code revisions explicitly.
"""

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OVERRIDES = {
    "extension/train.py": "scripts/dex3_training/train.py",
    "extension/v2d_dex3_flashchord/resume_contract.py": "scripts/dex3_training/v2d_dex3_flashchord/resume_contract.py",
    "extension/v2d_dex3_flashchord/controlled_experiment.py": "scripts/dex3_training/v2d_dex3_flashchord/controlled_experiment.py",
    "extension/prepare_controlled_experiments.py": "scripts/dex3_training/prepare_controlled_experiments.py",
    "Dockerfile": "docker/dex3-training/Dockerfile.repaired",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(parent_context, checkpoint, out):
    parent_context = parent_context.resolve()
    manifest_path = parent_context / "context-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with checkpoint.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= size <= min(64 * 1024 * 1024, checkpoint.stat().st_size - 8):
            raise ValueError("Invalid checkpoint header")
        meta = json.loads(stream.read(size))["__metadata__"]
    if (
        sha(manifest_path) != meta["user.dex3_image_context_sha256"]
        or manifest["selected_source_dirty"]
        or manifest["private_episode_inputs_included"]
    ):
        raise ValueError(
            "Parent context does not match the saved clean code-only image"
        )
    if out.exists():
        raise FileExistsError(out)
    source_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--",
            *OVERRIDES.values(),
            str(Path(__file__).relative_to(ROOT)),
        ],
        text=True,
    ).strip():
        raise ValueError("Commit the selected controlled experiment source first")
    verified = {}
    for name, digest in manifest["files"].items():
        path = (parent_context / name).resolve()
        if (
            not path.is_relative_to(parent_context)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or sha(path) != digest
        ):
            raise ValueError("Parent context payload hash/path differs: " + name)
        if not name.startswith(("runtime/", "extension/")) and name != "Dockerfile":
            raise ValueError("Unexpected parent source payload")
        verified[name] = path.read_bytes()
    out.mkdir(parents=True)
    provenance = {}
    for name, content in verified.items():
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        provenance[name] = manifest["source_code_commit"]
    for name, source in OVERRIDES.items():
        content = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", source_commit + ":" + source]
        )
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        provenance[name] = source_commit
    result = {
        "build_kind": "controlled-entrypoint-layer",
        "runtime_commit": manifest["runtime_commit"],
        "source_code_commit": source_commit,
        "parent_source_code_commit": manifest["source_code_commit"],
        "parent_image_context_sha256": sha(manifest_path),
        "parent_checkpoint_sha256": sha(checkpoint),
        "selected_source_dirty": False,
        "private_episode_inputs_included": False,
        "files": {
            p.relative_to(out).as_posix(): sha(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
        "payload_source_code_commits": provenance,
    }
    (out / "context-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-context", "checkpoint", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = create(args.parent_context, args.checkpoint, args.out)
    print(
        json.dumps(
            {
                "files": len(result["files"]),
                "source": result["source_code_commit"],
                "parent": result["parent_source_code_commit"],
            }
        )
    )
