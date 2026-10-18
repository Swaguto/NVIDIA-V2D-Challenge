"""Export the committed host/CPU code needed between reconstruction and PPO.

Robot mesh bytes stay in a separate private, hash-locked mount. The training
runtime is supplied by its immutable image, rather than copied from a checkout.
"""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIXES = [
    "scripts/track3/prepare_scored_case.py",
    "scripts/track3/run_scored_policy.py",
    "scripts/track3/continue_scored_policy.py",
    "scripts/reproduction/_dex3_static_load.py",
    "scripts/dex3_training/prepare_window.py",
    "scripts/dex3_training/check_native_scene.py",
    "scripts/dex3_training/audit_initial_contacts.py",
    "scripts/dex3_training/audit_reference_contacts.py",
    "robotic_grounding/scripts/retarget/track3_to_dex3.py",
    "robotic_grounding/scripts/generate_rigid_urdfs.py",
    "robotic_grounding/source/robotic_grounding/robotic_grounding",
]


def create(out):
    if out.exists():
        raise FileExistsError(out)
    if subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", *PREFIXES],
        text=True,
    ).strip():
        raise ValueError("Commit selected policy source before exporting it")
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    payload = subprocess.check_output(
        ["git", "-C", str(ROOT), "archive", commit, *PREFIXES]
    )
    out.mkdir(parents=True)
    # Docker cannot create a nested bind target inside a read-only parent mount.
    # This directory is empty; the actual robot closure is mounted separately.
    (
        out / "code/robotic_grounding/source/robotic_grounding/robotic_grounding/assets"
    ).mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive:
            path = Path(member.name)
            if not member.isfile() or path.suffix != ".py" or "assets" in path.parts:
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe source path")
            dest = out / "code" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.extractfile(member).read())
    manifest = {
        "schema": "v2d.scored-policy-context.v1",
        "source_commit": commit,
        "private_inputs_included": False,
        "robot_assets_included": False,
        "model_weights_included": False,
        "credentials_included": False,
        "files": {
            p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    }
    (out / "context-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = create(args.out)
    print(
        json.dumps(
            {"source_commit": result["source_commit"], "files": len(result["files"])}
        )
    )
