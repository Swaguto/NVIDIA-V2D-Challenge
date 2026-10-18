"""Build a code-only scored perception context from committed Git objects."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIXES = [
    "scripts/track3",
    "scripts/eval",
    "scripts/dex3_review",
    "reconstruction/validation/issue6",
    "robotic_grounding/source/robotic_grounding/robotic_grounding/retarget/track3_bundle.py",
]
PREFIXES += [
    "reconstruction/modules/" + name
    for name in (
        "v2d_common",
        "v2d_mv",
        "v2d_hamer/lib",
        "v2d_issue6",
        "v2d_sam2/lib",
        "v2d_hand_alignment/lib",
        "v2d_stereo_depth/lib",
    )
]
ALLOWED = {".py", ".toml", ".json", ".yaml", ".yml", ".md", ".patch", ".txt"}


def create(out):
    if out.exists():
        raise FileExistsError(out)
    commit = subprocess.check_output(
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
            *PREFIXES,
            "docker/track3-scored",
        ],
        text=True,
    ).strip():
        raise ValueError("Commit all selected source before making the runtime context")
    data = subprocess.check_output(
        ["git", "-C", str(ROOT), "archive", commit, *PREFIXES]
    )
    out.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            path = Path(member.name)
            if not member.isfile():
                continue
            if (
                str(path).startswith("reconstruction/validation/issue6/")
                and path.parent.as_posix() != "reconstruction/validation/issue6"
            ):
                continue
            if (
                str(path).startswith("scripts/dex3_review/")
                and path.parent.as_posix() != "scripts/dex3_review"
            ):
                continue
            if path.suffix not in ALLOWED and path.name != "LICENSE":
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe source archive path")
            target = out / "code" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    for name in ["Dockerfile", "Dockerfile.geometry", "hand-requirements.txt"]:
        (out / name).write_bytes(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "show",
                    commit + ":docker/track3-scored/" + name,
                ]
            )
        )
    manifest = {
        "schema": "v2d.scored-perception-context.v1",
        "source_commit": commit,
        "private_footage_included": False,
        "model_weights_included": False,
        "credentials_included": False,
        "inference_accepted": False,
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
