"""Verify and restore both private pilot archives into a fresh workspace."""

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return p


def verify_archive(path):
    with tarfile.open(fileobj=io.BytesIO(path.read_bytes()), mode="r:gz") as source:
        members = source.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError("Duplicate archive members")
        for m in members:
            safe_name(m.name)
            if not m.isfile():
                raise ValueError(f"Non-file archive member: {m.name}")
        manifest = json.load(source.extractfile("artifact-manifest.json"))
        records = manifest["files"]
        if set(records) | {"artifact-manifest.json"} != {m.name for m in members}:
            raise ValueError("Archive member set does not match manifest")
        for m in members:
            if m.name == "artifact-manifest.json":
                continue
            record = records[m.name]
            if m.size != record["bytes"]:
                raise ValueError(f"Wrong archive member size: {m.name}")
            with source.extractfile(m) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != record["sha256"]:
                raise ValueError(f"Wrong archive member hash: {m.name}")
    return manifest, len(members)


def baseline_location(name):
    p = safe_name(name)
    first, *tail = p.parts
    if first in ("episode2-bundle-v2", "retarget-trial3", "penetration-trial2"):
        return PurePosixPath("runs/issue8-validation") / p
    if first == "source":
        return PurePosixPath("issue8-pilot", *tail)
    if first == "evidence":
        if tail[0] == "scripts":
            tail = tail[1:]
        return PurePosixPath("runs/issue8-validation", *tail)
    if first == "simulator":
        return PurePosixPath("simulator/replay9", *tail)
    if first == "issue7-review":
        if tail == ["REVIEW.md"]:
            return PurePosixPath("issue7-refinement/docs/team/issue7/REVIEW.md")
        if tail == ["track3_refinement_cpu.yml"]:
            return PurePosixPath(
                "issue7-refinement/.github/workflows/track3_refinement_cpu.yml"
            )
    raise ValueError(f"Unrecognized baseline path: {name}")


def restore(path, out, baseline):
    with tarfile.open(fileobj=io.BytesIO(path.read_bytes()), mode="r:gz") as source:
        for member in source.getmembers():
            if member.name == "artifact-manifest.json":
                name = (
                    "runs/issue8-validation/restored-baseline-artifact-manifest.json"
                    if baseline
                    else "runs/issue8-validation/restored-derived-artifact-manifest.json"
                )
                relative = safe_name(name)
            else:
                relative = (
                    baseline_location(member.name)
                    if baseline
                    else safe_name(member.name)
                )
            target = (out / Path(*relative.parts)).resolve()
            if not target.is_relative_to(out):
                raise ValueError("Restoration escapes intended workspace")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, target.open("wb") as dest:
                shutil.copyfileobj(stream, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("Use a fresh restoration workspace")
    baseline, nbase = verify_archive(args.baseline)
    supplement, nsupp = verify_archive(args.supplement)
    baseline_sha = sha(args.baseline)
    if baseline_sha != supplement["requires_baseline_archive"]["sha256"]:
        raise ValueError("Wrong baseline archive")
    out.mkdir(parents=True)
    restore(args.baseline, out, True)
    restore(args.supplement, out, False)
    # Verify the final overlay as well as every pre-restoration archive byte.
    for name, record in supplement["files"].items():
        target = out / Path(*safe_name(name).parts)
        if sha(target) != record["sha256"]:
            raise ValueError(f"Wrong restored supplement byte: {name}")
    report = {
        "status": "both archives and restored overlay byte-verified",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(out),
        "baseline_sha256": baseline_sha,
        "supplement_sha256": sha(args.supplement),
        "baseline_verified_members": nbase,
        "supplement_verified_members": nsupp,
        "physical_manipulation_success": False,
    }
    (out / "restoration-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
