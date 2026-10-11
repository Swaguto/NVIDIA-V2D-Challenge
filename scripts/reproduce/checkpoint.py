"""Restore hash-pinned private checkpoints and replay their CPU measurements."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]

# Small geometry queries run faster without excessive BLAS worker startup.
# Respect a caller's explicit setting; apply before importing NumPy/SciPy.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
    ):
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def restore(archive, expected_sha, out):
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    if archive.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("Archive exceeds the supported 256 MiB size")
    payload = archive.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("Archive SHA256 differs from the published catalog")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as source:
        members = source.getmembers()
        if len({str(safe_name(m.name)) for m in members}) != len(members):
            raise ValueError("Duplicate archive paths")
        for member in members:
            safe_name(member.name)
            if not member.isfile():
                raise ValueError("Only ordinary files are allowed")
        manifest = json.load(source.extractfile("artifact-manifest.json"))
        if set(manifest["files"]) | {"artifact-manifest.json"} != {
            m.name for m in members
        }:
            raise ValueError("Archive membership differs from manifest")
        for member in members:
            if member.name == "artifact-manifest.json":
                continue
            record = manifest["files"][member.name]
            with source.extractfile(member) as stream:
                digest = hashlib.sha256(stream.read()).hexdigest()
            if member.size != record["bytes"] or digest != record["sha256"]:
                raise ValueError(f"Corrupt archive member: {member.name}")
        # Verify every byte before reserving or writing the destination.
        out.mkdir(parents=True)
        for member in members:
            target = out.joinpath(*safe_name(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, target.open("wb") as dest:
                shutil.copyfileobj(stream, dest)
    return manifest


def compare(expected, actual, path="", tolerance=1e-8):
    """Exact masks/counts/unknowns; bounded floating point differences."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError(f"Different fields: {path}")
        for key in expected:
            compare(expected[key], actual[key], f"{path}/{key}", tolerance)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"Different list shape: {path}")
        for i, value in enumerate(expected):
            compare(value, actual[i], f"{path}/{i}", tolerance)
    elif isinstance(expected, float):
        if not isinstance(actual, (float, int)) or isinstance(actual, bool):
            raise ValueError(f"Different numeric type: {path}")
        if not math.isfinite(actual) or abs(expected - actual) > tolerance:
            raise ValueError(f"Numeric mismatch: {path}: {expected} vs {actual}")
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"Different value: {path}")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replay6(workspace):
    sys.path.insert(0, str(REPO / "reconstruction/modules/v2d_issue6"))
    from issue6.pipeline import contacts_stage

    manifest = read(workspace / "manifest.json")
    for obj in manifest["objects"]:
        obj["mesh"] = str(workspace / obj["mesh"])
    for frame in manifest["frames"]:
        for obj in frame["objects"].values():
            if "pose_path" in obj:
                obj["pose_path"] = str(workspace / obj["pose_path"])
    output = workspace / "recomputed-contacts"
    output.mkdir()
    contacts_stage(manifest, workspace, output)
    for filename in ("contacts.json", "dominant_objects.json", "settings.json"):
        compare(
            read(workspace / "expected-contacts" / filename), read(output / filename)
        )
    rows = read(output / "contacts.json")
    historical = read(workspace / "historical-contacts/contacts.json")
    changed_normals = 0
    for old, new in zip(historical, rows, strict=True):
        old, new = dict(old), dict(new)
        old_normal, new_normal = old.pop("normal_object"), new.pop("normal_object")
        compare(old, new)
        try:
            compare(old_normal, new_normal)
        except ValueError:
            changed_normals += 1
    return {
        "frames": len(manifest["frames"]),
        "contact_rows": len(rows),
        "raw_positive_rows": sum(r["raw_contact"] is True for r in rows),
        "unknown_rows": sum(r["raw_contact"] is None for r in rows),
        "all20_quality_accepted": 0,
        "historical_contact_decisions_exact": True,
        "canonical_normals_changed_from_historical": changed_normals,
        "scope": "frozen v3 baseline contact computation",
    }


def replay7(workspace):
    scripts = REPO / "scripts/track3"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "refinement_report", scripts / "refinement_report.py"
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    inputs = read(workspace / "inputs.json")
    result = report.evaluate(
        report.load_scene(workspace / "scene.npz"),
        read(workspace / "contacts.json"),
        0.05,
        10,
        0.005,
        20000,
        read(workspace / "rest-annotations.json"),
        inputs.get("object_provenance", {}),
        read(workspace / "contact-truth.json"),
        inputs.get("camera_provenance"),
        2,
    )
    result["contact"]["reference_sha256"] = sha(workspace / "contact-truth.json")
    result["episode"] = 2
    expected = read(workspace / "expected-report.json")
    expected.pop(
        "official", None
    )  # Requires the separate public reference dataset/scorer.
    compare(expected, result)
    (workspace / "recomputed-report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return {
        "frames": result["n_frames"],
        "camera_initial_jitter_mm": result["camera"]["initial"]["translation"][
            "highfreq_rms_mm"
        ],
        "camera_refined_jitter_mm": result["camera"]["refined"]["translation"][
            "highfreq_rms_mm"
        ],
        "scope": "CPU metrics from frozen GPU scene; official reference scoring excluded",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", type=int, choices=(6, 7), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    catalog = read(REPO / "docs/team/reproduction/artifacts.json")
    key = f"issue{args.issue}_cpu"
    entry = catalog[key]
    workspace = args.out.resolve()
    files = [Path(__file__), REPO / "scripts/reproduce/requirements-cpu.txt"]
    if args.issue == 6:
        files += list((REPO / "reconstruction/modules/v2d_issue6/issue6").glob("*.py"))
    else:
        files += [
            REPO / "scripts/track3/refinement_report.py",
            REPO / "scripts/track3/contact_validation.py",
        ]
    source_hashes = {p.relative_to(REPO).as_posix(): sha(p) for p in files}
    restore(args.archive, entry["sha256"], workspace)
    result = replay6(workspace) if args.issue == 6 else replay7(workspace)
    receipt = {
        "status": "reproduced",
        "issue": args.issue,
        "archive_sha256": entry["sha256"],
        "python": sys.version,
        "source_sha256": source_hashes,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "Pillow", "trimesh", "rtree")
        },
        "absolute_float_tolerance": 1e-8,
        **result,
    }
    (workspace / "reproduction.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
