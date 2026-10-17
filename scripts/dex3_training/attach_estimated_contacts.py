"""Copy a frozen task with separately bound estimated hints; preserve all inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from v2d_dex3_flashchord.estimated_reference import contacts, KIND


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach(base, estimate, out):
    base, estimate, out = base.resolve(), estimate.resolve(), out.resolve()
    if out.exists() or out.is_relative_to(base):
        raise ValueError("Need a fresh output outside the frozen base inputs")
    manifest = json.loads((base / "manifest.json").read_text())
    if manifest["schema"] != "v2d.dex3-training-task.v3" or manifest.get("estimated_contact_reference"):
        raise ValueError("Need an unmodified frozen task without estimated hints")
    if manifest.get("contacts_known") is not False:
        raise ValueError("Do not replace measured contact labels with estimates")
    for name, expected in manifest["files"].items():
        path = (base / name).resolve()
        if not path.is_relative_to(base) or digest(path) != expected:
            raise ValueError("Frozen base asset hash/path mismatch")
    receipt_path = estimate / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    source = estimate / "estimated-contact-candidates.npz"
    if digest(source) != receipt["files"][source.name]:
        raise ValueError("Candidate data differs from preparation receipt")
    shutil.copytree(base, out)
    folder = out / "estimated-contacts"
    folder.mkdir()
    for path in (source, receipt_path):
        destination = folder / path.name
        shutil.copy2(path, destination)
        manifest["files"][destination.relative_to(out).as_posix()] = digest(destination)
    manifest["estimated_contact_reference"] = dict(
        kind=KIND, data="estimated-contacts/estimated-contact-candidates.npz",
        receipt="estimated-contacts/receipt.json",
        base_input_manifest_sha256=digest(base / "manifest.json"),
        measured_contact_labels_unchanged=True,
    )
    with np.load(out / "targets.npz", allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    contacts(out, manifest, data)
    for name, expected in json.loads((base / "manifest.json").read_text())["files"].items():
        if digest(out / name) != expected:
            raise ValueError("Estimated hints changed an existing frozen asset")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(input_manifest_sha256=digest(out / "manifest.json"), base_input_manifest_sha256=digest(base / "manifest.json"), all_original_assets_preserved=True, measured_labels_unchanged=True, candidate_receipt=receipt, training_guidance_activated=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(attach(args.base, args.estimate, args.out), indent=2))
