"""Make a portable, label-free scored input view from verified acquisitions."""

import argparse
import json
import os
from pathlib import Path

from acquire_scored_inputs import hashes


def prepare(root, receipt_path, out):
    root, out = root.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(out)
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != "v2d.scored-input-acquisition.v1"
        or receipt.get("pose_labels_downloaded") is not False
    ):
        raise ValueError(
            "Expected verified video/asset acquisition without pose labels"
        )
    for item in receipt["files"]:
        source = root / item["path"]
        if (
            not source.resolve().is_relative_to(root)
            or source.stat().st_size != item["size"]
            or hashes(source)[0] != item["sha256"]
        ):
            raise ValueError("Acquired input bytes differ: " + item["path"])
    out.mkdir(parents=True)
    links = {"videos": root / "track_3/evaluation/videos"}
    links.update(
        {name: root / "track_3/public" / name for name in ("mesh", "urdf", "meta")}
    )
    for name, source in links.items():
        (out / name).symlink_to(os.path.relpath(source, out), target_is_directory=True)
    manifest = {
        "schema": "v2d.scored-input-view.v1",
        "dataset_revision": receipt["revision"],
        "acquisition_sha256": hashes(receipt_path)[0],
        "relative_links": {
            name: os.path.relpath(source, out) for name, source in links.items()
        },
        "no_pose_tables": True,
        "portability": "Preserve the relative dataset/view layout; mount their common parent read-only",
        "provided_meshes_are_not_claimed_as_reconstructions": True,
    }
    (out / "input-view.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "receipt", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.receipt, args.out), indent=2))
