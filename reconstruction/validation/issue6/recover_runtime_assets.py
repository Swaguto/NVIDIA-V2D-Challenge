"""Recover hash-pinned public weights and verify privately supplied model inputs."""

import argparse
import datetime
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def target_under(root, relative):
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes root: {relative}")
    return target


def verify(path, item):
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in item and path.stat().st_size != item["bytes"]:
        raise ValueError(f"Asset byte count differs: {item['path']}")
    value = digest(path)
    if value != item["sha256"]:
        raise ValueError(f"Asset SHA256 differs: {item['path']}")
    return {"path": item["path"], "bytes": path.stat().st_size, "sha256": value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(__file__).with_name("runtime-asset-inventory.json"),
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--verify-only", action="store_true", help="Make no network requests"
    )
    parser.add_argument(
        "--mano-dir",
        type=Path,
        required=True,
        help="Private licensed directory containing MANO_LEFT/RIGHT.pkl",
    )
    parser.add_argument(
        "--foundationpose-dir",
        type=Path,
        required=True,
        help="Root downloaded from the official weights folder",
    )
    parser.add_argument(
        "--hamer-dir",
        type=Path,
        help="Verify extracted _DATA checkpoint as well as its archive",
    )
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    root = args.root.resolve()
    records = []
    for item in inventory["public"]:
        target = target_under(root, item["path"])
        if not target.exists() and not args.verify_only:
            if not item["url"].startswith("https://"):
                raise ValueError("Public assets require HTTPS")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with (
                urllib.request.urlopen(item["url"], timeout=120) as source,
                partial.open("wb") as output,
            ):
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            verify(partial, item)
            partial.replace(target)
        records.append({"group": "public", **verify(target, item)})
        print(json.dumps({"verified": item["path"]}), flush=True)
    for group, directory in [
        ("licensed_mano", args.mano_dir),
        ("foundationpose", args.foundationpose_dir),
    ]:
        for item in inventory[group]:
            records.append(
                {
                    "group": group,
                    **verify(target_under(directory.resolve(), item["path"]), item),
                }
            )
    if args.hamer_dir:
        for item in inventory["extracted_hamer"]:
            records.append(
                {
                    "group": "extracted_hamer",
                    **verify(
                        target_under(args.hamer_dir.resolve(), item["path"]), item
                    ),
                }
            )
    receipt = {
        "status": "verified",
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "inventory_sha256": digest(args.inventory),
        "files": records,
        "extracted_hamer_checkpoint_verified": args.hamer_dir is not None,
        "scope": "Model input reachability only; not inference or episode acceptance",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": "verified", "files": len(records), "receipt": str(args.receipt)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
