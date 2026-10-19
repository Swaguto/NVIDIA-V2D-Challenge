"""Verify or restore the exact private producers without running experiments."""
import argparse
import hashlib
import json
from pathlib import Path


def restored_source(root, row):
    name = row["path"]
    if Path(name).name != name or not name.endswith(".py"):
        raise ValueError("Archive producer must be a local Python filename")
    data = (root / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != row["archive_sha256"]:
        raise ValueError("Archive source changed: " + name)
    lines = data.decode().splitlines(keepends=True)
    for item in row.get("blank_line_padding", []):
        index, padding = item["line_1based"] - 1, item["padding"]
        if lines[index] != "\n" or padding.strip():
            raise ValueError("Invalid blank-line restoration")
        lines[index] = padding + "\n"
    restored = "".join(lines).encode()
    if hashlib.sha256(restored).hexdigest() != row["original_source_sha256"]:
        raise ValueError("Original producer SHA mismatch: " + name)
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-workspace", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write and args.private_workspace is None:
        parser.error("--write requires --private-workspace")
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source-manifest.json").read_text())
    for row in manifest["sources"]:
        data = restored_source(root, row)
        if args.write:
            target = args.private_workspace.resolve() / "compute" / row["path"]
            if target.exists() and target.read_bytes() != data:
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print(f"Verified {len(manifest['sources'])} original source hashes")


if __name__ == "__main__":
    main()
