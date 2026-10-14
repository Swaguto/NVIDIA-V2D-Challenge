"""Restore byte-verified original-CAD audit methods and measured evidence."""

import argparse
import json
from pathlib import Path

from reproduce_frozen_dex3_replay import restore_methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "manifest", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    restore_methods(args.archive, args.manifest, args.out)
    manifest = json.loads(args.manifest.read_text())
    print(
        json.dumps(
            {
                "verified_files": len(manifest["files_sha256"]),
                "out": str(args.out.resolve()),
                "audit_geometry_or_simulator_rerun": False,
            }
        )
    )


if __name__ == "__main__":
    main()
