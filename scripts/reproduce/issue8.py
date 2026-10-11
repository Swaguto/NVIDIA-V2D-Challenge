"""Verify, restore and reproduce the frozen 343-frame derived-model pilot."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from checkpoint import REPO, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_hashes = {
        p.name: sha(p)
        for p in (Path(__file__), Path(__file__).with_name("restore_derived_pilot.py"))
    }
    catalog = json.loads(
        (REPO / "docs/team/reproduction/artifacts.json").read_text(encoding="utf-8")
    )
    for key, path in [
        ("issue8_baseline", args.baseline),
        ("issue8_derived", args.supplement),
    ]:
        if sha(path) != catalog[key]["sha256"]:
            raise ValueError(f"{key}: archive differs from published SHA256")
    # Frozen helpers contain assertion checks: never disable them.
    env = dict(os.environ, PYTHONOPTIMIZE="0")
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("restore_derived_pilot.py")),
            "--baseline",
            str(args.baseline.resolve()),
            "--supplement",
            str(args.supplement.resolve()),
            "--out",
            str(args.out.resolve()),
        ],
        check=True,
        env=env,
    )
    helper = args.out.resolve() / "runs/issue8-validation/reproduce_derived_pilot.py"
    subprocess.run(
        [sys.executable, "-u", str(helper), "--workspace", str(args.out.resolve())],
        check=True,
        env=env,
    )
    measured = json.loads(
        (helper.parent / "derived-portable-reproduction.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        measured.get("fully_checked_derived_model_accepted_frames"),
        measured.get("rejected_frames"),
        measured.get("unsupported_frames"),
        measured.get("accepted_mask_exact"),
    ) != (72, 199, 72, True):
        raise ValueError("Unexpected reproduced disposition")
    receipt = {
        "status": "reproduced",
        "issue": 8,
        "frames": 343,
        "accepted_frames": 72,
        "rejected_frames": 199,
        "unsupported_frames": 72,
        "accepted_mask_exact": True,
        "python": sys.version,
        "source_sha256": source_hashes,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "Pillow", "trimesh", "rtree")
        },
        "baseline_sha256": catalog["issue8_baseline"]["sha256"],
        "supplement_sha256": catalog["issue8_derived"]["sha256"],
        "floating_gate_tolerance_m": 1e-12,
        "scope": "derived-model timestamp bounds; no physics or between-frame certification",
    }
    (args.out.resolve() / "reproduction.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
