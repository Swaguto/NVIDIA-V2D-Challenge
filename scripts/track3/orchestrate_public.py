#!/usr/bin/env python3
"""Process public episodes as their temporal masks become available."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from batch_public import DATA_DEFAULT, roster, stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fp-python", default=os.environ.get(
        "V2D_FP_PYTHON", "/home/ubuntu/fp_venv/bin/python"))
    parser.add_argument("--skip", default="", help="comma-separated episodes handled separately")
    args = parser.parse_args()
    skipped = {int(part) for part in args.skip.split(",") if part.strip()}
    results = {}
    for episode, names in roster(args.data):
        if episode in skipped:
            continue
        mask = args.root / f"e{episode:03d}" / "masks/report.json"
        while not mask.is_file():
            manifest = args.root / "batch_masks.json"
            state = json.loads(manifest.read_text()) if manifest.is_file() else {}
            if state.get(str(episode), "").startswith("failed"):
                results[str(episode)] = {"masks": state[str(episode)]}
                break
            print(f"waiting for episode {episode} masks", flush=True)
            time.sleep(20)
        if mask.is_file():
            results[str(episode)] = {}
            for phase in ("poses", "scores", "overlays"):
                try:
                    result = stage(phase, episode, names, args.data, args.root,
                                   sys.executable, args.fp_python)
                    results[str(episode)][phase] = result
                except Exception as error:
                    traceback.print_exc()
                    results[str(episode)][phase] = f"failed: {error}"
                    if phase == "poses":
                        break
                print(f"episode {episode} {phase}: {results[str(episode)][phase]}", flush=True)
                (args.root / "batch_pipeline.json").write_text(json.dumps(results, indent=2))
        (args.root / "batch_pipeline.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
