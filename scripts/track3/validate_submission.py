#!/usr/bin/env python3
"""Validate a Track 3 submission directory before shipping it.

The scorer rejects a candidate whose rows are not all present, so a missing frame is a
hard failure rather than a degraded score. This walks the submission and checks, per
episode/object, that every expected frame exists with a finite translation and a unit
quaternion, and reports the camera distance so an absurd pose (e.g. metres instead of
centimetres, or an inverted frame) is obvious at a glance.

Episode/object names are discovered from the submission itself and the expected frame
count is read from the public reference, so this also works for a private-set layout whose
object names we do not know in advance.

    python3 scripts/track3/validate_submission.py --predictions /path/to/pred
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_ROOT = Path(os.environ.get("V2D_PUBLIC", "/home/ubuntu/v2d_track3/data/hf/track_3/public"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))


def discover(root: Path) -> dict[str, dict[str, Path]]:
    """{episode: {object: object_to_cam/ego_cam_c dir}} for every episode in the tree."""
    out: dict[str, dict[str, Path]] = {}
    for ep_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("e")):
        objs: dict[str, Path] = {}
        for body in sorted((ep_dir / "ego_object_poses" / ep_dir.name).iterdir()):
            cams = body / "object_to_cam"
            if not cams.is_dir():
                continue
            for cam in sorted(cams.iterdir()):
                if cam.is_dir():
                    objs[body.name] = cam
        if objs:
            out[ep_dir.name] = objs
    return out


def expected_steps(ep: str) -> int | None:
    """Frame count from the public reference; None when the episode is not public."""
    try:
        from reference_loader import load_reference
        return int(load_reference(int(ep[1:]), _ROOT).steps)
    except Exception:
        return None


def check(root: Path) -> int:
    layout = discover(root)
    if not layout:
        print(f"no episode directories under {root}")
        return 1
    failures = 0
    for ep, objs in sorted(layout.items()):
        steps = expected_steps(ep)
        for obj, d in sorted(objs.items()):
            have = {p.name for p in d.glob("*.json")}
            problems: list[str] = []
            if steps is not None:
                want = {f"{i:06d}.json" for i in range(steps)}
                missing, extra = want - have, have - want
                if missing:
                    problems.append(f"{len(missing)} missing (e.g. {sorted(missing)[:2]})")
                if extra:
                    problems.append(f"{len(extra)} unexpected (e.g. {sorted(extra)[:2]})")
            worst_q, worst_t, nonfinite = 0.0, 0.0, 0
            for name in sorted(have):
                rec = json.loads((d / name).read_text())
                q = rec.get("rotation")
                t = rec.get("translation")
                if q is None or t is None:
                    problems.append(f"{name}: missing rotation/translation")
                    continue
                if not all(math.isfinite(v) for v in q) or not all(math.isfinite(v) for v in t):
                    nonfinite += 1
                    continue
                worst_q = max(worst_q, abs(1.0 - math.sqrt(sum(v * v for v in q))))
                worst_t = max(worst_t, math.sqrt(sum(v * v for v in t)))
            if nonfinite:
                problems.append(f"{nonfinite} non-finite")
            if worst_q > 1e-3:
                problems.append(f"quaternion norm off by {worst_q:.2e}")
            ok = not problems
            failures += 0 if ok else 1
            exp = f"{steps}" if steps is not None else "?"
            print(f"{'OK  ' if ok else 'FAIL'} {ep} {obj:22} frames={len(have):>4}/{exp:<4} "
                  f"|1-|q||={worst_q:.1e} max|t|={worst_t:.3f}m"
                  + (f"  {'; '.join(problems)}" if problems else ""))
    print(f"\n{len(layout)} episodes checked, "
          f"{sum(len(v) for v in layout.values()) - failures}/"
          f"{sum(len(v) for v in layout.values())} object directories valid")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, type=Path)
    a = ap.parse_args()
    return check(a.predictions)


if __name__ == "__main__":
    raise SystemExit(main())
