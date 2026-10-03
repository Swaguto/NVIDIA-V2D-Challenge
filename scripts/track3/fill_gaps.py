#!/usr/bin/env python3
"""Repair per-frame pose tracks: drop tracking failures, then fill the holes.

Two distinct failure modes show up in real runs and both cost official-metric coverage:

1. ICP emitted no frame at all (detector miss).
2. ICP emitted a *confidently wrong* frame -- e.g. the sweeper's depth ran to 1.5 m with
   10x the normal ICP residual. These are worse than gaps, because a garbage pose still
   counts as a prediction and is scored.

So: frames whose depth is outside a plausible band, or whose residual exceeds a
threshold, are deleted first and then refilled by interpolating between the nearest
surviving frames. Rotation interpolates on the manifold (slerp), translation linearly.
Gaps wider than --max-gap are left empty rather than fabricated across a real failure.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def _scalar(v):
    return float(v[0]) if isinstance(v, list) else float(v)


def load(d):
    out = {}
    for f in d.glob("*.json"):
        r = json.loads(f.read_text())
        out[int(f.stem)] = (
            np.asarray(r["translation"], float),
            np.asarray(r["rotation"], float),
            _scalar(r.get("residual", np.nan)),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True,
                    help="total frames expected, i.e. the GT step count")
    ap.add_argument("--max-gap", type=int, default=3)
    ap.add_argument("--reject-z-min", type=float, default=None,
                    help="drop frames with translation z below this (m)")
    ap.add_argument("--reject-z-max", type=float, default=None,
                    help="drop frames with translation z above this (m)")
    ap.add_argument("--reject-residual", type=float, default=None,
                    help="drop frames whose ICP residual exceeds this (m)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ep = f"e{a.episode:03d}"
    base = a.predictions / ep / "ego_object_poses" / ep
    added = dropped = 0
    for obj in sorted(base.iterdir()):
        d = obj / "object_to_cam" / "ego_cam_c"
        recs = load(d)

        # stage 1: reject implausible poses
        reject = []
        for i, (t, q, res) in sorted(recs.items()):
            reasons = []
            if a.reject_z_min is not None and t[2] < a.reject_z_min:
                reasons.append(f"z={t[2]:.3f}")
            if a.reject_z_max is not None and t[2] > a.reject_z_max:
                reasons.append(f"z={t[2]:.3f}")
            if a.reject_residual is not None and np.isfinite(res) and res > a.reject_residual:
                reasons.append(f"residual={res:.4f}")
            if reasons:
                reject.append((i, ",".join(reasons)))
        for i, _ in reject:
            if not a.dry_run:
                (d / f"{i:06d}.json").unlink(missing_ok=True)
            recs.pop(i, None)
        dropped += len(reject)

        # stage 2: fill what is missing
        have = set(recs)
        missing = [i for i in range(a.steps) if i not in have]
        filled = skipped = 0
        for i in missing:
            lo = max((h for h in have if h < i), default=None)
            hi = min((h for h in have if h > i), default=None)
            if lo is None or hi is None or hi - lo > a.max_gap:
                skipped += 1
                continue
            tl, ql, _ = recs[lo]
            th, qh, _ = recs[hi]
            qlx, qhxyzw = ql[[1, 2, 3, 0]], qh[[1, 2, 3, 0]]
            if np.dot(qlx, qhxyzw) < 0:          # take the short arc
                qhxyzw = -qhxyzw
            w = (i - lo) / (hi - lo)
            q = Slerp([lo, hi], Rot.from_quat([qlx, qhxyzw]))([i]).as_quat()[0]
            t = tl * (1 - w) + th * w
            rec = {"translation": [float(x) for x in t],
                   "rotation": [float(q[3]), float(q[0]), float(q[1]), float(q[2])]}
            if not a.dry_run:
                (d / f"{i:06d}.json").write_text(json.dumps(rec))
            # keep recs in sync so later frames can interpolate against this one
            have.add(i)
            recs[i] = (t, np.array(rec["rotation"]), np.nan)
            filled += 1
        added += filled
        ex = reject[:3]
        print(f"{obj.name:18} rejected {len(reject):3d} "
              f"(e.g. {', '.join(f'{i}:{why}' for i, why in ex)})"
              f"{'...' if len(reject) > 3 else ''}"
              f" | filled {filled:3d}, left {skipped:2d} gaps > {a.max_gap}")
    print(f"total rejected {dropped}, total interpolated {added}")


if __name__ == "__main__":
    main()