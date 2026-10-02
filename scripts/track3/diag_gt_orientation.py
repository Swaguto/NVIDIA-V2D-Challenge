#!/usr/bin/env python3
"""Are GT object orientations discrete/canonical, or free?

Shape registration cannot observe a symmetric object's in-plane angle, yet the official metric
scores rotation. So the question is whether GT orientations are drawn from a small discrete
set, which would make snapping an estimate to a canonical orientation legitimate.

Reports, per object:
  - within-episode orientation spread (is the object rotating at all?)
  - number of distinct orientations across all episodes, clustered at 15 deg
  - the deviation of every sample from its nearest cluster centre

GT only, so no GPU needed.
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("V2D_REPO", "/home/ubuntu/repoeval"))
from reference_loader import load_reference, TRAIN_EPISODES, PROXY_EPISODES   # noqa: E402

ROOT = Path("/home/ubuntu/v2d_track3/data/hf/track_3/public")


def q2m(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def ang(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def cluster(Rs, tol=15.0):
    cents = []
    for R in Rs:
        for i, c in enumerate(cents):
            if ang(c.T @ R) < tol:
                break
        else:
            cents.append(R)
            continue
        cents[i] = c
    return cents


per_obj = defaultdict(list)
eps = []
for ep in sorted(set(TRAIN_EPISODES) | set(PROXY_EPISODES)):
    try:
        ref = load_reference(ep, ROOT)
    except Exception:
        continue
    eps.append(ep)
    for b, name in enumerate(ref.object_names):
        Rs = np.array([q2m(ref.pose_xyzw[t, b, 3:7]) for t in range(ref.pose_xyzw.shape[0])])
        per_obj[name].append((ep, Rs))

print(f"scanned {len(eps)} public episodes\n")
hdr = (f"{'object':<21} {'n':>6} {'eps':>4} {'within-ep spread':>17} "
       f"{'clusters>15d':>13} {'max dev':>9}")
print(hdr)
print("-" * len(hdr))
for name, rows in sorted(per_obj.items()):
    allR = np.concatenate([r for _, r in rows])
    # within-episode spread: deviation of each sample from its episode's first orientation
    devs = []
    for _, Rs in rows:
        devs.extend(ang(Rs[0].T @ R) for R in Rs)
    devs = np.array(devs)
    cents = cluster(allR)
    to_cent = np.array([min(ang(c.T @ R) for c in cents) for R in allR])
    print(f"{name:<21} {len(allR):>6} {len(rows):>4} "
          f"med {np.median(devs):5.1f}d p95 {np.percentile(devs, 95):5.1f}d "
          f"{len(cents):>13} {to_cent.max():>8.1f}d")

print("\nInterpretation:")
print("  within-ep spread ~0  -> objects are static in orientation per episode; the")
print("  inter-episode variation is placement choice, and a small cluster count means")
print("  snapping to canonical orientations would be well-founded.")
print("  large within-ep spread -> the object genuinely rotates, so in-plane angle must")
print("  be tracked rather than snapped.")