#!/usr/bin/env python3
"""Self-check for the Track 3 scoring harness.

Verifies the two invariants that make the proxy leaderboard trustworthy:
  1. Oracle (GT used as the candidate) scores AUC=SP-SR=MP-SR=1.0 and MPPE=0.
  2. A candidate with growing noise scores monotonically worse, and rigid
     alignment recovers a pure world-frame offset.

Usage:  python scripts/eval/selfcheck.py [--data <public_dir>]
Exit 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference_loader import PROXY_EPISODES, load_reference  # noqa: E402
from score_candidate import rigid_align, score_episode  # noqa: E402

TOL = {"add_auc": 1e-6, "sp_sr": 1e-6, "mp_sr": 1e-6, "mppe_cm": 1e-9}


def oracle_noise(base: str | os.PathLike) -> int:
    """1. Oracle -> perfect metrics. 2. Noise degrades monotonically."""
    # Perfect tracking seeded by partial GT (candidate = reference, world axis added).
    for episode_index in PROXY_EPISODES:
        ref = load_reference(episode_index, base)
        achieved = ref.pose_xyzw[:, None, :, :]
        scored = score_episode(episode_index, achieved, base)
        for metric, tol in TOL.items():
            want = {"add_auc": 1.0, "sp_sr": 1.0, "mp_sr": 1.0, "mppe_cm": 0.0}[metric]
            if abs(scored[metric] - want) > tol:
                print(f"FAIL oracle ep={episode_index} {metric}={scored[metric]:.6f} want={want}")
                return 1
        print(f"ok   oracle ep={episode_index} -> all metrics perfect")

    # Monotonic degradation under growing position noise.
    rng = np.random.default_rng(0)
    grouped: list[float] = []
    for std in (0.001, 0.01, 0.05, 0.15):
        level: list[float] = []
        for episode_index in PROXY_EPISODES:
            ref = load_reference(episode_index, base)
            noisy = ref.pose_xyzw[:, None].copy()
            noisy[..., :3] += rng.normal(0, std, noisy[..., :3].shape)
            level.append(score_episode(episode_index, noisy, base)["add_auc"])
        grouped.append(float(np.mean(level)))
    if not (grouped[0] >= grouped[1] >= grouped[2] >= grouped[3] and grouped[0] > grouped[3]):
        print(f"FAIL monotonic AUC across noise: {grouped}")
        return 1
    print(f"ok   AUC monotone with noise: {[round(g, 3) for g in grouped]}")

    # Rigid alignment recovers a world-frame offset.
    ref = load_reference(PROXY_EPISODES[0], base)
    offset = np.array([0.5, -0.3, 0.2])
    shifted = ref.pose_xyzw[:, None].copy()
    shifted[..., :3] += offset
    before = score_episode(PROXY_EPISODES[0], shifted, base)["add_auc"]
    aligned = rigid_align(shifted, ref.pose_xyzw, ref.visible)
    after = score_episode(PROXY_EPISODES[0], aligned, base)["add_auc"]
    if before > after - 0.05 and not np.isclose(after, 1.0, atol=0.05):
        print(f"FAIL rigid align recovery before={before:.3f} after={after:.3f}")
        return 1
    print(f"ok   rigid alignment recovers frame offset: AUC {before:.3f} -> {after:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.expanduser("~/v2d_track3/data/hf/track_3/public"),
    )
    args = parser.parse_args()
    return oracle_noise(args.data)


if __name__ == "__main__":
    raise SystemExit(main())