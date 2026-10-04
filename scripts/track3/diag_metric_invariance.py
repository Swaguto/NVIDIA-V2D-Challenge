#!/usr/bin/env python3
"""Measure how much the official Track 3 score actually responds to pose error.

This is the diagnostic behind the PR's central claim. Take any submission, derive several
deliberately *wrong* variants of it, and score all of them with
``scripts/eval/score_track3_official.py``:

  asis          the submission unchanged (control)
  identity_rot  every rotation replaced by wxyz [1,0,0,0], translations untouched
  inverse       full SE(3) inverse of every pose
  Rt            R transposed, translation left alone (wrong rotation, right position)
  tneg          translation negated, rotation left alone (mirrored through the camera)

Observed on our candidate (ADD-AUC / RPE cm / MPPE cm), episodes 0,11,18,23,39:

  asis          0.1946 / 12.67 / 15.48
  identity_rot  0.1946 / 16.95 / 16.73
  inverse       0.1946 / 56.51 / 35.90
  Rt            0.1946 / 17.92 / 17.62
  tneg          0.1946 / 18.07 / 16.63

ADD-AUC spread is **0.0000** across all five, including a submission whose poses are
actively inverted. At our accuracy level (~15 cm, against ADD thresholds of 1-9 cm) ADD-AUC
is therefore driven by the handful of frames that happen to fall under a threshold rather
than by pose quality, and it cannot rank submissions.

RPE and MPPE do respond, and in the expected direction, which is what makes them the useful
signals here. Two readings worth keeping:

  * ``asis`` beats ``identity_rot`` on RPE (12.67 vs 16.95 cm), so our per-frame rotation
    estimates carry real signal -- they are not noise. The ADD thresholds are simply too far
    below our error for that signal to reach the AUC.
  * ``asis`` vs ``inverse`` (12.67 vs 56.51 cm) confirms the pose convention is right:
    inverting it makes RPE 4.5x worse rather than better.

This is a statement about the metric *at our accuracy level*, not a claim that ADD-AUC is
broken in general. Centimetre-accurate poses would be ranked normally, and a leaderboard
won at that accuracy would be meaningful.

    python3 scripts/track3/diag_metric_invariance.py --predictions /home/ubuntu/pred_static5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(os.environ.get("V2D_PUBLIC", "/home/ubuntu/v2d_track3/data/hf/track_3/public"))
SCORER = Path(__file__).resolve().parents[1] / "eval" / "score_track3_official.py"


def discover(root: Path) -> list[tuple[str, str, Path]]:
    out = []
    for ep in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("e")):
        base = ep / "ego_object_poses" / ep.name
        if not base.is_dir():
            continue
        for body in sorted(base.iterdir()):
            for cam in sorted((body / "object_to_cam").iterdir()) if (body / "object_to_cam").is_dir() else []:
                if cam.is_dir():
                    out.append((ep.name, body.name, cam))
    return out


def R_wxyz(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_wxyz(R):
    m = np.asarray(R, float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = .25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, .25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, .25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, .25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def variant(rec: dict, kind: str, other: dict | None = None) -> dict:
    """Apply one deliberately-wrong transform, preserving every other field."""
    out = dict(rec)
    if kind == "identity_rot":
        out["rotation"] = [1.0, 0.0, 0.0, 0.0]
        return out
    if other is not None:                       # hybrid: rotations borrowed from `other`
        out["rotation"] = other["rotation"]
        return out
    R = R_wxyz(rec["rotation"])
    t = np.asarray(rec["translation"], float)
    if kind == "Rt":
        out["rotation"] = quat_wxyz(R.T).tolist()
    elif kind == "tneg":
        out["translation"] = (-t).tolist()
    elif kind == "inverse":
        out["rotation"] = quat_wxyz(R.T).tolist()
        out["translation"] = (-R.T @ t).tolist()
    return out


def build(src: Path, dst: Path, kind: str, other_src: Path | None = None) -> None:
    other_dirs = {}
    if other_src:
        other_dirs = {(e, o): d for e, o, d in discover(other_src)}
    for ep, obj, d in discover(src):
        od = dst / f"e{ep[1:]}" / "ego_object_poses" / f"e{ep[1:]}" / obj / "object_to_cam/ego_cam_c"
        od.mkdir(parents=True, exist_ok=True)
        od_other = other_dirs.get((ep, obj))
        for p in sorted(d.glob("*.json")):
            rec = json.loads(p.read_text())
            o = json.loads((od_other / p.name).read_text()) if od_other else None
            (od / p.name).write_text(json.dumps(variant(rec, kind, o)))


def score(pred: Path, episodes: str) -> tuple[float, float, float]:
    """-> (ADD-AUC, RPE cm, MPPE cm) from the official scorer's own aggregate line."""
    out = subprocess.run(
        [sys.executable, str(SCORER), "--predictions", str(pred),
         "--episodes", episodes, "--data", str(_ROOT)],
        capture_output=True, text=True, check=False)
    lines = out.stdout.splitlines()
    # the aggregate row is the last numeric-only line before the "(higher better" footer
    for i, line in enumerate(lines):
        if "(higher better" in line:
            for prev in reversed(lines[:i]):
                nums = re.findall(r"[\d.]+", prev)
                if len(nums) == 5:
                    return float(nums[0]), float(nums[3]), float(nums[4])
    # fallback: average the per-episode rows
    aucs, rpes, mppes = [], [], []
    for line in lines:
        m = re.search(r"AUC=([\d.]+).*?RPE=\s*([\d.]+)cm.*?MPPE=\s*([\d.]+)cm", line)
        if m:
            aucs.append(float(m.group(1)))
            rpes.append(float(m.group(2)))
            mppes.append(float(m.group(3)))
    if aucs:
        return float(np.mean(aucs)), float(np.mean(rpes)), float(np.mean(mppes))
    print(out.stdout[-2000:], out.stderr[-2000:], file=sys.stderr)
    raise RuntimeError(f"could not parse scorer output for {pred}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--episodes", default="0,11,18,23,39")
    ap.add_argument("--other", type=Path, help="submission to borrow rotations from (hybrid)")
    ap.add_argument("--keep", type=Path, help="write the variants here instead of a tmpdir")
    a = ap.parse_args()

    kinds = ["asis", "identity_rot", "inverse", "Rt", "tneg"]
    tmp = Path(tempfile.mkdtemp(prefix="invariance_"))
    dest = a.keep or tmp
    dest.mkdir(parents=True, exist_ok=True)

    print(f"{'variant':<14} {'ADD-AUC':>8} {'RPE cm':>8} {'MPPE cm':>8}")
    print("-" * 42)
    rows = []
    for k in kinds:
        d = dest / f"inv_{k}"
        build(a.predictions, d, k, a.other if k == "asis" else None)
        auc, rpe, mppe = score(d, a.episodes)
        rows.append((k, auc, rpe, mppe))
        print(f"{k:<14} {auc:>8.4f} {rpe:>8.2f} {mppe:>8.2f}", flush=True)

    spread = max(r[1] for r in rows) - min(r[1] for r in rows)
    print(f"\nADD-AUC spread across {len(rows)} variants "
          f"(one of them fully inverted): {spread:.4f}")
    print(f"RPE spread: {max(r[2] for r in rows) - min(r[2] for r in rows):.2f} cm")
    if not a.keep:
        print(f"\n(variants left in {tmp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
