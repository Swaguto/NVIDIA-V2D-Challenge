#!/usr/bin/env python3
"""Repair diverged pose trajectories without ground truth.

Re-anchored tracking diverged on 3 of 9 objects (translation range up to 150 cm, single
jumps of 146-168 cm): a registration occasionally locks onto a wrong pose, and because the
next frames are tracked from it the error persists. Those frames are detectable from two
self-consistency cues alone, no GT required:

  jump  : ``|p(t) - p(t-1)|`` above a physical threshold. Measured GT motion is
          <= 0.6 cm/frame, so 6 cm/frame is a factor of ten of headroom.
  range : ``|p(t) - median(p)|`` beyond a plausible bound for a tabletop object.

Flagged frames are dropped and replaced by linear interpolation across the gap, then the
series is boxcar-smoothed and the residual motion shrunk toward the temporal median:

    p(t) -> median + lam * (smooth(p_interp) - median)

Rotations are smoothed the same way and re-orthonormalised by SVD (never averaged as raw
matrices). ``lam=0`` collapses to the per-episode median pose, which is the strongest
single baseline we measured; ``lam=1`` keeps all corrected motion.

The submission layout is discovered from ``--src``, so this also works on a private split
whose object names we do not know in advance.

    python3 scripts/track3/repair_trajectory.py --src pred_fpreanchor --out pred_re_fix
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def discover(root: Path) -> list[tuple[str, str, Path]]:
    """[(episode, object, frame_dir)] for every frame directory in a submission."""
    out = []
    for ep_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("e")):
        base = ep_dir / "ego_object_poses" / ep_dir.name
        if not base.is_dir():
            continue
        for body in sorted(base.iterdir()):
            cams = body / "object_to_cam"
            if cams.is_dir():
                for cam in sorted(cams.iterdir()):
                    if cam.is_dir():
                        out.append((ep_dir.name, body.name, cam))
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


def boxcar(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return a.copy()
    pad = win // 2
    ap = np.pad(a, [(pad, pad)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    k = np.ones(win) / win
    out = np.empty_like(a)
    f, o = ap.reshape(ap.shape[0], -1), out.reshape(out.shape[0], -1)
    for j in range(f.shape[1]):
        o[:, j] = np.convolve(f[:, j], k, mode="valid")
    return out


def interp_gaps(ts: np.ndarray, bad: set[int]):
    """Replace flagged rows by linear interpolation across nearest good neighbours."""
    good = [i for i in range(len(ts)) if i not in bad]
    if not good:
        return ts, 0
    out = ts.copy()
    for i in bad:
        lo = max([g for g in good if g < i], default=None)
        hi = min([g for g in good if g > i], default=None)
        if lo is None:
            out[i] = ts[hi]
        elif hi is None:
            out[i] = ts[lo]
        else:
            a = (i - lo) / (hi - lo)
            out[i] = (1 - a) * ts[lo] + a * ts[hi]
    return out, len(bad)


def repair_one(d: Path, od: Path, jump_cm: float, range_cm: float, win: int, lam: float):
    ps = sorted(d.glob("*.json"))
    T = len(ps)
    if T == 0:
        return None
    ts = np.zeros((T, 3))
    Rs = np.zeros((T, 3, 3))
    for i, p in enumerate(ps):
        r = json.loads(p.read_text())
        ts[i] = r["translation"]
        Rs[i] = R_wxyz(r["rotation"])
    med = np.median(ts, axis=0)
    dev = np.linalg.norm(ts - med, axis=1) * 100
    step = np.zeros(T)
    step[1:] = np.linalg.norm(np.diff(ts, axis=0), axis=1) * 100
    bad = set(np.where(dev > range_cm)[0].tolist())
    bad |= set((np.where(step > jump_cm)[0] + 1).tolist())
    t_fixed, nbad = interp_gaps(ts, bad)
    t_sm = boxcar(t_fixed, win)
    med2 = np.median(t_sm, axis=0)
    t_out = med2 + lam * (t_sm - med2)

    R_sm = np.stack([boxcar(Rs[:, i, j][:, None], win)[:, 0]
                     for i in range(3) for j in range(3)], axis=1).reshape(T, 3, 3)
    od.mkdir(parents=True, exist_ok=True)
    for i in range(T):
        U, _, Vt = np.linalg.svd(R_sm[i])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        (od / f"{i:06d}.json").write_text(json.dumps(
            {"rotation": [float(v) for v in quat_wxyz(R)],
             "translation": [float(v) for v in t_out[i]]}))
    return {"frames": T, "bad": nbad,
            "range_cm": (t_out.max(axis=0) - t_out.min(axis=0)) * 100,
            "maxdev_cm": float(dev.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jump-cm", type=float, default=6.0)
    ap.add_argument("--range-cm", type=float, default=35.0)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--lam", type=float, default=1.0,
                    help="0 = collapse to the per-episode median pose, 1 = keep all motion")
    a = ap.parse_args()

    found = discover(a.src)
    if not found:
        print(f"no frame directories under {a.src}")
        return 1
    if a.out.exists():
        shutil.rmtree(a.out)
    for ep, obj, d in found:
        od = a.out / f"e{ep[1:]}" / "ego_object_poses" / f"e{ep[1:]}" / obj / "object_to_cam/ego_cam_c"
        r = repair_one(d, od, a.jump_cm, a.range_cm, a.window, a.lam)
        if r:
            print(f"{ep} {obj:22} bad={r['bad']:3d}/{r['frames']:<4} "
                  f"range_cm={np.round(r['range_cm'],1)} maxdev_cm={r['maxdev_cm']:.1f}", flush=True)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
