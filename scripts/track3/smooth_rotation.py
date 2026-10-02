#!/usr/bin/env python3
"""Temporal regularisation of object rotations.

Context: after fixing the quaternion writer, per-frame ICP rotations scored WORSE than
simply reporting identity (AUC 0.036 vs 0.169). That is not a bug in the fix -- it says the
per-frame rotation estimates are noisy, while GT rotations are nearly constant within an
episode (measured spread: lid 6.2 deg median). Averaging the rotation trajectory on the
manifold therefore reduces error, exactly as averaging a noisy signal does.

This replaces each frame's rotation by the geodesic mean of its episode (optionally a local
window), leaving translations untouched.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def geodesic_mean(quats, weights=None):
    """Weighted mean rotation = top eigenvector of the symmetric 4x4 markov matrix."""
    q = Rot.from_quat(np.asarray(quats, float))          # scipy is xyzw
    M = np.zeros((4, 4))
    for i in range(len(q)):
        w = q[i].as_quat()
        if weights is not None:
            w = w * weights[i]
        wx, wy, wz, ww = w
        M += np.outer(w, w)                               # accumulate rank-1 terms
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    out = V[:, int(np.argmax(w))]
    if out[3] < 0:                                        # canonical sign: w >= 0
        out = -out
    return Rot.from_quat(out / np.linalg.norm(out)).as_quat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--window", type=int, default=0,
                    help="0 = one constant rotation per episode; >0 = local window")
    a = ap.parse_args()

    ep = f"e{a.episode:03d}"
    if a.out.exists():
        shutil.rmtree(a.out)
    shutil.copytree(a.predictions, a.out)
    base = a.out / ep / "ego_object_poses" / ep

    for obj_dir in sorted(base.iterdir()):
        d = obj_dir / "object_to_cam" / "ego_cam_c"
        files = sorted(d.glob("*.json"))
        if not files:
            continue
        recs = [json.loads(f.read_text()) for f in files]
        # challenge order is (w, x, y, z); scipy wants (x, y, z, w)
        wxyz = np.array([r["rotation"] for r in recs], float)
        quats = wxyz[:, [1, 2, 3, 0]]
        if a.window and a.window < len(quats):
            means = []
            half = a.window // 2
            for i in range(len(quats)):
                lo, hi = max(0, i - half), min(len(quats), i + half + 1)
                means.append(geodesic_mean(quats[lo:hi]))
        else:
            means = [geodesic_mean(quats)] * len(quats)

        spread = np.degrees((Rot.from_quat(quats) * Rot.from_quat(means).inv()).magnitude())
        for r, q in zip(recs, means):
            r["rotation"] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        for f, r in zip(files, recs):
            f.write_text(json.dumps(r))
        print(f"  {obj_dir.name:18} frames={len(files):4d} "
              f"raw spread median={np.median(spread):6.2f} p90={np.percentile(spread,90):6.2f} deg")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()