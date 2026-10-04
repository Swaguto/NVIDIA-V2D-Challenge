#!/usr/bin/env python3
"""Optional rotation filtering of fixed-world trajectories, leaving translations intact.

Convert camera observations with convert_camera_poses.py before using this tool.
Camera-frame averaging erases real ego rotation even when observations are perfect.
Episode-wide averaging (--window 0) also erases real object rotation; compare raw
and short-window controls before choosing it. Outputs are inferred, with unvalidated
residuals; filtering is neither a new observation nor evidence of better tracking.
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
        weight = 1.0 if weights is None else float(weights[i])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weights must be finite and nonnegative")
        M += weight * np.outer(w, w)
    if np.trace(M) <= 0:
        raise ValueError("at least one positive weight is required")
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
    source, destination = a.predictions.resolve(), a.out.resolve()
    if (source == destination or source in destination.parents or destination in source.parents
            or destination.exists()):
        ap.error("--out must be a new directory outside the input tree")
    if a.window < 0:
        ap.error("--window must be nonnegative")
    bases = [source / ep / "ego_object_poses" / ep,
             source / "ego_object_poses" / ep, source]
    base = next((p for p in bases if p.is_dir() and
                 any((d / "object_to_world").is_dir() for d in p.iterdir())), None)
    if base is None:
        ap.error("no fixed-world object_to_world records; run convert_camera_poses.py first")

    pending, summaries = [], []
    for obj_dir in sorted(base.iterdir()):
        d = obj_dir / "object_to_world"
        files = sorted(d.glob("[0-9]*.json"), key=lambda f: int(f.stem))
        if not files:
            continue
        recs = [json.loads(f.read_text()) for f in files]
        frames = np.array([int(f.stem) for f in files])
        if len(set(frames)) != len(frames):
            raise ValueError(f"duplicate source frame IDs: {d}")
        for f, r in zip(files, recs):
            if r.get("coordinate_frame") != "fixed_world":
                raise ValueError(f"smoothing requires declared fixed-world coordinates: {f}")
            if r.get("frame_index", int(f.stem)) != int(f.stem) or r.get("episode", a.episode) != a.episode:
                raise ValueError(f"pose frame/episode mismatch: {f}")
            if r.get("status", "observed") not in ("observed", "inferred"):
                raise ValueError(f"cannot smooth rejected/fill records: {f}")
        # challenge order is (w, x, y, z); scipy wants (x, y, z, w)
        wxyz = np.array([r["rotation"] for r in recs], float)
        if wxyz.shape != (len(recs), 4) or not np.isfinite(wxyz).all():
            raise ValueError(f"expected finite wxyz rotations: {d}")
        quats = wxyz[:, [1, 2, 3, 0]]
        if a.window:
            means = []
            before, after = a.window // 2, (a.window - 1) // 2
            for frame in frames:
                keep = (frames >= frame - before) & (frames <= frame + after)
                means.append(geodesic_mean(quats[keep]))
        else:
            means = [geodesic_mean(quats)] * len(quats)

        spread = np.degrees((Rot.from_quat(quats) * Rot.from_quat(means).inv()).magnitude())
        for r, q in zip(recs, means):
            r["rotation"] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
            r["status"] = "inferred"
            r["smoothing"] = {"window": a.window, "coordinate_frame": "fixed_world",
                              "residual_validated": False}
            r.pop("residual", None)
        for f, r in zip(files, recs):
            pending.append((destination / f.relative_to(source), json.dumps(r, allow_nan=False)))
        summaries.append(f"  {obj_dir.name:18} frames={len(files):4d} "
              f"raw spread median={np.median(spread):6.2f} p90={np.percentile(spread,90):6.2f} deg")
    if not pending:
        ap.error("no fixed-world poses to smooth")
    # Validate every track before creating output. Preserve all original camera/raw files.
    shutil.copytree(source, destination)
    for f, content in pending:
        f.write_text(content)
    for summary in summaries:
        print(summary)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
