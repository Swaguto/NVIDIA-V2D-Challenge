#!/usr/bin/env python3
"""Is the relative-rotation error a fixed per-object frame offset, or tracking noise?

If GT poses are expressed in a different canonical frame than the public GLB I register
against (e.g. a URDF link frame vs the mesh's arbitrary origin), then every object's pose
carries a constant rotation offset Delta_i. The relative error Delta_1 Delta_0^T would then
be *constant over time*, and its residual after removing the mean would be small.

If instead the error is genuine per-frame tracking noise, the mean offset explains little and
the residual stays large. The distinction decides where to spend effort, so measure it.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("V2D_REPO", "/home/ubuntu/repoeval"))
from reference_loader import load_reference   # noqa: E402

ROOT = Path("/home/ubuntu/v2d_track3/data/hf/track_3/public")
PRED = Path(sys.argv[1])
EPS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0").split(",")]


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


def wq2m(w):
    return q2m([w[1], w[2], w[3], w[0]])


def load_pred(root, ep, name, n_obj):
    d = root / f"e{ep:03d}" / "ego_object_poses" / f"e{ep:03d}" / name \
        / "object_to_cam" / "ego_cam_c"
    out = {}
    for f in d.glob("*.json"):
        j = json.loads(f.read_text())
        T = np.eye(4)
        T[:3, :3] = wq2m(j["rotation"])
        T[:3, 3] = j["translation"]
        out[int(f.stem)] = T
    return out


def T_from(p):
    M = np.eye(4)
    M[:3, :3] = q2m(p[3:7])
    M[:3, 3] = p[0:3]
    return M


def log_so3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2 * np.sin(th)))


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def circ_mean(ws):
    """Mean direction of rotation vectors (SO(3) geodesic average)."""
    A = sum(so3_exp(w / max(np.linalg.norm(w), 1e-12) * min(np.linalg.norm(w), np.pi))
            for w in ws)
    U, _, Vt = np.linalg.svd(A)
    return U @ Vt


all_rel = {}
for ep in EPS:
    ref = load_reference(ep, ROOT)
    names = list(ref.object_names)
    if len(names) < 2:
        continue
    preds = [load_pred(PRED, ep, n, len(names)) for n in names]
    common = sorted(set(preds[0]) & set(preds[1]) & set(range(ref.pose_xyzw.shape[0])))
    if len(common) < 10:
        print(f"ep {ep}: only {len(common)} common frames, skipped")
        continue
    per_pair = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ws, ts = [], []
            for t in common:
                Pg = [T_from(ref.pose_xyzw[t, k]) for k in (a, b)]
                Pp = [preds[k][t] for k in (a, b)]
                rel_g = np.linalg.inv(Pg[1]) @ Pg[0]
                rel_p = np.linalg.inv(Pp[1]) @ Pp[0]
                ws.append(log_so3(rel_g[:3, :3].T @ rel_p[:3, :3]))
                ts.append(rel_p[:3, 3] - rel_g[:3, 3])
            ws, ts = np.array(ws), np.array(ts)
            Rm = circ_mean(ws)
            resid = np.array([np.degrees(np.linalg.norm(
                log_so3(so3_exp(w) @ Rm.T))) for w in ws])
            per_pair[(a, b)] = (Rm, np.degrees(np.linalg.norm(log_so3(Rm))), resid,
                                np.linalg.norm(ts, axis=1))
            all_rel.setdefault((names[a], names[b]), []).append((Rm, resid))
    print(f"\nep {ep}: {names}  frames={len(common)}")
    for (a, b), (Rm, mag, resid, terr) in per_pair.items():
        print(f"  {names[a]}_T_{names[b]}: mean offset {mag:6.2f} deg | "
              f"residual after removing it: median {np.median(resid):5.2f} "
              f"p90 {np.percentile(resid, 90):5.2f} deg | "
              f"rel-translation err median {np.median(terr) * 100:.1f} cm")

print("\n=== pooled across episodes ===")
for (na, nb), rows in sorted(all_rel.items()):
    Rm = circ_mean([log_so3(r[0]) for r in rows])
    mag = np.degrees(np.linalg.norm(log_so3(Rm)))
    resid = np.concatenate([r[1] for r in rows])
    print(f"  {na}_T_{nb}: mean offset {mag:6.2f} deg | pooled residual "
          f"median {np.median(resid):5.2f} p90 {np.percentile(resid, 90):5.2f} deg")
    print(f"      axis = {np.round(log_so3(Rm) / max(np.linalg.norm(log_so3(Rm)), 1e-9), 3)}")