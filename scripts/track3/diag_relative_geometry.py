#!/usr/bin/env python3
"""Diagnose relative inter-object geometry, independent of the unknown camera extrinsic.

There is no published camera->world transform, so predicted poses cannot be compared to GT
in a common frame. But the *relative* transform between two objects is frame-invariant, and
GT supplies it directly:

    rel_gt = inv(world_T_lid) @ world_T_pot
    rel_pred = inv(cam_T_lid)  @ cam_T_pot

If rel_pred tracks rel_gt, the pipeline's internal geometry is right and any remaining error
is just the (scored) offset to the reference frame. If rel_pred is off, the failure is in
segmentation or registration, which is worth far more attention.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("V2D_REPO", "/home/ubuntu/repoeval"))
from reference_loader import load_reference   # noqa: E402

ROOT = Path("/home/ubuntu/v2d_track3/data/hf/track_3/public")
PRED = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/pred_e000")
EP = int(sys.argv[2] if len(sys.argv) > 2 else 0)


def q2m(q):
    x, y, z, w = q
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = np.array(q) / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_pred(name):
    d = PRED / f"e{EP:03d}" / "ego_object_poses" / f"e{EP:03d}" / name \
        / "object_to_cam" / "ego_cam_c"
    out = {}
    for f in sorted(d.glob("*.json")):
        j = json.loads(f.read_text())
        T = np.eye(4)
        T[:3, :3] = (q2m(j["quat_xyzw"]) if "quat_xyzw" in j else _wq(j["rotation"]))
        T[:3, 3] = j["translation"]
        out[int(f.stem)] = T
    return out


def _wq(wxyz):
    w, x, y, z = wxyz
    n = np.linalg.norm([w, x, y, z])
    if n < 1e-12:
        return np.eye(3)
    return q2m([x / n, y / n, z / n, w / n])


def angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


ref = load_reference(EP, ROOT)
names = list(ref.object_names)
print(f"episode {EP}: {names}")

if len(names) < 2:
    print("single-object episode: relative-geometry diagnostic does not apply")
    raise SystemExit(0)

preds = [load_pred(n) for n in names]
gt = ref.pose_xyzw


def T_from(p):
    """p is [x, y, z, qx, qy, qz, qw] (reference_pose is already xyzw)."""
    M = np.eye(4)
    M[:3, :3] = q2m(p[3:7])
    M[:3, 3] = p[0:3]
    return M


common = sorted(set(preds[0]) & set(preds[1]) & set(range(gt.shape[0])))
print(f"frames with both predictions: {len(common)}/{gt.shape[0]}")

dt, dr, dt_obj, dr_obj = [], [], [[], []], [[], []]
for t in common:
    Pg = [T_from(gt[t, b]) for b in range(2)]
    Pp = [preds[b][t] for b in range(2)]
    rel_g = np.linalg.inv(Pg[1]) @ Pg[0]
    rel_p = np.linalg.inv(Pp[1]) @ Pp[0]
    dt.append(np.linalg.norm(rel_g[:3, 3] - rel_p[:3, 3]))
    dr.append(angle(rel_g[:3, :3].T @ rel_p[:3, :3]))
    for b in range(2):
        # per-object rotation is only meaningful up to the unknown global frame
        dt_obj[b].append(np.linalg.norm(Pp[b][:3, 3] - Pg[b][:3, 3]))
        dr_obj[b].append(angle(Pp[b][:3, :3]))

print("\nrelative transform  lid_T_pot  (frame-invariant, directly comparable to GT):")
print(f"  translation error : median {np.median(dt) * 100:.2f} cm   "
      f"p90 {np.percentile(dt, 90) * 100:.2f} cm")
print(f"  rotation error    : median {np.median(dr):.2f} deg  "
      f"p90 {np.percentile(dr, 90):.2f} deg")
print("\nper-object translation offset vs GT (absolute; confounded by the unknown camera pose):")
for b, n in enumerate(names):
    print(f"  {n:<18} median {np.median(dt_obj[b]) * 100:6.2f} cm")
print("\nsanity: GT object separation (cm), and predicted separation:")
for t in common[:5]:
    g = np.linalg.norm(T_from(gt[t, 0])[:3, 3] - T_from(gt[t, 1])[:3, 3]) * 100
    p = np.linalg.norm(preds[0][t][:3, 3] - preds[1][t][:3, 3]) * 100
    print(f"  t={t:>4}  gt_sep={g:7.2f}  pred_sep={p:7.2f}  diff={p - g:+7.2f}")