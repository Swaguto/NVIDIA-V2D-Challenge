#!/usr/bin/env python3
"""Detector-free Track 3 registration: propose objects from the depth map.

Grounding DINO mislocalises some objects badly enough that SAM2 masks the *table*
instead (measured: 0.255 m depth std inside the "pan" mask, IoU 0.03 between the
projected CAD and its own mask). ICP then fits the mesh to a table plane, which looks
plausible in a video and is completely wrong.

So do not depend on the detector for *where* the object is. Objects rest on a table;
remove the dominant plane, threshold the residual height, and take connected
components. Each blob is a candidate object. Blobs are then assigned to object names
by comparing their 3D extents to the known mesh extents, which needs no appearance
model and no labels.

The detector is still useful for naming; this path only replaces localisation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.environ.get("V2D_REPO", "/home/ubuntu/repoeval"))
sys.path.insert(0, os.environ.get("V2D_RUNTIME", "/home/ubuntu"))
from reference_loader import load_reference                      # noqa: E402
from stereo_depth import backproject_full, load_rig, stereo_depth  # noqa: E402
from register_v4 import (DEV, _init_rotations, _icp, _mat_to_wxyz,  # noqa: E402
                          load_cad, read_frame, trim_cloud, voxel_downsample)

_ROOT = Path("/home/ubuntu/v2d_track3/data/hf/track_3/public")
_VID = _ROOT / "videos" / "chunk-000"


def table_plane(depth, valid, pts):
    """Robust dominant horizontal plane, fitted on the lower part of the frame."""
    H = depth.shape[0]
    low = valid.copy()
    low[: int(0.45 * H)] = False
    P = pts[low.ravel()].astype(np.float64)
    if len(P) < 2000:
        low = valid
        P = pts[low.ravel()].astype(np.float64)
    P = P[np.random.default_rng(0).choice(len(P), min(12000, len(P)),
                                          replace=False)]
    n = None
    for _ in range(3):
        c = P.mean(0)
        _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
        n = Vt[2]
        if n[2] < 0:
            n = -n
        d = np.abs((P - c) @ n)
        keep = d < max(0.01, 2.5 * np.median(d))
        if keep.sum() < 1000:
            break
        P = P[keep]
    return n, P.mean(0)


def mesh_signature(name):
    """Sorted 3D extents of the mesh: rotation-invariant size fingerprint."""
    d = _ROOT / "mesh" / name
    for cand in (f"{name}_visual", name):
        p = d / f"{cand}.glb"
        if p.exists():
            m = trimesh.load(str(p), force="mesh", process=False)
            v = np.asarray(m.vertices, np.float64)
            return np.sort(v.max(0) - v.min(0)), v
    raise FileNotFoundError(name)


def blob_clouds(depth, valid, pts, plane, min_px=350, max_std=0.09):
    """Connected components of the height-above-table residual."""
    n, p0 = plane
    X = pts.reshape(-1, 3)
    h = ((X - p0) @ n).reshape(depth.shape)
    obj = valid & (h > 0.022) & (h < 0.75)
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    num, lab, stats, cent = cv2.connectedComponentsWithStats(obj, 8)
    out = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px or area > 120000:
            continue
        sel = (lab == i) & valid
        if sel.sum() < min_px:
            continue
        z = depth[sel].astype(np.float64)
        if z.std() > max_std:            # still a surface, not a compact object
            continue
        cloud = trim_cloud(voxel_downsample(pts[sel.ravel()], 0.004))
        if len(cloud) < 120:
            continue
        ext = np.sort(cloud.max(0) - cloud.min(0))
        out.append({"area": area, "mask": sel, "cloud": cloud, "ext": ext,
                    "centroid": cloud.mean(0), "box": stats[i, :4].tolist()})
    out.sort(key=lambda b: -b["area"])
    return out


def assign(blobs, sigs, names):
    """Greedy best-match of blob size fingerprints to mesh size fingerprints."""
    if not blobs:
        return {}
    cost = np.zeros((len(blobs), len(names)))
    for i, b in enumerate(blobs):
        for j, nm in enumerate(names):
            cost[i, j] = float(np.abs(np.log((b["ext"] + 1e-3) /
                                              (sigs[nm][0] + 1e-3))).sum())
    pairs = []
    for i in range(len(blobs)):
        for j in range(len(names)):
            pairs.append((cost[i, j], i, j))
    pairs.sort()
    used_b, used_n, res = set(), set(), {}
    for c, i, j in pairs:
        if i in used_b or j in used_n:
            continue
        used_b.add(i)
        used_n.add(j)
        res[names[j]] = blobs[i]
        if len(res) == len(names):
            break
    return res


def contact_height(cad_c, c_cad, c_dst, sol, plane):
    """Height (m) of the lowest CAD point above the table plane."""
    s, R, tc, _res = sol
    X = (s * (cad_c @ R.T) + (s * (R @ (-c_cad)) + tc + c_dst)).cpu().numpy()
    n, p0 = plane
    return float(((X - p0) @ n).min())


def select(cands, cad_c, c_cad, c_dst, plane, tol=0.015, gap=0.06,
           w_pen=3.0, w_gap=0.6):
    """Prefer the pose that actually rests on the table.

    A thin plate's silhouette barely changes when it is flipped over, so the
    point-to-point ICP residual cannot distinguish face-up from face-down and happily
    returns the mirrored pose. Contact with the surface it is resting on can: the true
    pose touches the table, the mirrored one either floats or sinks into it.
    """
    def key(sol):
        res = sol[3]
        if plane is None:
            return res
        h = contact_height(cad_c, c_cad, c_dst, sol, plane)
        return res + w_pen * max(0.0, tol - h) + w_gap * max(0.0, h - gap)
    return min(cands, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-h", type=float, default=0.022)
    ap.add_argument("--max-std", type=float, default=0.09)
    ap.add_argument("--max-blobs", type=int, default=8)
    ap.add_argument("--log", type=int, default=25)
    a = ap.parse_args()

    ref = load_reference(a.episode, _ROOT)
    names = list(ref.object_names)
    steps = a.frames or ref.steps
    rig = load_rig()
    sigs, cads = {}, {}
    for n in names:
        sigs[n], _ = mesh_signature(n)
        cads[n] = torch.tensor(load_cad(n), dtype=torch.float32, device=DEV)
    print(f"episode {a.episode}: {names} steps={steps}")
    for n in names:
        print(f"  {n:18} mesh extents={np.round(sigs[n], 4)}")

    vlp = _VID / f"observation.images.{rig.left}" / f"episode_{a.episode:06d}.mp4"
    vrp = _VID / f"observation.images.{rig.right}" / f"episode_{a.episode:06d}.mp4"
    prev = {n: None for n in names}
    t_start = time.time()

    for t in range(0, steps, a.stride):
        try:
            il, ir = read_frame(vlp, t), read_frame(vrp, t)
        except RuntimeError:
            break
        il, _ = rig.rectify_pair(il, ir)
        depth, valid = stereo_depth(rig, il, ir)
        pts = backproject_full(depth, valid, rig.P1)
        plane = table_plane(depth, valid, pts)
        blobs = blob_clouds(depth, valid, pts, plane, max_std=a.max_std)
        blobs = blobs[: a.max_blobs]
        pick = assign(blobs, sigs, names)

        for n in names:
            b = pick.get(n)
            if b is None:
                continue
            d = a.out / f"e{a.episode:03d}" / "ego_object_poses" / \
                f"e{a.episode:03d}" / n / "object_to_cam" / "ego_cam_c"
            d.mkdir(parents=True, exist_ok=True)
            dst = torch.tensor(b["cloud"], dtype=torch.float32, device=DEV)
            cad = cads[n]
            c_cad, c_dst = cad.mean(0), dst.mean(0)
            cadc, dstc = cad - c_cad, dst - c_dst
            z = torch.zeros(3, dtype=cad.dtype, device=DEV)
            cands = []
            R0 = t0_ = None
            if prev[n] is not None:
                Rp, tp, _ = prev[n]
                if torch.norm(Rp @ c_cad + tp - c_dst) < 0.6:   # blob still nearby
                    R0 = Rp.clone()
                    t0_ = tp.clone() + (R0 @ c_cad) - c_dst
            if R0 is not None:
                cands.append(_icp(cadc, dstc, R0, t0_, trim=0.6, iters=40))
            sub = cadc[torch.randperm(cadc.shape[0],
                                     device=DEV)[:900]] if cadc.shape[0] > 900 else cadc
            scored = []
            for R in _init_rotations(sub, dstc):
                scored.append(_icp(sub, dstc, R, z, trim=0.6, iters=10))
            scored.sort(key=lambda x: x[3])
            for _, Rr, tt, _ in scored[:2]:
                cands.append(_icp(cadc, dstc, Rr, tt, trim=0.6, iters=50))
            best = select(cands, cadc, c_cad, c_dst, plane)
            s, R, tc, res = best
            t_cam = s * (R @ (-c_cad)) + tc + c_dst
            prev[n] = (R, t_cam, res)
            (d / f"{t:06d}.json").write_text(json.dumps({
                "translation": [float(x) for x in t_cam.cpu().numpy()],
                "rotation": [float(x) for x in _mat_to_wxyz(R.cpu().numpy())],
                "scale": float(s), "residual": float(res),
                "cloud_px": int(b["area"]), "blob_ext": [float(x) for x in b["ext"]],
            }))
        if a.log and (t // a.stride) % a.log == 0:
            got = {n: round(prev[n][2], 4) for n in names if prev[n] is not None}
            print(f"  t={t:>4}/{steps} blobs={len(blobs)} res={got} "
                  f"[{time.time() - t_start:.0f}s]", flush=True)
    print(f"done {a.episode} in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()