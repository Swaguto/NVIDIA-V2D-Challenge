#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fit per-frame ego-camera world->cam poses from object depth + masks.

Replaces FoundationPose (whose per-camera 6D orientation was unreliable) with a
self-contained point-to-plane ICP fitter that reuses only the *verified* inputs:

  - undistorted frames, pinhole intrinsics   (work/intrinsics)
  - SAM2 masks                               (work/masks/{cam}/{obj}/0)
  - FoundationStereo depth                   (work/depth/{cam}/{frame}.png)
  - scanned object GLB meshes                (data_root/mesh)
  - parquet GT world->object per frame       (reference_loader)

Key modelling fact.  The two ego cameras form a rigid rig and move together:

    T_WC(cam_right) = M @ T_WC(cam_left),   M = stereo_pair.ego.left_to_right

with ego stereo pair ``left = ego_cam_c`` and ``right = ego_cam_b``.  So per
frame and per object there is exactly ONE unknown, an SE(3) ``T = T_WC(cam_left)``.
For a mesh point in world coords ``p``:

    cam_left:  m_l = T @ p
    cam_right: m_r = M @ (T @ p)

Both cameras' observed depth clouds inside the object mask must coincide with
the mesh (robust, Tukey-weighted point-to-plane), minimised jointly by
Levenberg-Gauss-Newton on SE(3).  Because both cameras constrain the *same* T
through a fixed rigid transform, rotation is far better identified than with
per-camera FP; temporal continuity and (optionally) the FoundationPose poses
are used only as initialisations.

Multi-object cross check.  Each object is fit independently; every frame where
two objects are simultaneously *passing* must yield the same T, which is a
strong consistency gate that FP never had.

Outputs under ``{work}/ego_cam_poses``:
  {obj}/world_to_cam/{cam}/{frame:06d}.json    Transform3d world->cam
  {obj}/object_to_cam/{cam}/{frame:06d}.json   Transform3d object->cam
  report.json                                  metrics + per-frame pass list

Run on the VM inside the pipeline venv (CPU only; needs scipy + trimesh):

  python -m v2d.world_calib.vm.fit_ego_rig \
      --data-root ~/v2d_track3/data/hf/track_3/public \
      --work ~/v2d_calib_work \
      --episode 2 [--objects white_pot,white_pot_lid] [--no-smooth]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
# <repo>/reconstruction/modules/v2d_world_calib/vm/ -> repo root is parents[4]
REPO_ROOT = HERE.parents[4]


# --------------------------------------------------------------------------- #
# SE(3) / quaternion helpers
# --------------------------------------------------------------------------- #


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64
    )


def _qmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rmat_to_quat_wxyz(R: np.ndarray) -> list[float]:
    q = np.zeros(4)
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s,
             (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                 (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                 (R[1, 2] + R[2, 1]) / s]
        else:
            s = np.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                 (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return [float(v) for v in q]


def _exp_se3(xi: np.ndarray) -> np.ndarray:
    w = np.asarray(xi[:3], dtype=np.float64)
    v = np.asarray(xi[3:], dtype=np.float64)
    th = float(np.linalg.norm(w))
    T = np.eye(4)
    if th < 1e-9:
        T[:3, :3] = np.eye(3) + _skew(w)
        T[:3, 3] = v
        return T
    K = _skew(w)
    T[:3, :3] = np.eye(3) + (np.sin(th) / th) * K + ((1 - np.cos(th)) / (th * th)) * (K @ K)
    V = np.eye(3) + ((1 - np.cos(th)) / (th * th)) * K + ((th - np.sin(th)) / (th ** 3)) * (K @ K)
    T[:3, 3] = V @ v
    return T


def _inv4(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    M = np.eye(4)
    M[:3, :3] = R.T
    M[:3, 3] = -R.T @ t
    return M


def _log_se3_vee(T: np.ndarray) -> np.ndarray:
    from scipy.linalg import logm

    A = logm(T)
    return np.concatenate([np.array([A[2, 1], A[0, 2], A[1, 0]]), A[:3, 3]])


def mat_to_js(M: np.ndarray) -> dict:
    return {
        "rotation": _rmat_to_quat_wxyz(M[:3, :3]),
        "translation": [float(v) for v in M[:3, 3]],
        "scale": [1.0, 1.0, 1.0],
    }


def js_to_mat(d: dict) -> np.ndarray:
    q = np.asarray(d["rotation"], dtype=np.float64)
    M = np.eye(4)
    M[:3, :3] = _qmat(q)
    M[:3, 3] = np.asarray(d["translation"], dtype=np.float64)
    return M


# --------------------------------------------------------------------------- #
# input loading
# --------------------------------------------------------------------------- #


def load_intrinsics(path: Path) -> tuple[np.ndarray, int, int]:
    d = json.loads(path.read_text())
    K = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1.0]], dtype=np.float64)
    return K, int(d["width"]), int(d["height"])


def load_depth_meters(path: Path) -> np.ndarray:
    from PIL import Image

    a = np.asarray(Image.open(str(path)), dtype=np.float64)
    return 65535.0 / np.maximum(a, 1.0) - 1.0


def load_mask(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(str(path)).convert("L")) > 0


def backproject_cloud(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame XYZ (OpenCV: +z forward) for mask pixels with finite depth."""
    import cv2

    yy, xx = np.nonzero(
        cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
    )
    z = depth[yy, xx]
    ok = np.isfinite(z) & (z > 0.05) & (z < 3.0)
    yy, xx, z = yy[ok], xx[ok], z[ok]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    pts = np.empty((z.size, 3), dtype=np.float32)
    pts[:, 0] = (xx - K[0, 2]) / K[0, 0] * z
    pts[:, 1] = (yy - K[1, 2]) / K[1, 1] * z
    pts[:, 2] = z
    return pts


def mesh_points_and_normals(path: Path, k: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    m = trimesh.load(str(path), force=None)
    if isinstance(m, trimesh.Scene):
        m = next(iter(m.geometry.values()))
    v = np.asarray(m.vertices, dtype=np.float64)
    vn = np.asarray(m.vertex_normals, dtype=np.float64)
    if len(v) > k:
        idx = np.random.default_rng(seed).choice(len(v), k, replace=False)
        v, vn = v[idx], vn[idx]
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = vn / np.maximum(n, 1e-9)
    return v, vn


def load_reference_loader(repo_root: Path):
    path = repo_root / "scripts" / "eval" / "reference_loader.py"
    spec = importlib.util.spec_from_file_location("reference_loader", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("reference_loader", mod)
    spec.loader.exec_module(mod)
    return mod


def ego_rig_matrix(rig: dict) -> tuple[str, str, np.ndarray]:
    sp = next((s for s in rig.get("stereo_pairs", []) if s.get("camera") == "ego"), None)
    if sp is None:
        raise ValueError("no ego stereo_pair in camera_calibration.json")
    return sp["left"], sp["right"], np.array(sp["left_to_right"], dtype=np.float64).reshape(4, 4)


def _w_obj(pq: np.ndarray) -> np.ndarray:
    """world->object (maps object coords -> world) from GT xyzw."""
    W = np.eye(4)
    W[:3, :3] = _qmat(pq[3:])
    W[:3, 3] = pq[:3]
    return W


# --------------------------------------------------------------------------- #
# optimisation core
# --------------------------------------------------------------------------- #


def _residual_metrics(mesh_cam: np.ndarray, n_cam: np.ndarray, q: np.ndarray,
                      tukey: float) -> dict:
    if len(q) == 0:
        return {"count": 0, "rmse": float("nan"), "inlier": 0.0}
    from scipy.spatial import cKDTree

    idx = np.clip(cKDTree(mesh_cam.astype(np.float32)).query(q.astype(np.float32), k=1)[1],
                  0, len(mesh_cam) - 1)
    res = np.einsum("ij,ij->i", n_cam.T[idx], q - mesh_cam[idx])
    return {
        "count": int(len(q)),
        "rmse": float(np.sqrt(np.mean(res ** 2))),
        "inlier": float((np.abs(res) <= tukey).mean()),
    }


def frame_cloud(mask_p: Path, depth_p: Path, intr_p: Path, max_pts: int, t: int) -> np.ndarray:
    if not (mask_p.exists() and depth_p.exists() and intr_p.exists()):
        return np.empty((0, 3), dtype=np.float32)
    mask = load_mask(mask_p)
    if mask.sum() < 20:
        return np.empty((0, 3), dtype=np.float32)
    K, _, _ = load_intrinsics(intr_p)
    q = backproject_cloud(load_depth_meters(depth_p), mask, K)
    if len(q) > max_pts:
        q = q[np.random.default_rng(t).choice(len(q), max_pts, replace=False)]
    return q


def solve_frame(
    T0: np.ndarray,
    mesh: np.ndarray,
    mesh_n: np.ndarray,
    M: np.ndarray,
    ql: np.ndarray,
    qr: np.ndarray,
    iters: int,
    tukey: float,
    start_sigma: float = 0.35,
) -> tuple[np.ndarray, dict]:
    """Single-unknown SE(3) T (world->cam_left). Joint GN over both cameras.

    Residual (point-to-plane, per camera):  r = n . (q - m(exp(xi) p)).
    Linearising m(xi) = p + w x p + v gives J = [n^T skew(p), -n^T] and the
    Gauss-Newton step del = -(J^T W J + damp)^-1 J^T W r with Tukey weights.
    """
    from scipy.spatial import cKDTree

    T = np.array(T0, dtype=np.float64)
    R_M = M[:3, :3]
    # coarse-to-fine Tukey: let far correspondences pull in early, refine late
    sched = np.geomspace(start_sigma, tukey, 8)
    for it in range(iters):
        sigma = float(sched[min(it, len(sched) - 1)])
        p_l = (T[:3, :3] @ mesh.T).T + T[:3, 3]            # mesh in cam_left
        p_r = (R_M @ p_l.T).T + M[:3, 3]                   # mesh in cam_right
        n_l = T[:3, :3] @ mesh_n.T                          # (3, N)
        n_r = R_M @ n_l

        As, rs, ws = [], [], []
        for q, p, n in ((ql, p_l, n_l), (qr, p_r, n_r)):
            if len(q) == 0:
                continue
            idx = np.clip(cKDTree(p.astype(np.float32)).query(q.astype(np.float32), k=1)[1],
                          0, len(p) - 1)
            pp, nn = p[idx], n.T[idx]
            r = np.einsum("ij,ij->i", nn, q - pp)
            w = np.maximum(1.0 - (r / sigma) ** 2, 0.0) ** 2
            nt = len(q)
            A = np.empty((nt, 6), dtype=np.float64)
            A[:, 3:] = -nn
            A[:, :3] = np.cross(nn, pp)     # -(skew(p) @ n) == n x p
            As.append(A)
            rs.append(r)
            ws.append(w)
        if not As:
            break
        A = np.concatenate(As)
        r = np.concatenate(rs)
        w = np.concatenate(ws)
        H = A.T @ (A * w[:, None]) + np.diag(1e-4 * np.ones(6))
        delta = -np.linalg.solve(H, A.T @ (w * r))
        if np.linalg.norm(delta[:3]) > 0.35:
            delta *= 0.35 / np.linalg.norm(delta[:3])
        T = _exp_se3(delta) @ T
        if np.linalg.norm(delta) < 1e-5:
            break

    p_l = (T[:3, :3] @ mesh.T).T + T[:3, 3]
    p_r = (R_M @ p_l.T).T + M[:3, 3]
    n_l = T[:3, :3] @ mesh_n.T
    n_r = R_M @ n_l
    return T, {
        "cam_left": _residual_metrics(p_l, n_l, ql, tukey),
        "cam_right": _residual_metrics(p_r, n_r, qr, tukey),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--objects", default="", help="comma list; default = all in parquet")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--mesh-pts", type=int, default=2500)
    ap.add_argument("--max-pts", type=int, default=3000)
    ap.add_argument("--tukey-m", type=float, default=0.010)
    ap.add_argument("--start-sigma", type=float, default=0.35,
                    help="coarse-to-fine Tukey start (m); keeps bad inits attracted")
    ap.add_argument("--pass-inlier", type=float, default=0.85)
    ap.add_argument("--cover2d-min", type=float, default=0.0,
                    help="min fraction of mesh verts projecting into the right-cam mask "
                         "(reported-only; right masks are often partial/hand-occluded)")
    ap.add_argument("--joint-right", action="store_true",
                    help="include the right-cam cloud in the solve (default: left-only; the "
                         "right camera stereo depth has a scale bug ~1.35x and would drag)")
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--reuse-fp", action="store_true", help="FP poses (work/poses) as extra inits")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    work = Path(args.work)
    rig_root = data_root / "meta" / "camera_calibration.json"
    if not rig_root.exists():
        sys.exit(f"{rig_root}: camera_calibration.json not found")
    rig = json.loads(rig_root.read_text())
    cam_left, cam_right, M = ego_rig_matrix(rig)

    loader = load_reference_loader(REPO_ROOT)
    ref = loader.load_reference(args.episode, str(data_root))
    objects = list(ref.object_names)
    obj_order = [o.strip() for o in args.objects.split(",") if o.strip()] or objects
    mesh_dirs = loader.object_mesh_dir(obj_order, str(data_root))

    print(f"episode {args.episode}: objects={obj_order} ego cams=({cam_left},{cam_right})")
    print(f"ego rig |t|={np.linalg.norm(M[:3, 3]):.4f} m")

    out_root = work / "ego_cam_poses"
    out_root.mkdir(parents=True, exist_ok=True)
    Tmax = int(ref.steps)
    per_obj_out: dict[str, dict] = {}

    for b, obj in enumerate(obj_order):
        mesh, mesh_n = mesh_points_and_normals(mesh_dirs[obj], args.mesh_pts)
        mesh = np.asarray(mesh, dtype=np.float64)
        mesh_n = np.asarray(mesh_n, dtype=np.float64)
        for cam in (cam_left, cam_right):
            (out_root / obj / "world_to_cam" / cam).mkdir(parents=True, exist_ok=True)
            (out_root / obj / "object_to_cam" / cam).mkdir(parents=True, exist_ok=True)
        Kr, _, _ = load_intrinsics(work / "intrinsics" / f"{cam_right}.json")
        Kc, _, _ = load_intrinsics(work / "intrinsics" / f"{cam_left}.json")
        print(f"[fit] {obj}: mesh {len(mesh)} pts")

        t_start = time.time()
        Ts: list[np.ndarray | None] = [None] * Tmax
        quals: list[dict | None] = [None] * Tmax
        prev: np.ndarray | None = None
        for t in range(Tmax):
            if t % 100 == 0:
                print(f"  frame {t}/{Tmax}")
            if not bool(ref.visible[t, b]):
                continue
            ql = frame_cloud(
                work / "masks" / cam_left / obj / "0" / f"{t:06d}.png",
                work / "depth" / cam_left / f"{t:06d}.png",
                work / "intrinsics" / f"{cam_left}.json", args.max_pts, t,
            )
            qr = (np.empty((0, 3), dtype=np.float32) if not args.joint_right else frame_cloud(
                work / "masks" / cam_right / obj / "0" / f"{t:06d}.png",
                work / "depth" / cam_right / f"{t:06d}.png",
                work / "intrinsics" / f"{cam_right}.json", args.max_pts, t,
            ))
            if len(ql) == 0 and len(qr) == 0:
                continue

            pq = np.asarray(ref.pose_xyzw[t, b], dtype=np.float64)  # xyzw
            W_obj_world = _w_obj(pq)
            W_inv = _inv4(W_obj_world)
            mesh_w = (W_obj_world[:3, :3] @ mesh.T).T + W_obj_world[:3, 3]
            mesh_nw = (W_obj_world[:3, :3] @ mesh_n.T).T

            inits: list[tuple[str, np.ndarray]] = []
            if prev is not None:
                inits.append(("propagate", prev))
            if args.reuse_fp:
                fp_l = work / "poses" / cam_left / obj / f"{t:06d}.json"
                if fp_l.exists():
                    inits.append(("fp_cam_c", js_to_mat(json.loads(fp_l.read_text())) @ W_inv))
                fp_r = work / "poses" / cam_right / obj / f"{t:06d}.json"
                if fp_r.exists():
                    T_wc_r = js_to_mat(json.loads(fp_r.read_text())) @ W_inv
                    inits.append(("fp_cam_b", _inv4(M) @ T_wc_r))
            if len(ql) > 0:
                T0 = np.eye(4)
                T0[:3, 3] = np.mean(ql, axis=0) - W_obj_world[:3, 3]
                inits.append(("centroid", T0))

            best_score, best = -1.0, None
            for label_candidate, T0 in inits:
                Tf, met = solve_frame(T0, mesh_w, mesh_nw, M, ql, qr, args.iters, args.tukey_m,
                                      args.start_sigma)
                n_cam = (met["cam_left"]["count"] > 0) + (met["cam_right"]["count"] > 0)
                combined = (met["cam_left"]["inlier"] if met["cam_left"]["count"] else 0.0) \
                    + (met["cam_right"]["inlier"] if met["cam_right"]["count"] else 0.0)
                score = combined / max(n_cam, 1)
                if score > best_score:
                    best_score = score
                    met["init"] = label_candidate
                    best = (Tf, met)
            if best is None:
                continue
            Tf, met = best
            Ts[t], quals[t], prev = Tf, met, Tf

        if not args.no_smooth and sum(x is not None for x in Ts) > 3:
            window, sigma = 4, 2.0
            for t in range(Tmax):
                if Ts[t] is None:
                    continue
                xi = np.zeros(6)
                wtot = 0.0
                for k in range(max(0, t - window), min(Tmax, t + window + 1)):
                    if Ts[k] is None:
                        continue
                    wgt = np.exp(-0.5 * ((k - t) / sigma) ** 2)
                    xi += wgt * _log_se3_vee(Ts[k] @ _inv4(Ts[t]))
                    wtot += wgt
                if wtot > 0:
                    Ts[t] = _exp_se3(xi / wtot) @ Ts[t]

        per_obj: dict = {"frames": Tmax, "solved": 0, "passed": 0, "framedetails": {}}
        for t in range(Tmax):
            if Ts[t] is None:
                continue
            pq = np.asarray(ref.pose_xyzw[t, b], dtype=np.float64)
            W_inv = _inv4(_w_obj(pq))
            T_lft, T_rgt = Ts[t], M @ Ts[t]
            per_obj["solved"] += 1
            met = quals[t]
            inls = (met["cam_left"]["inlier"], met["cam_right"]["inlier"])
            flags = (met["cam_left"]["count"] > 0, met["cam_right"]["count"] > 0)
            # right-cam 2D coverage: fraction of mesh verts (rig-projected) inside mask
            cover2d = None
            mask_rp = work / "masks" / cam_right / obj / "0" / f"{t:06d}.png"
            if mask_rp.exists():
                im_r = load_mask(mask_rp)
                if im_r.sum() > 0:
                    p_r = (T_rgt[:3, :3] @ mesh_w.T).T + T_rgt[:3, 3]
                    u = Kr[0, 0] * p_r[:, 0] / p_r[:, 2] + Kr[0, 2]
                    v = Kr[1, 1] * p_r[:, 1] / p_r[:, 2] + Kr[1, 2]
                    ok = (u >= 0) & (u < Kr[0, 2] * 2) & (v >= 0) & (v < 2 * Kr[1, 2])
                    if ok.sum():
                        cover2d = float(im_r[np.clip(v[ok].astype(int), 0, 799),
                                            np.clip(u[ok].astype(int), 0, 1279)].mean())
            passed = bool(
                met["cam_left"]["count"] > 0
                and met["cam_left"]["inlier"] >= args.pass_inlier
                and (cover2d is None or cover2d >= args.cover2d_min)
            )
            for cam, Twc in ((cam_left, T_lft), (cam_right, T_rgt)):
                (out_root / obj / "world_to_cam" / cam / f"{t:06d}.json").write_text(
                    json.dumps(mat_to_js(Twc), indent=2))
                (out_root / obj / "object_to_cam" / cam / f"{t:06d}.json").write_text(
                    json.dumps(mat_to_js(Twc @ W_inv), indent=2))
            per_obj["framedetails"][str(t)] = {
                "inlier_left": (inls[0] if flags[0] else None),
                "inlier_right": (inls[1] if flags[1] else None),
                "rmse_left": met["cam_left"]["rmse"],
                "rmse_right": met["cam_right"]["rmse"],
                "count_left": met["cam_left"]["count"],
                "count_right": met["cam_right"]["count"],
                "cover_right_2d": cover2d,
                "pass": passed,
                "init": met["init"],
            }
            if passed:
                per_obj["passed"] += 1
        per_obj["elapsed_s"] = round(time.time() - t_start, 1)
        per_obj_out[obj] = per_obj
        print(f"  {obj}: solved {per_obj['solved']}/{per_obj['frames']} "
              f"passed {per_obj['passed']} ({per_obj['elapsed_s']:.1f}s)")

    pass_frames = {
        o: {int(t) for t, d in per_obj_out[o]["framedetails"].items() if d.get("pass")}
        for o in obj_order
    }
    cross: dict = {}
    if len(obj_order) >= 2:
        common = sorted(pass_frames[obj_order[0]] & pass_frames[obj_order[1]])
        angs, dists = [], []
        for t in common:
            Ta = js_to_mat(json.loads(
                (out_root / obj_order[0] / "world_to_cam" / cam_left / f"{t:06d}.json").read_text()))
            Tb = js_to_mat(json.loads(
                (out_root / obj_order[1] / "world_to_cam" / cam_left / f"{t:06d}.json").read_text()))
            Del = _inv4(Ta) @ Tb
            angs.append(float(np.degrees(np.arccos(np.clip((np.trace(Del[:3, :3]) - 1) / 2, -1, 1)))))
            dists.append(float(np.linalg.norm(Del[:3, 3])))
        if common:
            cross = {
                "frames": len(common),
                "median_angle_deg": float(np.median(angs)),
                "median_disp_m": float(np.median(dists)),
                "max_angle_deg": float(np.max(angs)),
                "max_disp_m": float(np.max(dists)),
            }

    report = {
        "episode": args.episode,
        "cam_left": cam_left,
        "cam_right": cam_right,
        "baseline_m": round(float(np.linalg.norm(M[:3, 3])), 4),
        "params": {"iters": args.iters, "mesh_pts": args.mesh_pts,
                   "tukey_m": args.tukey_m, "start_sigma_m": args.start_sigma,
                   "pass_inlier": args.pass_inlier, "cover2d_min": args.cover2d_min,
                   "joint_right": args.joint_right,
                   "smoothing": not args.no_smooth, "reuse_fp": args.reuse_fp},
        "cross_object_agreement_left": cross,
        "objects": per_obj_out,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    tot_solved = int(sum(o["solved"] for o in per_obj_out.values()))
    tot_pass = int(sum(o["passed"] for o in per_obj_out.values()))
    print(f"\nSUMMARY solved={tot_solved} passed={tot_pass}")
    print(f"cross-object agreement: {cross}")
    print(f"report: {out_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())