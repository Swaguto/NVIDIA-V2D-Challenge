#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Estimate per-frame object->world poses from CAD meshes + masks + depth.

Issue5 / [A2] Object tracking from CAD templates.  Inverse of fit_ego_rig:
there the world->cam pose is the unknown and the object pose is fixed by GT;
here the world->cam poses are the verified W1 output (work/ego_cam_poses)
and the unknown is one SE(3) ``X = T_WO`` (object->world) per frame and per
object, with the mutually registered mesh as the CAD template.

  For a mesh point ``p`` in object coords:
    cam_left:  m_l = (T_WC_left @ X) @ p
    cam_right: m_r = M @ (T_WC_left @ X) @ p

so the unknown reduces to ``P = T_WC_left @ X`` (object->cam_left) and the
same joint Levenberg-Gauss-Newton point-to-plane solver as fit_ego_rig
applies verbatim (solve_frame already solves "body->cam_left").  Both
cameras constrain the same P through the fixed rig (M), so rotation is well
identified even for textureless objects (white_pot / white_pot_lid).

Outputs under ``{work}/ego_object_poses``:
  {obj}/object_to_world/{frame:06d}.json   Transform3d object->world (w-first)
  {obj}/object_to_cam/{cam}/{frame:06d}.json
  report.json                              metrics + per-frame pass list

Run on the VM inside the pipeline venv (CPU only; needs scipy + trimesh):

  python -m v2d.world_calib.vm.fit_object_pose \
      --data-root ~/v2d_track3/data/hf/track_3/public \
      --work ~/v2d_calib_work --episode 2 --objects white_pot \
      --score-gt                        # public split: score vs parquet GT
      [--no-gt]                         # eval split: no parquet; masks drive frames

When --score-gt and GT is visible, per-frame rot/trans/chamfer errors vs the
parquet world_T_object are accumulated into report.json under "gt_score".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from v2d.world_calib.vm.fit_ego_rig import (
    _exp_se3,
    _inv4,
    _log_se3_vee,
    _w_obj,
    ego_rig_matrix,
    frame_cloud,
    js_to_mat,
    load_intrinsics,
    load_reference_loader,
    mat_to_js,
    mesh_points_and_normals,
    solve_frame,
)


# --------------------------------------------------------------------------- #
# scoring helpers
# --------------------------------------------------------------------------- #


def _pose_error(X: np.ndarray, W_gt: np.ndarray) -> tuple[float, float]:
    Del = _inv4(W_gt) @ X
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Del[:3, :3]) - 1) / 2, -1, 1))))
    disp = float(np.linalg.norm(Del[:3, 3]))
    return ang, disp


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d_ab = cKDTree(a.astype(np.float32)).query(b.astype(np.float32), k=1)[0]
    d_ba = cKDTree(b.astype(np.float32)).query(a.astype(np.float32), k=1)[0]
    return float(1000.0 * (d_ab.mean() + d_ba.mean()) / 2.0)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--objects", default="", help="comma list; default = all in parquet")
    ap.add_argument("--pose-obj", default="",
                    help="object whose fixed world_to_cam poses seed the solve "
                         "(default: first of --objects)")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--mesh-pts", type=int, default=2500)
    ap.add_argument("--max-pts", type=int, default=3000)
    ap.add_argument("--tukey-m", type=float, default=0.010)
    ap.add_argument("--start-sigma", type=float, default=0.35)
    ap.add_argument("--pass-inlier", type=float, default=0.85)
    ap.add_argument("--joint-right", action="store_true",
                    help="also constrain P with the right-cam cloud (scaled by "
                         "--right-depth-scale); the FS depth-scale bug ~1.35x applies")
    ap.add_argument("--right-depth-scale", type=float, default=1.0)
    ap.add_argument("--jump-deg", type=float, default=40.0, help="max ok temporal jump")
    ap.add_argument("--jump-m", type=float, default=0.30, help="max ok temporal jump")
    ap.add_argument("--score-gt", action="store_true",
                    help="load parquet GT: drive frames by GT visibility, report errors")
    ap.add_argument("--no-gt", action="store_true",
                    help="eval split: no parquet; process frames where the mask exists")
    ap.add_argument("--no-smooth", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    work = Path(args.work)
    if not args.score_gt and not args.no_gt:
        sys.exit("need --score-gt (public) or --no-gt (eval)")

    rig = json.loads((data_root / "meta" / "camera_calibration.json").read_text())
    cam_left, cam_right, M = ego_rig_matrix(rig)

    loader = load_reference_loader(Path(__file__).resolve().parents[4])
    ref = loader.load_reference(args.episode, str(data_root)) if args.score_gt else None
    objects = list(ref.object_names) if ref is not None else []
    obj_order = [o.strip() for o in args.objects.split(",") if o.strip()] or objects
    if not obj_order:
        sys.exit("--no-gt: no parquet; pass an explicit --objects list")
    mesh_dirs = loader.object_mesh_dir(obj_order, str(data_root))
    pose_obj = args.pose_obj or obj_order[0]

    print(f"episode {args.episode}: objects={obj_order} cams=({cam_left},{cam_right})")
    print(f"pose source: ego_cam_poses/{pose_obj}  rig |t|={np.linalg.norm(M[:3, 3]):.4f} m")

    cam_root = work / "ego_cam_poses" / pose_obj / "world_to_cam"
    if not (cam_root / cam_left).exists():
        sys.exit(f"{cam_root / cam_left}: fixed world_to_cam poses missing (run fit_ego_rig)")

    out_root = work / "ego_object_poses"
    out_root.mkdir(parents=True, exist_ok=True)
    Tmax = int(ref.steps) if ref is not None else -1
    per_obj_out: dict[str, dict] = {}

    for b, obj in enumerate(obj_order):
        mesh_o, mesh_on = mesh_points_and_normals(mesh_dirs[obj], args.mesh_pts)
        mesh_o = np.asarray(mesh_o, dtype=np.float64)
        mesh_on = np.asarray(mesh_on, dtype=np.float64)
        for cam in (cam_left, cam_right):
            (out_root / obj / "object_to_world").mkdir(parents=True, exist_ok=True)
            (out_root / obj / "object_to_cam" / cam).mkdir(parents=True, exist_ok=True)
        c_o = mesh_o.mean(axis=0)
        Tmax_o = Tmax
        if Tmax_o < 0:
            frames = sorted(int(p.stem) for p in
                            (work / "masks" / cam_left / obj / "0").glob("[0-9]*.png"))
            Tmax_o = (frames[-1] + 1) if frames else 0
        print(f"[fit] {obj}: mesh {len(mesh_o)} pts, frames {Tmax_o}")

        t_start = time.time()
        Xs: list[np.ndarray | None] = [None] * Tmax_o
        quals: list[dict | None] = [None] * Tmax_o
        prev: np.ndarray | None = None
        for t in range(Tmax_o):
            if ref is not None and not args.no_gt and not bool(ref.visible[t, b]):
                continue
            wc_l = cam_root / cam_left / f"{t:06d}.json"
            if not wc_l.exists():
                continue
            C_l = js_to_mat(json.loads(wc_l.read_text()))
            ql = frame_cloud(
                work / "masks" / cam_left / obj / "0" / f"{t:06d}.png",
                work / "depth" / cam_left / f"{t:06d}.png",
                work / "intrinsics" / f"{cam_left}.json", args.max_pts, t,
            )
            qr = (np.empty((0, 3), dtype=np.float32) if not args.joint_right else frame_cloud(
                work / "masks" / cam_right / obj / "0" / f"{t:06d}.png",
                work / "depth" / cam_right / f"{t:06d}.png",
                work / "intrinsics" / f"{cam_right}.json", args.max_pts, t,
                scale=args.right_depth_scale,
            ))
            if len(ql) == 0 and len(qr) == 0:
                continue

            P0s: list[tuple[str, np.ndarray]] = []
            if prev is not None:
                P0s.append(("propagate", C_l @ prev))
            if len(ql) > 0:
                Pc = np.eye(4)
                rot0 = (prev[:3, :3] if prev is not None else np.eye(3))
                Pc[:3, :3] = rot0
                Pc[:3, 3] = ql.mean(axis=0) - rot0 @ c_o
                P0s.append(("centroid", Pc))

            best_score, best = -1.0, None
            for label, P0 in P0s:
                Pf, met = solve_frame(P0, mesh_o, mesh_on, M, ql, qr,
                                      args.iters, args.tukey_m, args.start_sigma)
                n_cam = (met["cam_left"]["count"] > 0) + (met["cam_right"]["count"] > 0)
                combined = (met["cam_left"]["inlier"] if met["cam_left"]["count"] else 0.0) \
                    + (met["cam_right"]["inlier"] if met["cam_right"]["count"] else 0.0)
                score = combined / max(n_cam, 1)
                if score > best_score:
                    best_score = score
                    met["init"] = label
                    best = (Pf, met)
            if best is None:
                continue
            Pf, met = best
            Xs[t], quals[t], prev = _inv4(C_l) @ Pf, met, Pf

        if not args.no_smooth and sum(x is not None for x in Xs) > 3:
            window, sigma = 4, 2.0
            for t in range(len(Xs)):
                if Xs[t] is None:
                    continue
                xi = np.zeros(6)
                wtot = 0.0
                for k in range(max(0, t - window), min(len(Xs), t + window + 1)):
                    if Xs[k] is None:
                        continue
                    wgt = np.exp(-0.5 * ((k - t) / sigma) ** 2)
                    xi += wgt * _log_se3_vee(Xs[k] @ _inv4(Xs[t]))
                    wtot += wgt
                if wtot > 0:
                    Xs[t] = _exp_se3(xi / wtot) @ Xs[t]
            for t in range(len(Xs)):
                if Xs[t] is not None and quals[t] is not None:
                    quals[t]["init"] = quals[t].get("init", "smooth")

        per_obj: dict = {"frames": Tmax_o, "solved": 0, "passed": 0, "framedetails": {},
                         "gt_score": {}}
        gt_ang, gt_disp, gt_chm = [], [], []
        for t in range(len(Xs)):
            if Xs[t] is None:
                continue
            C_l = js_to_mat(json.loads((cam_root / cam_left / f"{t:06d}.json").read_text()))
            X, met = Xs[t], quals[t]
            per_obj["solved"] += 1
            Pf = C_l @ X
            jump_ok = True
            if t > 0 and quals[t - 1] is not None and Xs[t - 1] is not None:
                Del = _inv4(Xs[t - 1]) @ X
                ang = float(np.degrees(np.arccos(np.clip((np.trace(Del[:3, :3]) - 1) / 2, -1, 1))))
                disp = float(np.linalg.norm(Del[:3, 3]))
                jump_ok = ang <= args.jump_deg and disp <= args.jump_m
            inlier_ok = bool(
                met["cam_left"]["count"] > 0 and met["cam_left"]["inlier"] >= args.pass_inlier
            )
            right_ok = True
            if args.joint_right and met["cam_right"]["count"] > 0:
                right_ok = met["cam_right"]["inlier"] >= 0.7 * args.pass_inlier
            passed = inlier_ok and right_ok and jump_ok

            meta: dict = {
                "inlier_left": met["cam_left"]["inlier"],
                "inlier_right": met["cam_right"]["inlier"],
                "count_left": met["cam_left"]["count"],
                "count_right": met["cam_right"]["count"],
                "jump_ok": jump_ok,
                "init": met["init"],
                "pass": passed,
            }
            (out_root / obj / "object_to_world" / f"{t:06d}.json").write_text(
                json.dumps(mat_to_js(X), indent=2))
            for cam, Pc in ((cam_left, Pf), (cam_right, M @ Pf)):
                (out_root / obj / "object_to_cam" / cam / f"{t:06d}.json").write_text(
                    json.dumps(mat_to_js(Pc), indent=2))

            if args.score_gt and not args.no_gt:
                pq = np.asarray(ref.pose_xyzw[t, b], dtype=np.float64)
                W_gt = _w_obj(pq)
                ang, disp = _pose_error(X, W_gt)
                mesh_w_fit = (X[:3, :3] @ mesh_o.T).T + X[:3, 3]
                mesh_w_gt = (W_gt[:3, :3] @ mesh_o.T).T + W_gt[:3, 3]
                chm = _chamfer(mesh_w_fit, mesh_w_gt)
                meta["rot_err_deg"] = round(ang, 3)
                meta["trans_err_m"] = round(disp, 4)
                meta["chamfer_mm"] = round(chm, 3)
                meta["gt_pass"] = bool(ang <= 10.0 and disp <= 0.05 and chm <= 15.0)
                if passed:
                    gt_ang.append(ang)
                    gt_disp.append(disp)
                    gt_chm.append(chm)
            per_obj["framedetails"][str(t)] = meta
            if passed:
                per_obj["passed"] += 1
        if gt_ang:
            per_obj["gt_score"] = {
                "n": len(gt_ang),
                "median_rot_deg": float(np.median(gt_ang)),
                "median_trans_m": float(np.median(gt_disp)),
                "median_chamfer_mm": float(np.median(gt_chm)),
                "p90_rot_deg": float(np.percentile(gt_ang, 90)),
                "p90_trans_m": float(np.percentile(gt_disp, 90)),
                "pass_5cm_10deg": float(np.mean(
                    [a <= 10.0 and d <= 0.05 for a, d in zip(gt_ang, gt_disp)])),
            }
        per_obj["elapsed_s"] = round(time.time() - t_start, 1)
        per_obj_out[obj] = per_obj
        print(f"  {obj}: solved {per_obj['solved']} passed {per_obj['passed']} "
              f"({per_obj['elapsed_s']:.1f}s) {per_obj['gt_score']}")

    report = {
        "episode": args.episode,
        "cam_left": cam_left,
        "cam_right": cam_right,
        "pose_source": f"ego_cam_poses/{pose_obj}",
        "params": {"iters": args.iters, "mesh_pts": args.mesh_pts,
                   "tukey_m": args.tukey_m, "start_sigma_m": args.start_sigma,
                   "pass_inlier": args.pass_inlier, "joint_right": args.joint_right,
                   "right_depth_scale": args.right_depth_scale,
                   "jump_deg": args.jump_deg, "jump_m": args.jump_m,
                   "smoothing": not args.no_smooth},
        "objects": per_obj_out,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    tot_solved = int(sum(o["solved"] for o in per_obj_out.values()))
    tot_pass = int(sum(o["passed"] for o in per_obj_out.values()))
    print(f"\nSUMMARY solved={tot_solved} passed={tot_pass}")
    print(f"report: {out_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())