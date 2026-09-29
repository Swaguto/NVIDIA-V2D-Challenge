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

Outputs under ``{work}/ego_object_poses/e{episode}/``:
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
    ego_rig_matrix,
    frame_cloud,
    js_to_mat,
    load_reference_loader,
    mat_to_js,
    mesh_points_and_normals,
    solve_frame,
)


# --------------------------------------------------------------------------- #
# scoring helpers
# --------------------------------------------------------------------------- #


def _pose_error(X: np.ndarray, W_gt: np.ndarray) -> tuple[float, float]:
    """Return rotation error in degrees and translation error in metres."""
    Del = _inv4(W_gt) @ X
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Del[:3, :3]) - 1) / 2, -1, 1))))
    disp = float(np.linalg.norm(Del[:3, 3]))
    return ang, disp


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Return symmetric mean nearest-neighbour distance in mm, or NaN if empty."""
    from scipy.spatial import cKDTree

    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d_ab = cKDTree(a.astype(np.float32)).query(b.astype(np.float32), k=1)[0]
    d_ba = cKDTree(b.astype(np.float32)).query(a.astype(np.float32), k=1)[0]
    return float(1000.0 * (d_ab.mean() + d_ba.mean()) / 2.0)


def _summarize_gt(angles: list[float], displacements: list[float],
                  chamfers: list[float], n_visible: int) -> dict:
    """Aggregate solved visible frames; retain coverage even when none solve.

    Errors and pass rate are conditional on solved frames. Coverage uses all
    GT-visible frames; unavailable errors/rates are JSON null, never zero.
    """
    score = {
        "n_solved_gt": len(angles),
        "n_visible": n_visible,
        "coverage_visible": round(len(angles) / n_visible, 3) if n_visible else None,
        "median_rot_deg": None,
        "median_trans_m": None,
        "median_chamfer_mm": None,
        "p90_rot_deg": None,
        "p90_trans_m": None,
        "pass_5cm_10deg": None,
    }
    if angles:
        score.update({
            "median_rot_deg": float(np.median(angles)),
            "median_trans_m": float(np.median(displacements)),
            "median_chamfer_mm": float(np.median(chamfers)),
            "p90_rot_deg": float(np.percentile(angles, 90)),
            "p90_trans_m": float(np.percentile(displacements, 90)),
            "pass_5cm_10deg": float(np.mean(
                [a <= 10.0 and d <= 0.05 for a, d in zip(angles, displacements)])),
        })
    return score


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def main() -> int:
    """Fit object poses from fixed camera inputs and write per-episode reports."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--objects", default="", help="comma list; default = all in parquet")
    ap.add_argument("--pose-obj", default="auto",
                    help="which object's fixed world_to_cam poses seed the solve. "
                         "'auto' (default) gives each object its own rig, falling "
                         "back to the first available one when a per-object rig is "
                         "missing. Otherwise name a single object to share.")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--mesh-pts", type=int, default=2500)
    ap.add_argument("--max-pts", type=int, default=3000)
    ap.add_argument("--tukey-m", type=float, default=0.010)
    ap.add_argument("--start-sigma", type=float, default=0.35)
    ap.add_argument("--pass-inlier", type=float, default=0.85)
    ap.add_argument("--joint-right", action="store_true",
                    help="also constrain P with the right-cam cloud (scaled by "
                         "--right-depth-scale); the FS depth-scale bug ~1.35x applies")
    ap.add_argument("--right-depth-scale", type=float, default=1.0,
                    help="mult on right-cam depth. 1.0 is known-wrong: FoundationStereo "
                         "inflates the right view ~1.35x (issue #23). Use "
                         "--est-right-scale to derive it, or pass ~0.72 by hand.")
    ap.add_argument("--jump-deg", type=float, default=40.0, help="max ok temporal jump")
    ap.add_argument("--jump-m", type=float, default=0.30, help="max ok temporal jump")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--score-gt", action="store_true",
                    help="load parquet GT: drive frames by GT visibility, report errors")
    mode.add_argument("--no-gt", action="store_true",
                    help="eval split: no parquet; process frames where the mask exists")
    ap.add_argument("--no-smooth", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    work = Path(args.work)

    rig = json.loads((data_root / "meta" / "camera_calibration.json").read_text())
    cam_left, cam_right, M = ego_rig_matrix(rig)

    loader = load_reference_loader(Path(__file__).resolve().parents[4])
    ref = loader.load_reference(args.episode, str(data_root)) if args.score_gt else None
    objects = list(ref.object_names) if ref is not None else []
    obj_order = [o.strip() for o in args.objects.split(",") if o.strip()] or objects
    if not obj_order:
        sys.exit("--no-gt: no parquet; pass an explicit --objects list")
    if ref is not None:
        unknown = set(obj_order) - set(objects)
        if unknown:
            ap.error(f"objects absent from episode GT: {sorted(unknown)}")
    mesh_dirs = loader.object_mesh_dir(obj_order, str(data_root))

    # Shared rig root, used when an object has no rig of its own. Resolved before
    # any reporting so the printout below never names a literal "auto".
    if args.pose_obj == "auto":
        avail = [o for o in obj_order
                 if (work / "ego_cam_poses" / o / "world_to_cam" / cam_left).exists()]
        if not avail:
            sys.exit(f"no world_to_cam under {work / 'ego_cam_poses'}/<obj>/ "
                     f"for any of {obj_order} (run fit_ego_rig first)")
        shared = avail[0]
    else:
        if args.pose_obj not in obj_order:
            sys.exit(f"--pose-obj {args.pose_obj} not in {obj_order}")
        shared = args.pose_obj
    cam_root = work / "ego_cam_poses" / shared / "world_to_cam"

    print(f"episode {args.episode}: objects={obj_order} cams=({cam_left},{cam_right})")
    print(f"pose source: {'per-object' if args.pose_obj == 'auto' else 'shared ' + shared}"
          f"  rig |t|={np.linalg.norm(M[:3, 3]):.4f} m")

    out_root = work / "ego_object_poses"
    ep_dir = out_root / f"e{args.episode:03d}"
    Tmax = int(ref.steps) if ref is not None else -1
    per_obj_out: dict[str, dict] = {}

    for obj in obj_order:
        b = objects.index(obj) if ref is not None else None
        mesh_o, mesh_on = mesh_points_and_normals(mesh_dirs[obj], args.mesh_pts)
        mesh_o = np.asarray(mesh_o, dtype=np.float64)
        mesh_on = np.asarray(mesh_on, dtype=np.float64)
        for cam in (cam_left, cam_right):
            (ep_dir / obj / "object_to_world").mkdir(parents=True, exist_ok=True)
            (ep_dir / obj / "object_to_cam" / cam).mkdir(parents=True, exist_ok=True)
        c_o = mesh_o.mean(axis=0)
        Tmax_o = Tmax
        if Tmax_o < 0:
            frames = sorted(int(p.stem) for p in
                            (work / "masks" / cam_left / obj / "0").glob("[0-9]*.png"))
            Tmax_o = (frames[-1] + 1) if frames else 0
        # Rig poses are solved per object by fit_ego_rig, and they differ in
        # quality: driving every object from one object's rig couples them and
        # lets a bad rig contaminate a good one. Prefer this object's own rig.
        if args.pose_obj == "auto":
            own = work / "ego_cam_poses" / obj / "world_to_cam"
            pose_root = own if (own / cam_left).exists() else cam_root
        else:
            pose_root = cam_root
        if pose_root is not cam_root:
            print(f"[pose] {obj}: own rig {pose_root.parent.parent.name}")

        print(f"[fit] {obj}: mesh {len(mesh_o)} pts, frames {Tmax_o}")

        t_start = time.time()
        n_vis = int(ref.visible[:, b].sum()) if ref is not None else Tmax_o
        Xs: list[np.ndarray | None] = [None] * Tmax_o
        quals: list[dict | None] = [None] * Tmax_o
        prev: np.ndarray | None = None
        used_rig: dict[int, Path] = {}
        for t in range(Tmax_o):
            if ref is not None and not args.no_gt and not bool(ref.visible[t, b]):
                continue
            # An object may have its own rig, but that rig covers only the frames
            # fit_ego_rig actually solved. Fall back to the shared rig per frame
            # rather than dropping the frame, so a partial per-object rig
            # degrades to shared coverage instead of losing frames.
            wc_l = pose_root / cam_left / f"{t:06d}.json"
            if not wc_l.exists():
                alt = cam_root / cam_left / f"{t:06d}.json"
                if not alt.exists():
                    continue
                wc_l = alt
            C_l = js_to_mat(json.loads(wc_l.read_text()))
            used_rig[t] = wc_l
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
                rot0 = ((C_l @ prev)[:3, :3] if prev is not None else np.eye(3))
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
            Xs[t], quals[t] = _inv4(C_l) @ Pf, met
            prev = Xs[t]

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
            C_l = js_to_mat(json.loads(used_rig[t].read_text()))
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
            (ep_dir / obj / "object_to_world" / f"{t:06d}.json").write_text(
                json.dumps(mat_to_js(X), indent=2))
            for cam, Pc in ((cam_left, Pf), (cam_right, M @ Pf)):
                (ep_dir / obj / "object_to_cam" / cam / f"{t:06d}.json").write_text(
                    json.dumps(mat_to_js(Pc), indent=2))

            if args.score_gt and not args.no_gt:
                pq = np.asarray(ref.pose_xyzw[t, b], dtype=np.float64)
                # Reference poses are xyzw; Transform3d expects wxyz.
                W_gt = js_to_mat({"rotation": pq[[6, 3, 4, 5]],
                                  "translation": pq[:3]})
                ang, disp = _pose_error(X, W_gt)
                mesh_w_fit = (X[:3, :3] @ mesh_o.T).T + X[:3, 3]
                mesh_w_gt = (W_gt[:3, :3] @ mesh_o.T).T + W_gt[:3, 3]
                chm = _chamfer(mesh_w_fit, mesh_w_gt)
                meta["rot_err_deg"] = round(ang, 3)
                meta["trans_err_m"] = round(disp, 4)
                meta["chamfer_mm"] = round(chm, 3)
                meta["gt_pass"] = bool(ang <= 10.0 and disp <= 0.05)
                gt_ang.append(ang)
                gt_disp.append(disp)
                gt_chm.append(chm)
            per_obj["framedetails"][str(t)] = meta
            if passed:
                per_obj["passed"] += 1
        if args.score_gt:
            per_obj["gt_score"] = _summarize_gt(gt_ang, gt_disp, gt_chm, n_vis)
        per_obj["elapsed_s"] = round(time.time() - t_start, 1)
        per_obj_out[obj] = per_obj
        print(f"  {obj}: solved {per_obj['solved']} passed {per_obj['passed']} "
              f"({per_obj['elapsed_s']:.1f}s) {per_obj['gt_score']}")

    report = {
        "episode": args.episode,
        "cam_left": cam_left,
        "cam_right": cam_right,
        "pose_source": "per-object" if args.pose_obj == "auto" else f"ego_cam_poses/{shared}",
        "params": {"iters": args.iters, "mesh_pts": args.mesh_pts,
                   "tukey_m": args.tukey_m, "start_sigma_m": args.start_sigma,
                   "pass_inlier": args.pass_inlier, "joint_right": args.joint_right,
                   "right_depth_scale": args.right_depth_scale,
                   "jump_deg": args.jump_deg, "jump_m": args.jump_m,
                   "smoothing": not args.no_smooth},
        "objects": per_obj_out,
    }
    (ep_dir / "report.json").write_text(json.dumps(report, indent=2))
    tot_solved = int(sum(o["solved"] for o in per_obj_out.values()))
    tot_pass = int(sum(o["passed"] for o in per_obj_out.values()))
    print(f"\nSUMMARY solved={tot_solved} passed={tot_pass}")
    print(f"report: {ep_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
