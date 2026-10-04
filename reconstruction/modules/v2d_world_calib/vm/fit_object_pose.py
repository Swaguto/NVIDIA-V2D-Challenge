#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Estimate per-frame object->world poses from CAD meshes + masks + depth.

Issue5 / [A2] Object tracking from CAD templates.  Inverse of fit_ego_rig:
there the world->cam pose is the unknown and the object pose is fixed by GT;
here one shared time-varying world->cam trajectory is supplied by W1
and the unknown is one SE(3) ``X = T_WO`` (object->world) per frame and per
object, with the mutually registered mesh as the CAD template.

  For a mesh point ``p`` in object coords:
    cam_left:  m_l = (T_WC_left @ X) @ p
    cam_right: m_r = M @ (T_WC_left @ X) @ p

so the unknown reduces to ``P = T_WC_left @ X`` (object->cam_left) and the
same joint Levenberg-Gauss-Newton point-to-plane solver as fit_ego_rig
applies verbatim (solve_frame already solves "body->cam_left").  Both
cameras constrain the same P through the fixed rig (M). Rotation observability
still depends on visible geometry; symmetric spin can remain ambiguous.

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
import hashlib
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
    pose_metrics,
    inlier_pass,
    smooth_poses,
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


def temporal_check(previous, current, frame_delta=1, time_delta=None, fps=None,
                   jump_deg=40.0, jump_m=0.30):
    """Gate velocity using actual elapsed time, or source-frame distance when unknown."""
    if previous is None:
        return True
    if frame_delta <= 0 or (time_delta is not None and time_delta <= 0):
        raise ValueError("frames and timestamps must increase strictly")
    scale = time_delta * fps if time_delta is not None and fps else frame_delta
    angle, displacement = _pose_error(current, previous)
    return bool(angle <= jump_deg * scale and displacement <= jump_m * scale)


def main() -> int:
    """Fit object poses from fixed camera inputs and write per-episode reports."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", type=Path, help="fresh episode output directory")
    ap.add_argument("--camera-root", type=Path,
                    help="one shared world_to_cam root containing <camera>/<frame>.json")
    ap.add_argument("--require-independent", action="store_true",
                    help="reject cameras lacking explicit independent GT-use provenance")
    ap.add_argument("--frame-times", type=Path, help="JSON mapping source frame IDs to capture times")
    ap.add_argument("--max-gap-frames", type=int, default=10,
                    help="reinitialize after this many source frames without an accepted fit")
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--objects", default="", help="comma list; default = all in parquet")
    ap.add_argument("--pose-obj", default="auto",
                    help="single camera source shared by every object; auto selects the first available")
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
    smoothing = ap.add_mutually_exclusive_group()
    smoothing.add_argument("--smooth", action="store_true")
    smoothing.add_argument("--no-smooth", action="store_true", help="raw poses (default)")
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
    if args.camera_root:
        shared = str(args.camera_root)
        cam_root = args.camera_root
    elif args.pose_obj == "auto":
        avail = [o for o in obj_order
                 if (work / "ego_cam_poses" / o / "world_to_cam" / cam_left).exists()]
        if not avail:
            sys.exit(f"no world_to_cam under {work / 'ego_cam_poses'}/<obj>/ "
                     f"for any of {obj_order} (run fit_ego_rig first)")
        shared = avail[0]
        cam_root = work / "ego_cam_poses" / shared / "world_to_cam"
    else:
        shared = args.pose_obj
        cam_root = work / "ego_cam_poses" / shared / "world_to_cam"

    print(f"episode {args.episode}: objects={obj_order} cams=({cam_left},{cam_right})")
    print(f"pose source: shared {shared}"
          f"  rig |t|={np.linalg.norm(M[:3, 3]):.4f} m")

    out_root = work / "ego_object_poses"
    ep_dir = args.out or out_root / f"e{args.episode:03d}"
    if ep_dir.exists() and any(ep_dir.iterdir()):
        ap.error("episode output already exists; pass --out with a new directory")
    if len(set(obj_order)) != len(obj_order):
        ap.error("object names must be distinct")
    if ref is not None:
        frame_ids = np.asarray(getattr(ref, "frame_index", np.arange(ref.steps)), dtype=int)
        times = dict(zip(frame_ids, ref.capture_time)) if hasattr(ref, "capture_time") else {}
    else:
        frame_ids = np.array(sorted({int(p.stem) for obj in obj_order for p in
            (work / "masks" / cam_left / obj / "0").glob("[0-9]*.png")}), dtype=int)
        times = {}
    if args.frame_times:
        times = {int(k): float(v) for k, v in json.loads(args.frame_times.read_text()).items()}
    if len(frame_ids) and (np.diff(frame_ids) <= 0).any():
        ap.error("source frame IDs must increase strictly")
    Tmax = len(frame_ids)
    fps = getattr(ref, "fps", None)
    info_path = data_root / "meta" / "info.json"
    if fps is None and info_path.exists():
        fps = json.loads(info_path.read_text()).get("fps")
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        ap.error("metadata fps must be finite and positive")
    if any(not np.isfinite(value) for value in times.values()):
        ap.error("capture times must be finite")
    cameras, camera_sources = {}, {}
    for frame in frame_ids:
        path = cam_root / cam_left / f"{frame:06d}.json"
        if not path.exists():
            continue
        raw = path.read_bytes()
        record = json.loads(raw)
        if record.get("status", "observed") != "observed":
            continue
        if record.get("frame_index", int(frame)) != int(frame):
            ap.error(f"camera frame ID mismatch: {path}")
        if record.get("episode", args.episode) != args.episode:
            ap.error(f"camera episode mismatch: {path}")
        if record.get("camera_frame", cam_left) != cam_left:
            ap.error(f"camera axes mismatch: {path}")
        if record.get("coordinate_frame", "original_camera") != "original_camera":
            ap.error(f"prepared-input fitter requires original camera axes: {path}")
        gt_use = record.get("gt_use", "unknown")
        if record.get("timestamp") is not None:
            stamp = float(record["timestamp"])
            if not np.isfinite(stamp) or (int(frame) in times and not np.isclose(times[int(frame)], stamp, rtol=0, atol=1e-9)):
                ap.error(f"camera timestamp mismatch: {path}")
            times[int(frame)] = stamp
        if args.require_independent and gt_use != "independent":
            ap.error(f"camera is {gt_use}, not independently produced: {path}")
        cameras[int(frame)] = js_to_mat(record)
        camera_sources[int(frame)] = {"path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(), "gt_use": gt_use,
            "camera_frame": cam_left}
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

        print(f"[fit] {obj}: mesh {len(mesh_o)} pts, frames {Tmax_o}")

        t_start = time.time()
        n_vis = int(ref.visible[:, b].sum()) if ref is not None else Tmax_o
        Xs: list[np.ndarray | None] = [None] * Tmax_o
        quals: list[dict | None] = [None] * Tmax_o
        prev: np.ndarray | None = None
        prev_frame, prev_time = None, None
        for t, frame in enumerate(frame_ids):
            frame = int(frame)
            if ref is not None and not args.no_gt and not bool(ref.visible[t, b]):
                continue
            if frame not in cameras:
                continue
            C_l = cameras[frame]
            if prev_frame is not None and frame - prev_frame > args.max_gap_frames:
                prev, prev_frame, prev_time = None, None, None
            ql = frame_cloud(
                work / "masks" / cam_left / obj / "0" / f"{frame:06d}.png",
                work / "depth" / cam_left / f"{frame:06d}.png",
                work / "intrinsics" / f"{cam_left}.json", args.max_pts, frame,
            )
            qr = (np.empty((0, 3), dtype=np.float32) if not args.joint_right else frame_cloud(
                work / "masks" / cam_right / obj / "0" / f"{frame:06d}.png",
                work / "depth" / cam_right / f"{frame:06d}.png",
                work / "intrinsics" / f"{cam_right}.json", args.max_pts, frame,
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
            X = _inv4(C_l) @ Pf
            timestamp = times.get(frame)
            jump_ok = temporal_check(prev, X, frame - prev_frame if prev_frame is not None else 1,
                                    timestamp - prev_time if timestamp is not None and prev_time is not None else None,
                                    fps, args.jump_deg, args.jump_m)
            met["jump_ok"] = jump_ok
            met["pass"] = inlier_pass(met, args.pass_inlier, args.joint_right) and jump_ok
            quals[t] = met
            if met["pass"]:
                Xs[t] = X
                prev, prev_frame, prev_time = X, frame, timestamp

        raw_Xs = [None if x is None else x.copy() for x in Xs]
        if args.smooth:
            Xs = smooth_poses(Xs, frame_ids=frame_ids)

        per_obj: dict = {"frames": Tmax_o, "solved": 0, "passed": 0, "framedetails": {},
                         "gt_score": {}}
        gt_ang, gt_disp, gt_chm = [], [], []
        prev, prev_frame, prev_time = None, None, None
        for t, frame in enumerate(frame_ids):
            frame = int(frame)
            if Xs[t] is None:
                if quals[t] is not None:
                    per_obj["framedetails"][str(frame)] = {"pass": False, "status": "rejected",
                        "jump_ok": quals[t]["jump_ok"], "camera_source": camera_sources[frame]}
                continue
            C_l = cameras[frame]
            X = Xs[t]
            per_obj["solved"] += 1
            Pf = C_l @ X
            ql = frame_cloud(work / "masks" / cam_left / obj / "0" / f"{frame:06d}.png",
                             work / "depth" / cam_left / f"{frame:06d}.png",
                             work / "intrinsics" / f"{cam_left}.json", args.max_pts, frame)
            qr = (frame_cloud(work / "masks" / cam_right / obj / "0" / f"{frame:06d}.png",
                              work / "depth" / cam_right / f"{frame:06d}.png",
                              work / "intrinsics" / f"{cam_right}.json", args.max_pts, frame,
                              scale=args.right_depth_scale) if args.joint_right else np.empty((0, 3)))
            met = pose_metrics(Pf, mesh_o, mesh_on, M, ql, qr, args.tukey_m)
            timestamp = times.get(frame)
            delta = frame - prev_frame if prev_frame is not None else None
            dt = timestamp - prev_time if timestamp is not None and prev_time is not None else None
            if delta is not None and delta > args.max_gap_frames:
                prev = None
            jump_ok = temporal_check(prev, X, delta or 1, dt, fps, args.jump_deg, args.jump_m)
            passed = inlier_pass(met, args.pass_inlier, args.joint_right) and jump_ok

            meta: dict = {
                "inlier_left": met["cam_left"]["inlier"],
                "inlier_right": met["cam_right"]["inlier"],
                "count_left": met["cam_left"]["count"],
                "count_right": met["cam_right"]["count"],
                "jump_ok": jump_ok,
                "init": quals[t]["init"],
                "pass": passed,
                "status": "observed" if passed else "rejected",
                "frame_index": frame, "timestamp": timestamp,
                "frame_delta": delta, "time_delta": dt,
                "camera_source": camera_sources[frame],
                "rmse_left": met["cam_left"]["rmse"],
                "rmse_right": met["cam_right"]["rmse"],
            }
            provenance = {"episode": args.episode, "frame_index": frame, "timestamp": timestamp,
                          "status": "observed", "camera_source": camera_sources[frame],
                          "gt_use": "gt_assisted" if args.score_gt else camera_sources[frame]["gt_use"],
                          "inference_gt": "visibility" if args.score_gt else "none"}
            raw_dir = ep_dir / obj / "raw_object_to_world"
            raw_dir.mkdir(exist_ok=True)
            (raw_dir / f"{frame:06d}.json").write_text(json.dumps(
                {**mat_to_js(raw_Xs[t]), **provenance, "coordinate_frame": "fixed_world"}, indent=2))
            if not passed:
                per_obj["framedetails"][str(frame)] = meta
                continue
            prev, prev_frame, prev_time = X, frame, timestamp
            (ep_dir / obj / "object_to_world" / f"{frame:06d}.json").write_text(json.dumps(
                {**mat_to_js(X), **provenance, "coordinate_frame": "fixed_world"}, indent=2))
            for cam, Pc in ((cam_left, Pf), (cam_right, M @ Pf)):
                (ep_dir / obj / "object_to_cam" / cam / f"{frame:06d}.json").write_text(json.dumps(
                    {**mat_to_js(Pc), **provenance, "camera_frame": cam,
                     "coordinate_frame": "original_camera"}, indent=2))

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
                meta["trans_err_cm"] = round(100 * disp, 3)
                meta["chamfer_mm"] = round(chm, 3)
                meta["gt_pass"] = bool(ang <= 10.0 and disp <= 0.05)
                gt_ang.append(ang)
                gt_disp.append(disp)
                gt_chm.append(chm)
            per_obj["framedetails"][str(frame)] = meta
            if passed:
                per_obj["passed"] += 1
        if args.score_gt:
            per_obj["gt_score"] = _summarize_gt(gt_ang, gt_disp, gt_chm, n_vis)
            per_obj["gt_score"]["median_trans_cm"] = float(100 * np.median(gt_disp)) if gt_disp else None
            per_obj["gt_score"]["p90_trans_cm"] = float(100 * np.percentile(gt_disp, 90)) if gt_disp else None
        per_obj["attempted"] = sum(q is not None for q in quals)
        per_obj["raw_accepted"] = sum(x is not None for x in raw_Xs)
        per_obj["exported"] = per_obj["passed"]
        per_obj["elapsed_s"] = round(time.time() - t_start, 1)
        per_obj_out[obj] = per_obj
        print(f"  {obj}: solved {per_obj['solved']} passed {per_obj['passed']} "
              f"({per_obj['elapsed_s']:.1f}s) {per_obj['gt_score']}")

    report = {
        "episode": args.episode,
        "cam_left": cam_left,
        "cam_right": cam_right,
        "pose_source": str(cam_root.resolve()),
        "camera_sources": {str(k): v for k, v in camera_sources.items()},
        "coordinate_frame": "fixed_world", "quaternion_order": "wxyz", "units": "metres",
        "gt_score_kind": "unaligned public diagnostic; official alignment is in the separate scorer",
        "inference_reads_gt": bool(args.score_gt),
        "gt_use": ["gt_assisted"] if args.score_gt else sorted({v["gt_use"] for v in camera_sources.values()}),
        "camera_gt_use": sorted({v["gt_use"] for v in camera_sources.values()}),
        "params": {"iters": args.iters, "mesh_pts": args.mesh_pts,
                   "tukey_m": args.tukey_m, "start_sigma_m": args.start_sigma,
                   "pass_inlier": args.pass_inlier, "joint_right": args.joint_right,
                   "right_depth_scale": args.right_depth_scale,
                   "jump_deg": args.jump_deg, "jump_m": args.jump_m,
                   "smoothing": args.smooth},
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
