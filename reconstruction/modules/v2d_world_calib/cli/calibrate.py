#!/usr/bin/env python3
"""V2D Track 3 - mocap<->camera world calibration CLI.

Solves the per-camera rigid ``world -> camera`` transform from GT object
geometry paired with image keypoints, then verifies transfer on held-out
frames:

    Stage 0  camera_calibration.json -> intrinsics / distortion / cam-cam
    Stage 1  keypoint correspondences (GT pose applied to mesh points)
    Stage 3  RANSAC-PnP init + Huber reprojection refinement (per camera)
    Verify   held-out transfer stats, spatial error map, cam-cam consistency

Run
---
    # synthetic correctness proof (no challenge data needed):
    python -m v2d.world_calib.cli.calibrate --self-test

    # real calibration (on the shared box with data + keypoint cache):
    python -m v2d.world_calib.cli.calibrate \
        --data-root ~/v2d_track3/data/hf/track_3/public \
        --camera ego_cam_a --camera ego_cam_b \
        --fit-episodes 3,7,15 \
        --test-episodes 27,43 \
        --keypoints-dir results/keypoints \
        --out-dir results/calib
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from v2d.world_calib.calibration.camera_geometry import CameraIntrinsics, RigCalibration
from v2d.world_calib.calibration.correspondences import (
    load_keypoint_correspondences,
    quaternion_to_matrix,
    synthetic_frontal_world_cam_pose,
    synthetic_keypoint_correspondences,
)
from v2d.world_calib.calibration.solve import CameraCalibrationSolve, solve_camera
from v2d.world_calib.calibration.verify import (
    correspondence_reprojection_errors,
    error_stats_pixels,
    pixel_error_grid_map,
    proxy_episode_guard,
    rig_consistency_rows,
)

_DEFAULT_DATA_ROOT = os.path.expanduser("~/v2d_track3/data/hf/track_3/public")


# --------------------------------------------------------------------------- #
# Report writer
# --------------------------------------------------------------------------- #


def _fmt_stats(stats: dict[str, float]) -> str:
    return (
        f"mean {stats['mean']:.2f} px | median {stats['median']:.2f} | "
        f"RMSE {stats['rmse']:.2f} | P95 {stats['p95']:.2f} | max {stats['max']:.2f}"
    )


def write_report(
    out_dir: str | os.PathLike,
    *,
    rig: RigCalibration | None,
    solves: dict[str, CameraCalibrationSolve],
    transfer_stats: dict[str, dict[str, float]],
    test_episodes: list[int],
    spatial_maps: dict[str, np.ndarray],
    consistency_rows,
    errors_mm_trans: dict[str, float] | None = None,
    trans_err_deg: dict[str, float] | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = ["# v2d_world_calib - camera<->mocap-world calibration (REPORT)", ""]
    if rig is not None:
        declared, measured = rig.ego_baseline_mm()
        lines += [
            "## Stage 0 - ego stereo baseline",
            f"- declared: {declared:.1f} mm",
            f"- measured: {measured:.1f} mm  (from extrinsics)",
            "",
        ]
    lines += ["## Stage 3 - per-camera world->camera solve", ""]
    lines += [
        "| camera | samples | inliers | fit_rms(px) | refined_rms(px) | iters |",
        "|---|---|---|---|---|---|",
    ]
    for cam, s in solves.items():
        lines.append(
            f"| {cam} | {s.samples} | {s.inliers} | {s.fit_rms_px:.2f} | "
            f"{s.refine_rms_px:.2f} | {s.iterations} |"
        )
    lines.append("")

    if errors_mm_trans:
        lines += ["## recovery vs ground truth (self-test)", ""]
        for cam, err in errors_mm_trans.items():
            lines.append(f"- {cam}: trans err {err:.1f} mm, rot err {trans_err_deg[cam]:.3f} deg")
        lines.append("")

    if transfer_stats:
        lines += ["## held-out transfer (pixel reprojection)", ""]
        lines += ["| camera | mean | median | RMSE | P95 | max |", "|---|---|---|---|---|---|"]
        for cam, st in sorted(transfer_stats.items()):
            lines.append(f"| {cam} | {st['mean']:.2f} | {st['median']:.2f} | {st['rmse']:.2f} | {st['p95']:.2f} | {st['max']:.2f} |")
        lines += ["", f"- test episodes: {''.join(map(str, test_episodes))}", ""]

    if spatial_maps:
        lines += ["## spatial error map (mean px per image region, rows x cols)", ""]
        for cam, grid in spatial_maps.items():
            lines += [f"### {cam}", "```", np.array2string(grid, precision=2, suppress_small=True), "```", ""]

    if consistency_rows:
        lines += ["## cam-to-cam consistency (estimated vs provided)", ""]
        lines += ["| source | target | rot delta (deg) | trans delta (m) | status |", "|---|---|---|---|---|"]
        for row in consistency_rows:
            lines.append(
                f"| {row.source} | {row.target} | {row.rot_delta_deg:.4f} | {row.trans_delta_m:.4f} | {row.status} |"
            )
        lines.append("")

    lines.append("## export")
    lines += ["- calibration.json (T_world->cam for every camera)"]
    lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines))

    calib_json = {
        "camera_transforms": {},
        "per_pose_quaternion_xyzw": {},
    }
    for cam, s in solves.items():
        calib_json["camera_transforms"][cam] = {
            "R": s.rot.tolist(),
            "t": s.trans.tolist(),
            "quat_xyzw": s.quat_xyzw.tolist(),
            "fit_rms_px": s.fit_rms_px,
            "refine_rms_px": s.refine_rms_px,
            "samples": s.samples,
            "inliers": s.inliers,
        }
        calib_json["per_pose_quaternion_xyzw"][cam] = s.quat_xyzw.tolist()
    (out / "calibration.json").write_text(json.dumps(calib_json, indent=2))


# --------------------------------------------------------------------------- #
# Real-data path
# --------------------------------------------------------------------------- #


def run_real(args) -> int:
    base = Path(args.data_root)
    cal_path = base / "meta" / "camera_calibration.json"
    if not cal_path.exists():
        print(
            f"camera_calibration.json not found at {cal_path} (pass --data-root).",
            file=sys.stderr,
        )
        return 2
    if not args.keypoints_dir:
        print("--keypoints-dir required for the real path (2D object keypoint cache).", file=sys.stderr)
        return 2

    rig = RigCalibration.load(cal_path)
    cameras = args.camera or rig.camera_names()
    fit_eps = [int(e) for e in args.fit_episodes.split(",") if e.strip()]
    test_eps = [int(e) for e in args.test_episodes.split(",") if e.strip()]
    if not fit_eps:
        print("need >= 1 fit episode (--fit-episodes).", file=sys.stderr)
        return 2
    for warn in proxy_episode_guard(fit_eps, test_eps):
        print(f"WARN: {warn}", file=sys.stderr)

    solves: dict[str, CameraCalibrationSolve] = {}
    transfer: dict[str, dict[str, float]] = {}
    spatial: dict[str, np.ndarray] = {}
    for cam in cameras:
        corr = load_keypoint_correspondences(cam, fit_eps, args.data_root, args.keypoints_dir)
        if corr.count < args.min_inliers:
            print(f"{cam}: only {corr.count} correspondences, skipping.", file=sys.stderr)
            continue
        s = solve_camera(cam, corr, rig.intrinsics[cam], min_inliers=args.min_inliers)
        solves[cam] = s
        errs = correspondence_reprojection_errors(corr, s.rot, s.trans, rig.intrinsics[cam])
        transfer[f"{cam} (fit)"] = error_stats_pixels(errs)
        K = rig.intrinsics[cam]
        spatial[cam] = pixel_error_grid_map(
            corr, errs, width=K.width or int(corr.points_image[:, 0].max()), height=K.height or int(corr.points_image[:, 1].max())
        )

        if test_eps:
            test_corr = load_keypoint_correspondences(cam, test_eps, args.data_root, args.keypoints_dir)
            if test_corr.count >= args.min_inliers:
                test_errs = correspondence_reprojection_errors(
                    test_corr, s.rot, s.trans, rig.intrinsics[cam]
                )
                transfer[f"{cam} (held-out)"] = error_stats_pixels(test_errs)

    consistency = rig_consistency_rows(rig, solves) if solves else []
    write_report(
        args.out_dir,
        rig=rig,
        solves=solves,
        transfer_stats=transfer,
        test_episodes=test_eps,
        spatial_maps=spatial,
        consistency_rows=consistency,
    )
    print(f"\nSolved {len(solves)} camera(s); REPORT + calibration.json -> {args.out_dir}")
    for cam, s in solves.items():
        print(f"  {cam}: refined reproj RMS {s.refine_rms_px:.2f} px over {s.samples} samples")
    print(json.dumps(transfer, indent=2, default=float))
    return 0 if solves else 2


# --------------------------------------------------------------------------- #
# Synthetic self-test (proves the engine works without challenge data)
# --------------------------------------------------------------------------- #


def _build_synthetic_rig(provided: dict[str, list] | None = None) -> RigCalibration:
    provided = provided or {}
    common = dict(fx=1024.0, fy=1024.0, cx=1014.0, cy=760.0, resolution=[2028, 1520])
    cams = {}
    for k, title in enumerate(("ego_cam_a", "ego_cam_b", "ego_cam_c")):
        entry = dict(common)
        if title in provided:
            entry["extrinsics_to_stereo_left"] = provided[title]
        else:
            entry["extrinsics_to_stereo_left"] = np.eye(3, 4).tolist()
        cams[title] = entry

    baseline_m = 0.075
    if "ego_cam_c" in provided and "ego_cam_b" in provided:
        pb = np.asarray(provided["ego_cam_b"], dtype=np.float64).reshape(3, 4)
        pc = np.asarray(provided["ego_cam_c"], dtype=np.float64).reshape(3, 4)
        cb = -pb[:3, :3].T @ pb[:, 3]
        cc = -pc[:3, :3].T @ pc[:, 3]
        baseline_m = float(np.linalg.norm(cc - cb))

    rig = RigCalibration(
        intrinsics={
            name: CameraIntrinsics(fx=common["fx"], fy=common["fy"], cx=common["cx"], cy=common["cy"], width=2028, height=1520)
            for name in cams
        },
        cameras=cams,
        stereo_pairs=[
            {"camera": "ego", "left": "ego_cam_b", "right": "ego_cam_c", "baseline_m": baseline_m}
        ],
        raw={"cameras": cams, "stereo_pairs": []},
    )
    return rig


def _synthetic_object_points(bodies: int = 3, views: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    local = np.stack(
        [
            rng.uniform(-0.12, 0.12, size=(views, 2)).__array__()
            for _ in range(bodies)
        ],
        axis=0,
    )
    local = np.concatenate([local, np.zeros((bodies, views, 1))], axis=-1)
    local[..., 2] = rng.uniform(0.05, 0.25, size=(bodies, views))
    return local


def _synthetic_world_poses(bodies: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = rng.uniform([-0.25, -0.25, 0.45], [0.25, 0.25, 0.75], size=(bodies, 3))
    quat = np.stack([np.ones(bodies), np.zeros(bodies), np.zeros(bodies), np.zeros(bodies)], axis=-1)
    return np.concatenate([pos, quat], axis=-1)


def run_self_test(args) -> int:
    local = _synthetic_object_points(bodies=3, views=64, seed=0)
    poses = _synthetic_world_poses(bodies=3, seed=1)
    rng = np.random.default_rng(7)

    world_all = []
    for b in range(poses.shape[0]):
        q = poses[b, 3:]
        qn = q / (np.linalg.norm(q) + 1e-12)
        world_all.append(local[b] @ quaternion_to_matrix(qn) + poses[b, :3])
    world_all = np.concatenate(world_all, axis=0)

    rig = _build_synthetic_rig()
    solves: dict[str, CameraCalibrationSolve] = {}
    true_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    trans_err_mm: dict[str, float] = {}
    rot_err_deg: dict[str, float] = {}
    transfer: dict[str, dict[str, float]] = {}
    spatial: dict[str, np.ndarray] = {}

    for cam, K in rig.intrinsics.items():
        true_rot, true_trans = synthetic_frontal_world_cam_pose(K, world_all, rng)
        true_poses[cam] = (true_rot, true_trans)

        fit_corr = synthetic_keypoint_correspondences(
            cam, K, true_rot, true_trans, local, poses, frames=10, noise_px=1.0, seed=int(args.seed)
        )
        heldout = synthetic_keypoint_correspondences(
            cam, K, true_rot, true_trans, local, poses, frames=5, noise_px=1.2, seed=int(args.seed) + 1
        )

        s = solve_camera(cam, fit_corr, K, min_inliers=args.min_inliers)
        solves[cam] = s
        rel = s.rot.T @ true_rot
        rot_err_deg[cam] = float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))))
        trans_err_mm[cam] = float(np.linalg.norm(s.trans - true_trans)) * 1e3

        fit_errs = correspondence_reprojection_errors(fit_corr, s.rot, s.trans, K)
        transfer[f"{cam} (fit)"] = error_stats_pixels(fit_errs)
        hold_errs = correspondence_reprojection_errors(heldout, s.rot, s.trans, K)
        transfer[f"{cam} (held-out)"] = error_stats_pixels(hold_errs)
        spatial[cam] = pixel_error_grid_map(fit_corr, fit_errs, K.width, K.height)

    # Provided cam<->cam extrinsics derived from the (hidden) ground-truth
    # poses relative to the stereo-left camera ego_cam_b, so the consistency
    # check has a correct reference to compare the *estimate* against.
    def _inv(rot: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return rot.T, -rot.T @ trans

    def _to_provided(rot: np.ndarray, trans: np.ndarray) -> list:
        m = np.eye(4)
        m[:3, :3] = rot
        m[:3, 3] = trans
        return m[:3].tolist()

    provided: dict[str, list] = {}
    left = "ego_cam_b"
    for cam in rig.camera_names():
        if cam == left:
            provided[cam] = _to_provided(np.eye(3), np.zeros(3))
        else:
            rl, tl = true_poses[left]
            rc, tc = true_poses[cam]
            rb, tb = _inv(rl, tl)
            provided[cam] = _to_provided(rc @ rb, rc @ tb + tc)
    rig = _build_synthetic_rig(provided)

    consistency = rig_consistency_rows(rig, solves)
    write_report(
        args.out_dir,
        rig=rig,
        solves=solves,
        transfer_stats=transfer,
        test_episodes=[999],  # synthetic held-out tag
        spatial_maps=spatial,
        consistency_rows=consistency,
        errors_mm_trans=trans_err_mm,
        trans_err_deg=rot_err_deg,
    )

    print(f"\nSELF-TEST  (synthetic rig, known T_world->cam, ~1 px observation noise)")
    print(f"{'camera':<12} {'fit_rms':>8} {'refined':>9} {'transerr_mm':>11} {'roterr_deg':>10}")
    for cam, s in solves.items():
        print(
            f"{cam:<12} {s.fit_rms_px:>8.2f} {s.refine_rms_px:>9.2f} "
            f"{trans_err_mm[cam]:>10.2f} {rot_err_deg[cam]:>10.4f}"
        )
    print("\nheld-out transfer pixel errors:")
    for key, st in sorted(transfer.items()):
        print(f"  {key:<18} {_fmt_stats(st)}")
    print("\ncam-to-cam consistency:")
    for row in consistency:
        print(f"  {row.source} -> {row.target}: rot {row.rot_delta_deg:.4f} deg, t {row.trans_delta_m*1e3:.2f} mm [{row.status}]")
    print(f"\nwrote REPORT.md + calibration.json -> {args.out_dir}")

    ok = (
        s.refine_rms_px < 2.5
        and all(v < 30.0 for v in trans_err_mm.values())
        and all(v < 1.0 for v in rot_err_deg.values())
    )
    print(("\nSELF-TEST PASSED - solver recovers the world->camera transform." if ok else "\nSELF-TEST FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--camera", action="append", default=None)
    parser.add_argument("--fit-episodes", default="3,7,15")
    parser.add_argument("--test-episodes", default="")
    parser.add_argument("--keypoints-dir", default=None)
    parser.add_argument("--out-dir", default="results/calib")
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())