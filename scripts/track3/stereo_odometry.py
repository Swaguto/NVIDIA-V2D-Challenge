#!/usr/bin/env python3
"""Estimate a metric, object-GT-free camera trajectory from the ego stereo pair.

Rectified left-frame features with valid stereo depth are tracked to the next
left frame. PnP-RANSAC estimates the relative transform. Every frame records
inlier and reprojection diagnostics; failed steps hold the previous transform
and are explicitly marked inferred. The world gauge is camera frame zero.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("V2D_RUNTIME", str(ROOT / "reconstruction/modules/v2d_stereo_depth/lib")))
from stereo_depth import load_rig, stereo_depth  # noqa: E402


def frame_pair(left, right):
    ok_l, image_l = left.read()
    ok_r, image_r = right.read()
    if ok_l != ok_r:
        raise ValueError("stereo videos have different frame counts")
    return (image_l, image_r) if ok_l else None


def estimate_step(previous_gray, current_gray, previous_depth, previous_valid, K,
                  exclude=None, min_inliers=35):
    valid = previous_valid & (previous_depth > 0.15) & (previous_depth < 2.5)
    if exclude is not None:
        valid &= ~exclude
    support = (valid.astype(np.uint8) * 255)
    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=2500,
                                      qualityLevel=0.005, minDistance=7,
                                      mask=support, blockSize=7)
    if points is None or len(points) < min_inliers:
        return None, {"features": 0 if points is None else len(points), "inliers": 0}
    forward, ok, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points,
                                               None, winSize=(21, 21), maxLevel=3)
    backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray,
                                                    forward, None, winSize=(21, 21), maxLevel=3)
    keep = ok.ravel().astype(bool) & back_ok.ravel().astype(bool)
    keep &= np.linalg.norm(backward.reshape(-1, 2) - points.reshape(-1, 2), axis=1) < 1.5
    uv0 = points.reshape(-1, 2)[keep]
    uv1 = forward.reshape(-1, 2)[keep]
    if len(uv0) < min_inliers:
        return None, {"features": len(points), "tracked": len(uv0), "inliers": 0}
    x = np.clip(np.rint(uv0[:, 0]).astype(int), 0, previous_gray.shape[1] - 1)
    y = np.clip(np.rint(uv0[:, 1]).astype(int), 0, previous_gray.shape[0] - 1)
    z = previous_depth[y, x].astype(np.float64)
    xyz = np.stack([(uv0[:, 0] - K[0, 2]) * z / K[0, 0],
                    (uv0[:, 1] - K[1, 2]) * z / K[1, 1], z], axis=1)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.ascontiguousarray(xyz), np.ascontiguousarray(uv1.astype(np.float64)),
        K, None, iterationsCount=300, reprojectionError=2.5, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP)
    count = len(inliers) if ok and inliers is not None else 0
    info = {"features": len(points), "tracked": len(uv0), "inliers": count,
            "inlier_fraction": count / max(len(uv0), 1)}
    if count < min_inliers or info["inlier_fraction"] < 0.25:
        return None, info
    indices = inliers.ravel()
    ok, rvec, tvec = cv2.solvePnP(np.ascontiguousarray(xyz[indices]),
                                  np.ascontiguousarray(uv1[indices].astype(np.float64)),
                                  K, None, rvec, tvec, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, info
    projected, _ = cv2.projectPoints(xyz[indices], rvec, tvec, K, None)
    error = np.linalg.norm(projected.reshape(-1, 2) - uv1[indices], axis=1)
    info["median_reprojection_px"] = float(np.median(error))
    rotation, _ = cv2.Rodrigues(rvec)
    step = np.eye(4)
    step[:3, :3], step[:3, 3] = rotation, tvec.ravel()
    info["step_translation_m"] = float(np.linalg.norm(tvec))
    info["step_rotation_deg"] = float(np.rad2deg(np.linalg.norm(rvec)))
    if (info["median_reprojection_px"] > 1.8 or
            info["step_translation_m"] > 0.15 or info["step_rotation_deg"] > 20):
        return None, info
    return step, info


def run(episode: int, data: Path, out: Path, mask_dir: Path | None = None):
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output must be a fresh directory: {out}")
    rig = load_rig()
    base = data / "videos/chunk-000"
    left = cv2.VideoCapture(str(base / f"observation.images.{rig.left}" /
                                f"episode_{episode:06d}.mp4"))
    right = cv2.VideoCapture(str(base / f"observation.images.{rig.right}" /
                                 f"episode_{episode:06d}.mp4"))
    if not left.isOpened() or not right.isOpened():
        raise FileNotFoundError(f"stereo episode {episode}")
    K = np.asarray(rig.P1[:3, :3], dtype=np.float64)
    out_cam = out / rig.left
    out_cam.mkdir(parents=True)
    camera = np.eye(4)
    previous = None
    diagnostics = []
    frame = 0
    while (pair := frame_pair(left, right)) is not None:
        raw_l, raw_r = pair
        rect_l, _ = rig.rectify_pair(raw_l, raw_r)
        gray = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        depth, valid = stereo_depth(rig, raw_l, raw_r)
        if previous is None:
            status, info = "observed", {"inliers": 0, "reference_frame": True}
        else:
            mask = None
            if mask_dir is not None:
                mask = np.zeros(gray.shape, np.uint8)
                for path in mask_dir.glob(f"*/{frame - 1:06d}.png"):
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        mask |= image
            step, info = estimate_step(previous[0], gray, previous[1], previous[2], K,
                                       exclude=(mask > 0) if mask is not None else None)
            status = "observed" if step is not None else "inferred"
            if step is not None:
                camera = step @ camera
        quat = Rotation.from_matrix(camera[:3, :3]).as_quat()
        record = {"episode": episode, "frame_index": frame,
                  "translation": camera[:3, 3].tolist(),
                  "rotation": quat[[3, 0, 1, 2]].tolist(),
                  "scale": 1.0, "status": status, "gt_use": "independent",
                  "coordinate_frame": "rectified_camera", "camera_frame": rig.left,
                  "method": "stereo_pnp_ransac", "diagnostics": info}
        (out_cam / f"{frame:06d}.json").write_text(json.dumps(record))
        diagnostics.append({"frame_index": frame, "status": status, **info})
        if frame % 50 == 0:
            print(f"  VO frame {frame}: {status}, inliers={info['inliers']}", flush=True)
        previous = (gray, depth, valid)
        frame += 1
    left.release()
    right.release()
    report = {"episode": episode, "frames": frame,
              "observed_frames": sum(x["status"] == "observed" for x in diagnostics),
              "failed_frames": [x["frame_index"] for x in diagnostics
                                if x["status"] != "observed"],
              "gt_use": "independent", "coordinate_frame": "rectified_camera",
              "diagnostics": diagnostics}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get(
        "V2D_PUBLIC", "/home/ubuntu/v2d_track3/data/hf/track_3/public")))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path)
    args = parser.parse_args()
    report = run(args.episode, args.data, args.out, args.mask_dir)
    print(f"stereo VO: {report['observed_frames']}/{report['frames']} accepted frames", flush=True)


if __name__ == "__main__":
    main()
