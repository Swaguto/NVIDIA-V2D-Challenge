#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate FP-tracked object->camera poses on fixed-rig consistency.

The ego rig is rigid (two cameras, published baseline ~0.075 m), so even though
the rig moves through the scene two camera-motion-invariant checks are available:

  Gate A (within-camera rigidity): while two objects are rigidly held, their
    relative object->object transform must stay ~constant across frames.
  Gate B (cross-camera baseline): the same object tracked in camera B and
    camera C must place B relative to C at the published baseline magnitude.

Frames passing both gates yield trustworthy per-(cam,object) 6D poses that can
be compounded with the parquet GT (world->object) into per-frame world->cam
ego-camera poses.

Usage:
  python -m v2d.world_calib.vm.validate_fp_poses \
      --poses-dir v2d_ep2_outputs/poses \
      --out results/fp_gates_ep2 \
      --rig-meta <data_root>/meta/camera_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _q_to_rmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_pose(path: Path) -> np.ndarray:
    d = json.loads(path.read_text())
    rot, trans = d["rotation"], d["translation"]
    T = np.eye(4)
    T[:3, :3] = _q_to_rmat(np.asarray(rot, dtype=np.float64))
    T[:3, 3] = np.asarray(trans, dtype=np.float64)
    return T


def rel_rot_angle(a: np.ndarray, b: np.ndarray) -> float:
    dR = a[:3, :3] @ b[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))


def load_poses(poses_root: Path, camera: str, obj: str) -> dict[int, np.ndarray]:
    d = poses_root / camera / obj
    out: dict[int, np.ndarray] = {}
    for p in sorted(d.glob("*.json")):
        out[int(p.stem)] = load_pose(p)
    return out


def ego_baseline(rig_json: dict) -> float | None:
    for sp in rig_json.get("stereo_pairs", []):
        if sp.get("camera") == "ego":
            return float(sp["baseline_m"])
    return None


@dataclass
class Result:
    cam: str
    o1: str
    o2: str
    rel_dt_px: float
    rel_rot_deg: float
    rel_dt_median: float
    frames_pass_gate_a: int
    frames_total: int


def gate_a(
    cam: str, poses: dict[int, dict[str, np.ndarray]], objs: list[str], max_trans_m: float, max_rot_deg: float
) -> tuple[dict[int, bool], float, float]:
    rel = {t: np.linalg.inv(poses[t][objs[1]]) @ poses[t][objs[0]] for t in poses if all(o in poses[t] for o in objs)}
    if not rel:
        return {}, float("nan"), float("nan")
    ts = sorted(rel)
    med = np.median(np.stack([rel[t][:3, 3] for t in ts]), axis=0)
    med_rot = np.median(np.stack([rel[t][:3, :3] for t in ts]), axis=0)
    ok: dict[int, bool] = {}
    dts, angs = [], []
    for t in ts:
        dt = float(np.linalg.norm(rel[t][:3, 3] - med))
        ang = rel_rot_angle(rel[t], np.eye(4))
        ang = float(np.degrees(np.arccos(np.clip((np.trace(rel[t][:3, :3] @ med_rot.T) - 1) / 2, -1, 1))))
        dts.append(dt)
        angs.append(ang)
        ok[t] = dt < max_trans_m and ang < max_rot_deg
    return ok, float(np.median(dts)), float(np.median(angs))


def gate_b(
    cam_b: str, cam_c: str, obj: str,
    poses_b: dict[int, np.ndarray], poses_c: dict[int, np.ndarray],
    baseline_m: float, tol_m: float, avg_rot_deg: float,
) -> tuple[dict[int, bool], list[float], list[float]]:
    ts = sorted(set(poses_b) & set(poses_c))
    passmap: dict[int, bool] = {}
    norms, angs = [], []
    for t in ts:
        T_b_from_c = poses_b[t] @ np.linalg.inv(poses_c[t])
        n = float(np.linalg.norm(T_b_from_c[:3, 3]))
        ang = rel_rot_angle(T_b_from_c, np.eye(4))
        norms.append(n)
        angs.append(ang)
        passmap[t] = abs(n - baseline_m) < tol_m and ang < avg_rot_deg
    return passmap, norms, angs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poses-dir", required=True)
    ap.add_argument("--rig-meta", required=True)
    ap.add_argument("--camera-b", default="ego_cam_b")
    ap.add_argument("--camera-c", default="ego_cam_c")
    ap.add_argument("--objects", nargs="+", default=["white_pot", "white_pot_lid"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-a-max-trans-mm", type=float, default=25.0)
    ap.add_argument("--gate-a-max-rot-deg", type=float, default=0.5)
    ap.add_argument("--gate-b-tol-mm", type=float, default=25.0)
    ap.add_argument("--gate-b-max-rot-deg", type=float, default=3.0)
    args = ap.parse_args()

    poses_dir = Path(args.poses_dir)
    rig = json.loads(Path(args.rig_meta).read_text())
    baseline = ego_baseline(rig)
    if baseline is None:
        sys.exit("no ego stereo_pair baseline found in rig meta")

    objs = args.objects
    out: list[dict] = []
    gate_a_pass = {t: True for t in range(100000)}
    for cam in (args.camera_b, args.camera_c):
        poses = {
            t: {o: load_pose(poses_dir / cam / o / f"{t:06d}.json") for o in objs}
            for t in range(100000)
            if (poses_dir / cam / objs[0] / f"{t:06d}.json").exists()
        }
        ok_a, med_dt, med_ang = gate_a(
            cam, poses, objs, args.gate_a_max_trans_mm / 1000.0, args.gate_a_max_rot_deg
        )
        # intersect across cameras: gate A must pass in BOTH cams for the frame to count
        gate_a_pass = {t: (gate_a_pass.get(t, False) and ok_a.get(t, False)) for t in set(gate_a_pass) & set(ok_a)}
        out.append(
            {
                "camera": cam, "gate": "A",
                "median_rel_dt_m": round(med_dt, 4), "median_rel_rot_deg": round(med_ang, 3),
                "frames_pass": sum(ok_a.values()), "frames_total": len(ok_a),
            }
        )

    poses_b = load_poses(poses_dir, args.camera_b, objs[0])
    poses_c = load_poses(poses_dir, args.camera_c, objs[0])
    pass_b, norms, angs = gate_b(
        args.camera_b, args.camera_c, objs[0], poses_b, poses_c,
        baseline, args.gate_b_tol_mm / 1000.0, args.gate_b_max_rot_deg,
    )
    out.append(
        {
            "camera": f"{args.camera_b}_{args.camera_c}", "gate": "B", "object": objs[0],
            "baseline_m": baseline, "median_norm_m": round(float(np.median(norms)), 4),
            "median_rot_deg": round(float(np.median(angs)), 2),
            "frames_pass": sum(pass_b.values()), "frames_total": len(pass_b),
        }
    )

    both = sorted(set(gate_a_pass) & set(t for t, p in pass_b.items() if p))
    out.append({"gate": "BOTH", "frames_pass": len(both), "objects_checked": objs})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    (Path(args.out).parent / f"{Path(args.out).name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if os_ := Path(args.out).parent / f"{Path(args.out).name}_pass_frames.json":
        (os_).write_text(json.dumps(both))
        print(f"pass frames -> {os_}")


if __name__ == "__main__":
    main()