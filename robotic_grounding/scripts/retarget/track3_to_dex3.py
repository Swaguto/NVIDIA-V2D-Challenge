# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retarget a validated Track 3 episode using the existing CHORD Dex3 QP.

Unknown observations remain NaN and reset the corresponding warm start. The
output is motion evidence, not a learned-policy rollout or a penetration pass.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source/robotic_grounding"))
from robotic_grounding.retarget.retarget_utils import (  # noqa: E402
    setup_dex3_kinematics,
)
from robotic_grounding.retarget.track3_bundle import (  # noqa: E402
    SIDES,
    digest,
    validate_bundle,
    write_json,
)  # noqa: E402


def distribution(values: np.ndarray) -> dict:
    """Summarize finite samples while retaining an empty result as unknown."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def retarget(
    root: Path, out: Path, scale: float, max_iter: int, end: int | None
) -> dict:
    """Fit supported wrists and fingertips with bounded CHORD differential IK."""
    validation = validate_bundle(root)
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("MANO-to-robot scale must be positive")
    out.mkdir(parents=True)
    start = time.monotonic()
    data = np.load(root / "episode.npz", allow_pickle=False)
    frames, times = data["frame_indices"], data["timestamps_s"]
    n = len(frames)
    models = {
        side: setup_dex3_kinematics(side=side, max_iter=max_iter) for side in SIDES
    }
    original_limits = {
        side: (
            kin.robot.model.lowerPositionLimit.copy(),
            kin.robot.model.upperPositionLimit.copy(),
        )
        for side, kin in models.items()
    }
    dimensions = {kin.robot.model.nq for kin in models.values()}
    if len(dimensions) != 1:
        raise ValueError("Unexpected per-side configuration dimensions")
    nq = dimensions.pop()
    qpos = np.full((n, 2, nq), np.nan)
    valid = np.zeros((n, 2), dtype=bool)
    errors = np.full((n, 2, 4), np.nan)
    orientation_error_deg = np.full((n, 2), np.nan)
    iterations = np.zeros((n, 2), dtype=int)
    previous = {side: None for side in SIDES}
    frame_poses = {
        side: np.full((n, len(kin.robot_frame_names), 7), np.nan)
        for side, kin in models.items()
    }
    progress = out / "progress.jsonl"
    processed = 0
    for row, frame in enumerate(frames):
        if end is not None and int(frame) >= end:
            break
        for side_id, side in enumerate(SIDES):
            kin = models[side]
            lower, upper = original_limits[side]
            kin.robot.model.lowerPositionLimit = lower.copy()
            kin.robot.model.upperPositionLimit = upper.copy()
            if not data["hand_valid"][row, side_id]:
                previous[side] = None
                continue
            joints = data["hand_joints_world"][row, side_id]
            wrist = data["hand_wrist_world"][row, side_id]
            correction = kin.get_frame_rotation_correction(f"{side}_hand_palm_link")
            # Refinement mirrors the right MANO geometry in camera X for left
            # hands. Its exported mirrored canonical frame has fingers along
            # +X and the thumb on -Y, matching the neutral left Dex3 frame.
            # The upstream left-MANO correction instead assumes another palm
            # convention. Cancel it at this adapter boundary, leaving the
            # original CHORD solver and dataset retargeters unchanged.
            target_rotation = wrist[:3, :3] @ (
                np.eye(3) if side == "left" else correction
            )
            source_palm_rotation = target_rotation @ correction.T
            # Tip orientation costs are zero in DEX3_TO_MANO_MAPPING. Repeating
            # the known wrist rotation supplies only the palm orientation task.
            quaternions = np.repeat(
                Rotation.from_matrix(source_palm_rotation).as_quat(scalar_first=True)[
                    None
                ],
                21,
                axis=0,
            )
            initial = previous[side]
            if initial is None:
                initial = kin.robot.q0.copy()
                initial[:3] = wrist[:3, 3]
                initial[3:7] = Rotation.from_matrix(target_rotation).as_quat()
                initial[7:] = np.clip(
                    initial[7:],
                    kin.robot.model.lowerPositionLimit[7:],
                    kin.robot.model.upperPositionLimit[7:],
                )
            else:
                # Solver dt limits each optimization increment, not the final
                # change between 20 Hz source frames. Bound final finger qpos
                # around the preceding supported sample inside the same QP.
                budget = kin.robot.model.velocityLimit[6:] * (
                    times[row] - times[row - 1]
                )
                dynamic_lower, dynamic_upper = lower.copy(), upper.copy()
                dynamic_lower[7:] = np.maximum(lower[7:], initial[7:] - budget)
                dynamic_upper[7:] = np.minimum(upper[7:], initial[7:] + budget)
                kin.robot.model.lowerPositionLimit = dynamic_lower
                kin.robot.model.upperPositionLimit = dynamic_upper
            result = kin.compute(
                joints, quaternions, source_to_robot_scale=scale, qpos=initial
            )
            q = result["q"]
            if not np.isfinite(q).all() or abs(np.linalg.norm(q[3:7]) - 1) > 1e-6:
                raise ValueError(f"Nonfinite or invalid IK result at {side}/{frame}")
            if np.any(q[7:] < kin.robot.model.lowerPositionLimit[7:] - 1e-6) or np.any(
                q[7:] > kin.robot.model.upperPositionLimit[7:] + 1e-6
            ):
                raise ValueError(f"Joint limit violation at {side}/{frame}")
            qpos[row, side_id] = q
            valid[row, side_id] = True
            previous[side] = q.copy()
            errors[row, side_id] = result["frame_task_errors"]
            frame_poses[side][row] = result["frame_pose"]
            iterations[row, side_id] = result["num_optimization_iterations"]
            orientation_error_deg[row, side_id] = np.rad2deg(
                Rotation.from_matrix(
                    target_rotation.T @ Rotation.from_quat(q[3:7]).as_matrix()
                ).magnitude()
            )
        processed = row + 1
        if processed % 25 == 0 or processed == n:
            record = {
                "frames_processed": processed,
                "supported_hand_fits": int(valid.sum()),
                "elapsed_seconds": time.monotonic() - start,
            }
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    arrays = {
        "frame_indices": frames,
        "timestamps_s": times,
        "source_hand_valid": data["hand_valid"],
        "robot_valid": valid,
        "robot_qpos": qpos,
        "task_position_errors_m": errors,
        "wrist_orientation_error_deg": orientation_error_deg,
        "iterations": iterations,
    }
    reports = {}
    joint_contract = {}
    for side_id, side in enumerate(SIDES):
        kin = models[side]
        kin.robot.model.lowerPositionLimit, kin.robot.model.upperPositionLimit = (
            original_limits[side]
        )
        names = list(kin.robot_finger_joint_names.values())
        joint_contract[side] = {
            "finger_joint_names": names,
            "qpos_order": "base_xyz, base_quaternion_xyzw, finger joints in declared order",
            "lower_radians": kin.robot.model.lowerPositionLimit[7:].tolist(),
            "upper_radians": kin.robot.model.upperPositionLimit[7:].tolist(),
            "velocity_limits_rad_s": kin.robot.model.velocityLimit[6:].tolist(),
            "frame_names": list(kin.robot_frame_names.values()),
        }
        arrays[f"robot_{side}_frame_poses"] = frame_poses[side]
        pairs = valid[1:, side_id] & valid[:-1, side_id]
        dt = np.diff(times)[pairs]
        velocity = np.diff(qpos[:, side_id, 7:], axis=0)[pairs] / dt[:, None]
        linear_speed = (
            np.linalg.norm(np.diff(qpos[:, side_id, :3], axis=0)[pairs], axis=1) / dt
        )
        velocity_violations = (
            np.abs(velocity) > kin.robot.model.velocityLimit[6:] + 1e-6
        )
        reports[side] = {
            "source_supported_frames": int(data["hand_valid"][:, side_id].sum()),
            "retargeted_frames": int(valid[:, side_id].sum()),
            "wrist_position_error_mm": distribution(errors[:, side_id, 0] * 1000),
            "three_tip_position_error_mm": distribution(errors[:, side_id, 1:] * 1000),
            "wrist_orientation_error_deg": distribution(
                orientation_error_deg[:, side_id]
            ),
            "adjacent_supported_pairs": int(pairs.sum()),
            "wrist_speed_m_s": distribution(linear_speed),
            "finger_speed_rad_s": distribution(np.abs(velocity)),
            "finger_velocity_limit_violating_pairs": int(
                np.any(velocity_violations, axis=1).sum()
            ),
        }
    np.savez_compressed(out / "retarget.npz", **arrays)
    manifest = {
        "schema": "v2d.track3.dex3.v1",
        "episode_bundle_manifest_sha256": validation["manifest_sha256"],
        "episode_bundle_payload_sha256": validation["payload_sha256"],
        "robot": "dual_floating_dex3",
        "hand_sides": list(SIDES),
        "joints": joint_contract,
        "solver": "CHORD HandKinematics / Pink differential IK / DAQP",
        "mano_to_robot_scale": scale,
        "max_iterations": max_iter,
        "solver_frequency_hz": 100,
        "unknown_policy": "NaN qpos; warm start reset across source observation gaps",
        "tip_orientation": "not measured or optimized; position-only fingertip tasks",
        "palm_frame_calibration": "mirrored refinement left frame uses identity; right uses upstream Z-180 correction",
        "source_frame_velocity_constraints": "final finger qpos bounded by preceding supported qpos +/- URDF velocity * source dt",
        "source_sha256": {
            str(Path(__file__).relative_to(REPO)): digest(Path(__file__)),
            "hand_kinematics.py": digest(
                REPO
                / "source/robotic_grounding/robotic_grounding/retarget/hand_kinematics.py"
            ),
        },
        "runtime": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "pin",
                "pin-pink",
                "qpsolvers",
                "daqp",
                "torch",
            )
        },
        "payload": {"path": "retarget.npz", "sha256": digest(out / "retarget.npz")},
        "processed_frames": processed,
        "source_frames": n,
        "penetration_gate": "pending",
        "simulator_replay": "pending",
        "learned_policy_rollout": False,
    }
    write_json(out / "manifest.json", manifest)
    report = {
        "status": (
            "IK complete; penetration and simulation gates pending"
            if processed == n
            else "partial IK diagnostic"
        ),
        "frames_processed": processed,
        "source_frames": n,
        "elapsed_seconds": time.monotonic() - start,
        "joint_position_limit_violations": 0,
        "per_hand": reports,
        "warning": "Source-frame finger velocities are constrained and measured; gaps remain unknown, penetration and simulator review still required",
    }
    write_json(out / "quality.json", report)
    return report


def main() -> None:
    """Run the command with validated inputs and explicit output artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mano-to-robot-scale", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            retarget(
                args.bundle,
                args.out,
                args.mano_to_robot_scale,
                args.max_iter,
                args.end_frame,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
