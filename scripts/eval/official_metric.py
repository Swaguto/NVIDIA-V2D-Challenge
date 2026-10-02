#!/usr/bin/env python3
"""Faithful reproduction of the OFFICIAL Track 3 pose metrics.

Source of truth: ``v2d_submission_kit/metric_code/track_3/*.py`` from the participant kit
(``assets/v2d_submission_kit.zip`` on the challenge site). This module mirrors those
functions one-for-one so a local leaderboard agrees with Kaggle.

Why this module exists
----------------------
``scripts/eval/score_candidate.py`` is a *proxy*, and its ``--align-rigid`` is not the
official alignment: it writes ``achieved[..., :3]`` only and leaves quaternions
untouched, so it silently reports rotation error that the official scorer would have
removed. The official scorer performs a full SE(3) frame-0 warp **including rotation**
(``_ALIGN_INITIAL_POSE = True`` in ``AUC.py``). Consequence: a candidate expressed in the
egocentric camera frame can be submitted directly, with no camera-to-world calibration.

Two details that are easy to get wrong, and that this module gets right:

1. ``_warp_to_reference_start`` derives one transform per **world** from **object slot 0
   only** (``reference[0, 0, 0]`` / ``achieved[0, :, 0]``), then applies it to *every*
   body and frame. It is not a per-body alignment.
2. The warp is a rigid SE(3); **scale is not** corrected. README: "Trajectory scale error
   remains scored."

Poses are ``(T, W, B, 7)`` xyzw quaternions in metres. Reference is ``(T, B, 7)`` (the
official scorer drops all but world 0 via ``reference[:, :1]``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# --- constants, verbatim from metric_code/track_3/AUC.py ---------------------
ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)          # 1..9 cm inclusive of 9
SPIDER_POSITION_THRESHOLD_M = 0.1
SPIDER_ORIENTATION_THRESHOLD_RAD = 0.5
MANIPTRANS_POSITION_THRESHOLD_M = 0.03
MANIPTRANS_ORIENTATION_THRESHOLD_DEG = 30.0
MPPE_KEYPOINT_RADIUS_M = 0.05
MPPE_KEYPOINT_DIRECTIONS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64,
)
ALIGN_INITIAL_POSE = True


# --- primitives --------------------------------------------------------------
def rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (quat_xyzw[..., i] for i in range(4))
    return np.stack([
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
        2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y),
    ], axis=-1).reshape(*quat_xyzw.shape[:-1], 3, 3)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quaternion_geodesic_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot)


def normalise_quaternions(pose: np.ndarray, who: str) -> np.ndarray:
    pose = pose.copy()
    norms = np.linalg.norm(pose[..., 3:], axis=-1)
    if np.any(norms < 1e-8):
        raise ValueError(f"{who} contains a degenerate (near-zero-norm) quaternion")
    pose[..., 3:] /= norms[..., None]
    return pose


# --- the alignment that matters ---------------------------------------------
def warp_to_reference_start(achieved: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Official frame-0 SE(3) warp. achieved (T,W,B,7), reference (T,1,B,7).

    One transform per world, derived from **object slot 0 only**, applied to every
    body and frame. Rotates positions *and* quaternions.
    """
    reference_start = reference[0, 0, 0]
    achieved_start = achieved[0, :, 0]
    conjugate = achieved_start[:, 3:] * np.array([-1.0, -1.0, -1.0, 1.0])
    q_align = quaternion_multiply(reference_start[None, 3:], conjugate)
    q_align = q_align / np.linalg.norm(q_align, axis=-1, keepdims=True)
    rot = rotation_matrix(q_align)
    offset = reference_start[None, :3] - np.einsum("wij,wj->wi", rot, achieved_start[:, :3])

    warped = np.empty_like(achieved)
    warped[..., :3] = np.einsum("wij,twbj->twbi", rot, achieved[..., :3]) + offset[None, :, None, :]
    warped[..., 3:] = quaternion_multiply(q_align[None, :, None, :], achieved[..., 3:])
    return warped


# --- metrics -----------------------------------------------------------------
def object_add(achieved: np.ndarray, reference: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """achieved (T,W,B,7), reference (T,1,B,7), vertices (B,V,3). -> (T,W,B)"""
    T, W, B, _ = achieved.shape
    delta_rot = rotation_matrix(achieved[..., 3:]) - rotation_matrix(reference[..., 3:])
    delta_pos = achieved[..., :3] - reference[..., :3]
    add = np.empty((T, W, B), dtype=np.float64)
    for b in range(B):
        offs = np.einsum("twij,vj->twvi", delta_rot[:, :, b], vertices[b])
        offs = offs + delta_pos[:, :, b, None, :]
        add[:, :, b] = np.linalg.norm(offs, axis=-1).mean(axis=-1)
    return add


def add_auc(add: np.ndarray) -> float:
    """Per-body AUC over the 1..9 cm thresholds, then averaged over bodies."""
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    axis = np.linspace(0.0, 1.0, ADD_THRESHOLDS_M.size)
    per_body = [float(trapezoid(np.array([np.mean(add[..., b] < t)
                                         for t in ADD_THRESHOLDS_M]), x=axis))
                for b in range(add.shape[2])]
    return float(np.mean(per_body))


def per_world_pose_error(achieved: np.ndarray, reference: np.ndarray):
    position = np.linalg.norm(achieved[..., :3] - reference[..., :3], axis=-1).mean(axis=0)
    orientation = quaternion_geodesic_angle(achieved[..., 3:], reference[..., 3:]).mean(axis=0)
    return position, orientation


def spider_success(achieved: np.ndarray, reference: np.ndarray) -> np.ndarray:
    position, orientation = per_world_pose_error(achieved, reference)
    return (position.mean(axis=-1) <= SPIDER_POSITION_THRESHOLD_M) & (
        orientation.mean(axis=-1) <= SPIDER_ORIENTATION_THRESHOLD_RAD)


def maniptrans_success(achieved: np.ndarray, reference: np.ndarray, object_ids: np.ndarray) -> np.ndarray:
    """Every tracked object must clear 3 cm and 30 deg."""
    position, orientation = per_world_pose_error(achieved, reference)
    orientation_deg = np.degrees(orientation)
    success = np.ones(achieved.shape[1], dtype=bool)
    for oid in np.unique(object_ids):
        members = object_ids == oid
        success &= (position[:, members].mean(axis=-1) < MANIPTRANS_POSITION_THRESHOLD_M) & (
            orientation_deg[:, members].mean(axis=-1) < MANIPTRANS_ORIENTATION_THRESHOLD_DEG)
    return success


def relative_position_error_cm(achieved: np.ndarray, reference: np.ndarray) -> float:
    """One object's position expressed in another object's frame. 0 for single-object."""
    bodies = achieved.shape[2]
    if bodies < 2:
        return 0.0
    errors = []
    for first in range(bodies):
        for second in range(first + 1, bodies):
            rot_a = rotation_matrix(achieved[:, :, first, 3:])
            rot_r = rotation_matrix(reference[:, :, first, 3:])
            d_a = achieved[:, :, second, :3] - achieved[:, :, first, :3]
            d_r = reference[:, :, second, :3] - reference[:, :, first, :3]
            local_a = np.einsum("twji,twj->twi", rot_a, d_a)
            local_r = np.einsum("twji,twj->twi", rot_r, d_r)
            errors.append(np.linalg.norm(local_a - local_r, axis=-1))
    return float(np.mean(np.stack(errors)) * 100.0)


def mean_per_point_error_cm(achieved: np.ndarray, reference: np.ndarray) -> float:
    offsets = MPPE_KEYPOINT_DIRECTIONS * MPPE_KEYPOINT_RADIUS_M
    rot_a = rotation_matrix(achieved[..., 3:])
    rot_r = rotation_matrix(reference[..., 3:])
    kp_a = achieved[..., None, :3] + np.einsum("...ij,kj->...ki", rot_a, offsets)
    kp_r = reference[..., None, :3] + np.einsum("...ij,kj->...ki", rot_r, offsets)
    return float(np.linalg.norm(kp_a - kp_r, axis=-1).mean() * 100.0)


# --- episode scorer ----------------------------------------------------------
def score_episode(achieved: np.ndarray, reference: np.ndarray, vertices: np.ndarray,
                  object_ids: np.ndarray | None = None) -> dict:
    """achieved (T,W,B,7) xyzw; reference (T,B,7); vertices (B,V,3). Returns all 5 metrics."""
    if object_ids is None:
        object_ids = np.arange(achieved.shape[2], dtype=np.int64)
    achieved = normalise_quaternions(np.asarray(achieved, np.float64), "submission")
    reference = normalise_quaternions(np.asarray(reference, np.float64), "reference")
    reference = reference[:, None, :, :]          # official keeps world 0 only

    if ALIGN_INITIAL_POSE:
        achieved = warp_to_reference_start(achieved, reference)

    return {
        "add_auc": add_auc(object_add(achieved, reference, vertices)),
        "spider_sr": float(np.mean(spider_success(achieved, reference))),
        "maniptrans_sr": float(np.mean(maniptrans_success(achieved, reference, object_ids))),
        "rpe_cm": relative_position_error_cm(achieved, reference),
        "mppe_cm": mean_per_point_error_cm(achieved, reference),
    }


def selfcheck() -> None:
    """Oracle must score 1.0 AUC / 0 RPE; a rotated copy must score 1.0 too.

    The second assertion is the whole point: the official warp removes a constant
    world-frame rotation, which the proxy scorer does not.
    """
    rng = np.random.default_rng(0)
    T, B, V, W = 40, 2, 64, 1
    ref = np.zeros((T, B, 7))
    ref[..., 3] = 1.0
    ref[..., :3] = rng.normal(scale=0.2, size=(T, B, 3))
    verts = rng.normal(scale=0.05, size=(B, V, 3))

    ach = np.repeat(ref[:, None], W, axis=1).copy()        # (T,W,B,7)
    got = score_episode(ach, ref, verts)
    assert abs(got["add_auc"] - 1.0) < 1e-9, got
    assert got["rpe_cm"] < 1e-6 and got["mppe_cm"] < 1e-6, got

    # rotate + translate the whole trajectory by one constant rigid transform
    q = np.array([0.2, -0.3, 0.1, 0.9]); q /= np.linalg.norm(q)
    R = rotation_matrix(q)
    off = np.array([0.5, -0.2, 1.0])
    warped = ach.copy()
    warped[..., :3] = np.einsum("ij,twbj->twbi", R, warped[..., :3]) + off
    warped[..., 3:] = quaternion_multiply(np.broadcast_to(q, warped[..., 3:].shape), warped[..., 3:])
    got2 = score_episode(warped, ref, verts)
    assert abs(got2["add_auc"] - 1.0) < 1e-9, got2
    print("selfcheck OK: oracle=1.0 and a constant rigid gauge change also scores 1.0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--candidate", help="json: {'poses': (T,B,7) xyzw, 'frame_index': [...]}")
    ap.add_argument("--reference", help="json: {'poses': (T,B,7) xyzw, 'vertices': (B,V,3)}")
    args = ap.parse_args()
    if args.selfcheck or not (args.candidate and args.reference):
        selfcheck()
        return 0
    cand = json.loads(Path(args.candidate).read_text())
    ref = json.loads(Path(args.reference).read_text())
    res = score_episode(np.asarray(cand["poses"], np.float64)[None],
                        np.asarray(ref["poses"], np.float64),
                        np.asarray(ref["vertices"], np.float64))
    for k, v in res.items():
        print(f"{k:>12}: {v:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())