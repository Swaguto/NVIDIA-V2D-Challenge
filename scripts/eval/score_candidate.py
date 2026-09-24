#!/usr/bin/env python3
"""Track 3 E2E scoring harness for the private proxy test set.

Scores any candidate object-trajectory output (reconstruction or recorded rollout)
against the held-out public-episode ground truth using the CHORD metric suite:

    ADD-AUC  (DexMachina ADD, area under accuracy vs 1..9 cm threshold)
    SP-SR    (SPIDER per-episode success: frame-mean position <= 10 cm & rot <= ~28.6 deg)
    MP-SR    (ManipTrans per-object success: every object < 3 cm and < 30 deg)
    MPPE     (CHORD mean per-frame keypoint pose error, cm)

Candidate formats:
  1. Parquet with the same ``observation.objects`` schema as the dataset (reconstruction).
  2. ``.npy`` array ``[T, B, 7]`` or ``[T, W, B, 7]`` of poses in **xyzw** (world already
     matched to GT) — a recorded rollout of one or many worlds.

Usage:
    python scripts/eval/score_candidate.py \
        --candidate data/outputs/ep_000000.parquet [--episodes 0]
    python scripts/eval/score_candidate.py --candidate data/outputs/ep_{:06d}.parquet
    python scripts/eval/score_candidate.py --candidate rollout_ep0.npy --episodes 0 --mode rollout
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chord_metrics import (  # noqa: E402
    TrackingThresholds,
    compute_object_tracking_metrics,
)
from reference_loader import (  # noqa: E402
    PROXY_EPISODES,
    batch_vertices_for,
    load_reference,
    load_vertex_clouds,
)

_W2Xyzw = np.array([0, 1, 2, 4, 5, 6, 3], dtype=np.int64)

_CHORD_THRESHOLDS = {"wrist_position_m": None, "object_position_m": None}


def _rmsd_finite(achieved: np.ndarray) -> float:
    return float(np.linalg.norm(achieved[..., :3]) / np.sqrt(max(1, achieved.size // 7)))


def load_candidate_parquet(path: Path, ref_names: Sequence[str]) -> np.ndarray:
    """Reconstruction candidate: same schema as GT reference -> ``(T, B, 7)`` xyzw."""
    pf = pq.ParquetFile(path)
    table = pf.read()
    meta = pf.schema_arrow.metadata
    objects = table.column("observation.objects").to_pylist()
    steps = len(objects)
    bodies = len(ref_names)
    pose = np.empty((steps, bodies, 7), dtype=np.float64)
    for t, frame_objects in enumerate(objects):
        by_name = {o["name"]: o for o in frame_objects}
        for b, name in enumerate(ref_names):
            if name not in by_name:
                raise KeyError(f"candidate frame {t} missing object {name!r}")
            pose[t, b] = np.asarray(by_name[name]["pose"], dtype=np.float64)
    return pose[..., _W2Xyzw]


def candidate_npy(path: Path, ref_steps: int, world_axis: int = 1) -> np.ndarray:
    """Recorded rollout: ``[T, B, 7]`` or ``[T, W, B, 7]`` numpy poses (xyzw)."""
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[2] == 7:
        return arr[:, None, :, :]  # (T, 1, B, 7)
    if arr.ndim == 4 and arr.shape[3] == 7:
        return arr
    raise ValueError(f"candidate npy must be [T,B,7] or [T,W,B,7], got {arr.shape}")


def rigid_align(
    achieved: np.ndarray, reference: np.ndarray, visible: np.ndarray, pass_pts: int = 20
) -> np.ndarray:
    """Umeyama rigid alignment of the whole trajectory to the reference frame.

    Only translation+rotation (no isotropic scale) over visible frames, sampled to
    keep memory small.  Applies the same transform to every world.
    """
    achieved = np.asarray(achieved, dtype=np.float64)
    T, B, _ = reference.shape
    world_count = achieved.shape[1]
    mask = np.tile(visible.any(axis=-1), (B, 1)).T if visible.ndim == 2 else np.ones((T, B), bool)
    idx = np.where(mask[:, 0])[0]
    if idx.size == 0:
        return achieved
    sample = idx[np.linspace(0, idx.size - 1, min(pass_pts, idx.size)).astype(int)]
    a = achieved[sample, 0, ..., :3]  # (S, B, 3)
    r = reference[sample, ..., :3]  # (S, B, 3)
    a_flat = a.reshape(-1, 3)
    r_flat = r.reshape(-1, 3)
    a_mean = a_flat.mean(axis=0)
    r_mean = r_flat.mean(axis=0)
    a_c = a_flat - a_mean
    r_c = r_flat - r_mean
    cov = r_c.T @ a_c
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    diag = np.array([1.0, 1.0, d])
    rotation = u @ np.diag(diag) @ vt
    translation = r_mean - rotation @ a_mean
    transformed = achieved.copy()
    transformed[..., :3] = (achieved[..., :3] @ rotation.T) + translation
    return transformed


def score_episode(
    episode_index: int,
    candidate: np.ndarray,  # (T, W, B, 7) xyzw
    reference_base_dir: str | os.PathLike,
    world_chunk: int = 8,
) -> dict:
    ref = load_reference(episode_index, reference_base_dir)
    clouds = load_vertex_clouds(ref.object_names, reference_base_dir, num_points=500, seed=0)
    vertices = batch_vertices_for(ref, clouds)
    body_object_ids = np.arange(ref.bodies, dtype=np.int64)
    body_names = list(ref.object_names)

    achieved = candidate
    time_candidate = achieved.shape[0]
    if time_candidate < ref.steps:
        raise ValueError(
            f"candidate has {time_candidate} steps but reference has {ref.steps} for ep {episode_index}"
        )
    achieved = achieved[: ref.steps]

    # Single trailing comparison per body => tracking_error = per-frame object errors.
    pos_err = np.linalg.norm(achieved[..., :3] - ref.pose_xyzw[:, None, ..., :3], axis=-1)  # (T,W,B)
    achieved_q = achieved[..., 3:] / np.clip(
        np.linalg.norm(achieved[..., 3:], axis=-1, keepdims=True), 1e-12, None
    )
    ref_q = ref.pose_xyzw[..., 3:]
    dot = np.clip(np.abs((achieved_q * ref_q[:, None]).sum(axis=-1)), 0.0, 1.0)
    rot_err = 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot)  # (T,W,B)
    object_pos_err = np.max(pos_err, axis=-1, keepdims=True)  # (T,W,1)
    object_rot_err = np.max(rot_err, axis=-1, keepdims=True)
    tracking_error = np.concatenate(
        [
            np.zeros_like(object_pos_err),  # wrist pos
            np.zeros_like(object_pos_err),  # wrist rot
            object_pos_err,  # object position (max over bodies)
            object_rot_err,
        ],
        axis=-1,
    )

    class _Completion:
        terminated = np.array([False] * achieved.shape[1], dtype=np.bool_)
        truncated = np.array([True] * achieved.shape[1], dtype=np.bool_)
        reference_progress = np.array([1.0] * achieved.shape[1], dtype=np.float64)

    metrics = compute_object_tracking_metrics(
        achieved,
        ref.pose_xyzw,
        vertices,
        body_object_ids,
        tracking_error,
        TrackingThresholds(),
        _Completion(),
        body_names,
        world_chunk=world_chunk,
    )
    return {
        "episode_index": int(episode_index),
        "objects": body_names,
        "steps": int(ref.steps),
        "worlds": int(achieved.shape[1]),
        "add_auc": metrics.add_auc,
        "add_auc_per_body": list(metrics.add_auc_per_body),
        "sp_sr": metrics.spider_sr_uncentered,
        "mp_sr": metrics.maniptrans_sr,
        "mppe_cm": metrics.mppe_cm,
        "mean_add_m": metrics.mean_add_m,
        "obj_pos_err_m": metrics.object_position_error_m,
        "obj_rot_err_deg": metrics.object_orientation_error_deg,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        required=True,
        help="Parquet (reconstruction, schema observation.objects) or .npy rollout.",
    )
    parser.add_argument(
        "--episodes",
        default=",".join(map(str, PROXY_EPISODES)),
        help="Comma list or single episode index to score against (default: proxy set).",
    )
    parser.add_argument("--mode", choices=["auto", "reconstruction", "rollout"], default="auto")
    parser.add_argument(
        "--data", default=os.path.expanduser("~/v2d_track3/data/hf/track_3/public")
    )
    parser.add_argument("--align-rigid", action="store_true", help="Umeyama-align candidate to GT first.")
    parser.add_argument("--episode-pattern", action="store_true", help="Treat --candidate as a style pattern w/ {:06d} placeholder")
    parser.add_argument("--json", default=None, help="Write JSON report here.")
    args = parser.parse_args(argv)

    episodes = [int(e.strip()) for e in args.episodes.split(",") if e.strip()]
    suffix = Path(args.candidate).suffix.lower()
    mode = args.mode

    results = []
    for episode_index in episodes:
        ref = load_reference(episode_index, args.data)
        if args.episode_pattern:
            candidate_path = Path(str(args.candidate).format(episode_index))
        else:
            candidate_path = Path(args.candidate)
        if not candidate_path.exists():
            raise FileNotFoundError(candidate_path)

        if mode == "reconstruction" or (mode == "auto" and suffix in (".parquet", ".pq")):
            candidate = load_candidate_parquet(candidate_path, ref.object_names)[:, None, :, :]
        elif mode == "rollout" or (mode == "auto" and suffix == ".npy"):
            candidate = candidate_npy(candidate_path, ref.steps)
        else:
            raise ValueError(f"cannot infer candidate mode for suffix {suffix!r}; use --mode")

        if args.align_rigid:
            candidate = rigid_align(candidate, ref.pose_xyzw, ref.visible)

        result = score_episode(episode_index, candidate, args.data)
        results.append(result)

    auc = float(np.mean([r["add_auc"] for r in results]))
    sp = float(np.mean([r["sp_sr"] for r in results]))
    mp = float(np.mean([r["mp_sr"] for r in results]))
    mppe = float(np.mean([r["mppe_cm"] for r in results]))
    mean_add = float(np.mean([r["mean_add_m"] for r in results]))

    print(f"\nTrack 3 proxy scorecard  ({len(results)} episode(s), worlds={results[0]['worlds']})")
    print(f"{'ep':>3}  {'objects':28s} {'steps':>5}  {'ADD-AUC':>7} {'SP-SR':>6} {'MP-SR':>6} {'MPPE_cm':>7} {'ADD_m':>6}")
    for r in results:
        print(
            f"{r['episode_index']:>3}  {','.join(r['objects']):28s} {r['steps']:>5}  "
            f"{r['add_auc']:>7.3f} {r['sp_sr']:>6.3f} {r['mp_sr']:>6.3f} {r['mppe_cm']:>7.3f} {r['mean_add_m']:>6.3f}"
        )
    print(f"\nMEAN  ADD-AUC={auc:.3f}  SP-SR={sp:.3f}  MP-SR={mp:.3f}  MPPE={mppe:.2f} cm  meanADD={mean_add:.3f} m")

    report = {
        "metrics": {
            "add_auc": float(auc),
            "sp_sr": float(sp),
            "mp_sr": float(mp),
            "mppe_cm": float(mppe),
            "mean_add_m": float(mean_add),
        },
        "per_episode": results,
        "episodes": episodes,
        "candidate": str(args.candidate),
        "note": (
            "CHORD metric definitions: ADD-AUC from DexMachina (500 vertex points, "
            "thresholds 1-9 cm, trapezoid-vs-normalized-threshold); SP-SR from SPIDER "
            "(frame-mean pos<=10cm & rot<=0.5rad); MP-SR from ManipTrans (every object "
            "<3cm & <30deg); MPPE from CHORD (6 keypoints at +/-5cm, cm)."
        ),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())