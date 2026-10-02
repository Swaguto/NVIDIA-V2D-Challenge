#!/usr/bin/env python3
"""Score Track 3 pose candidates with the OFFICIAL metric on the public proxy episodes.

Unlike ``score_candidate.py`` (a proxy whose ``--align-rigid`` leaves quaternions
untouched), this applies the official SE(3) frame-0 warp including rotation, so a
candidate expressed in the egocentric camera frame is scored exactly as it would be
uploaded. See ``official_metric.py`` for why that matters.

Predictions may be given either as per-episode ``.npy`` of shape ``(T, B, 7)`` (xyzw,
metres, one row per frame, bodies in reference order) or as the per-frame JSON layout
written by ``fit_object_pose``::

    <predictions>/<episode>/<object>/<frame:06d>.json   {"translation": [...], "rotation": wxyz}

Frames present in the reference but missing from the candidate are gap-filled
(hold-last, back-filled at the head). The official scorer rejects a submission with any
missing row outright, so this mirrors what a submittable trajectory has to look like.

Example
-------
    python scripts/eval/score_track3_official.py \
        --predictions /home/ubuntu/v2d_calib_work --kind object_to_cam \
        --episodes 0,11,23,31,39 \
        --data /home/ubuntu/v2d_track3/data/hf/track_3/public
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))

from official_metric import score_episode                      # noqa: E402
from reference_loader import (                                 # noqa: E402
    batch_vertices_for, load_reference, load_vertex_clouds,
)

DEFAULT_EPISODES = "0,11,23,31,39"


def _q2m_wxyz_to_xyzw(wxyz) -> np.ndarray:
    q = np.asarray(wxyz, dtype=np.float64)
    return np.array([q[1], q[2], q[3], q[0]])


def load_candidate_json(root: Path, episode: int, kind: str, object_names, steps: int) -> np.ndarray:
    """Read the fit_object_pose JSON layout and gap-fill to ``steps`` frames."""
    poses = np.full((steps, len(object_names), 7), np.nan)
    for b, name in enumerate(object_names):
        d = root / f"e{episode:03d}" / "ego_object_poses" / f"e{episode:03d}" / name / kind
        if kind == "object_to_cam":
            d = d / "ego_cam_c"
        for t in range(steps):
            f = d / f"{t:06d}.json"
            if not f.exists():
                continue
            j = json.loads(f.read_text())
            poses[t, b, :3] = j["translation"]
            q = _q2m_wxyz_to_xyzw(j["rotation"]) if "quat_xyzw" not in j else np.asarray(j["quat_xyzw"])
            poses[t, b, 3:] = q / np.linalg.norm(q)
    return poses


def gap_fill(poses: np.ndarray) -> tuple[np.ndarray, float]:
    """hold-last forward, first-pose backward. Returns (filled, coverage_before_fill)."""
    out = poses.copy()
    total = valid = 0
    for b in range(out.shape[1]):
        p = out[:, b]
        ok = ~np.isnan(p[:, 0])
        total += len(p)
        valid += int(ok.sum())
        if not ok.any():
            raise ValueError(f"body {b} has no predicted frames at all")
        first = int(np.argmax(ok))
        p[:first] = p[first]
        idx = np.where(ok, np.arange(len(p)), 0)
        np.maximum.accumulate(idx, out=idx)
        p[~ok] = p[idx[~ok]]
    return out, valid / max(total, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", required=True, type=Path,
                    help="root of the per-episode prediction directories")
    ap.add_argument("--kind", default="object_to_cam",
                    choices=["object_to_cam", "object_to_world"])
    ap.add_argument("--episodes", default=DEFAULT_EPISODES)
    ap.add_argument("--data", required=True, type=Path,
                    help="track_3/public root (data/, mesh/, meta/)")
    ap.add_argument("--vertices", type=int, default=500)
    ap.add_argument("--oracle", action="store_true",
                    help="also score GT against itself, as a sanity anchor (expect 1.0)")
    args = ap.parse_args()

    episodes = [int(e) for e in args.episodes.split(",") if e.strip()]
    agg: dict[str, list[float]] = {}
    rows = []

    for ep in episodes:
        try:
            ref = load_reference(ep, args.data)
        except Exception as error:                        # noqa: BLE001
            print(f"  ep {ep}: no public reference ({error}); skipped")
            continue
        names = list(ref.object_names)
        clouds = load_vertex_clouds(names, args.data, num_points=args.vertices, seed=0)
        verts = batch_vertices_for(ref, clouds)

        pred = load_candidate_json(args.predictions, ep, args.kind, names, ref.steps)
        pred, coverage = gap_fill(pred)
        got = score_episode(pred[:, None, :, :], ref.pose_xyzw, verts)
        rows.append((ep, names, coverage, got))

        if args.oracle:
            orc = score_episode(ref.pose_xyzw[:, None, :, :], ref.pose_xyzw, verts)
            assert abs(orc["add_auc"] - 1.0) < 1e-9, orc
        for k, v in got.items():
            agg.setdefault(k, []).append(v)

        print(f"  ep {ep:>3}  {','.join(names)[:44]:44s} cov={coverage:5.1%}  "
              f"AUC={got['add_auc']:.4f}  MP-SR={got['maniptrans_sr']:.2f}  "
              f"SP-SR={got['spider_sr']:.2f}  RPE={got['rpe_cm']:6.2f}cm  "
              f"MPPE={got['mppe_cm']:6.2f}cm")

    if not rows:
        print("no episodes scored")
        return 1

    print(f"\nOFFICIAL metric, {args.kind}, {len(rows)} episode(s), "
          f"frame-0 SE(3) alignment incl. rotation")
    print(f"  {'AUC':>8} {'MP-SR':>7} {'SP-SR':>7} {'RPE_cm':>8} {'MPPE_cm':>9}")
    print(f"  {np.mean(agg['add_auc']):>8.4f} {np.mean(agg['maniptrans_sr']):>7.3f} "
          f"{np.mean(agg['spider_sr']):>7.3f} {np.mean(agg['rpe_cm']):>8.3f} "
          f"{np.mean(agg['mppe_cm']):>9.3f}")
    print("  (higher better: AUC, MP-SR, SP-SR | lower better: RPE, MPPE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())