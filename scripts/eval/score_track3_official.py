#!/usr/bin/env python3
"""Score Track 3 pose candidates with the OFFICIAL metric on the public proxy episodes.

This applies one constant SE(3) frame-0 warp including rotation. Candidates must
first share a fixed world gauge; moving camera trajectories require conversion.
These reconstruction diagnostics are not a policy-rollout submission score.

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


def load_candidate_json(root: Path, episode: int, kind: str, object_names, steps: int,
                        frame_index=None, return_metadata=False) -> np.ndarray:
    """Read observed rows without filling; match source frame IDs, not row offsets."""
    frames = np.arange(steps) if frame_index is None else np.asarray(frame_index)
    if frames.shape != (steps,) or len(set(frames.tolist())) != steps:
        raise ValueError("reference frame IDs must be unique and match steps")
    npy = root if root.suffix == ".npy" else root / f"e{episode:03d}.npy"
    if npy.is_file():
        poses = np.load(npy, allow_pickle=False)
        if poses.shape != (steps, len(object_names), 7):
            raise ValueError("NPY candidate must be (T,B,7) in reference row/object order")
        sidecar = npy.with_suffix(".json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
            if meta.get("frame_index", frames.tolist()) != frames.tolist() or meta.get("object_names", list(object_names)) != list(object_names):
                raise ValueError("NPY frame/object metadata mismatch")
        poses = np.asarray(poses, dtype=float)
        metadata = {"gt_use": [meta.get("gt_use", "unknown") if sidecar.exists() else "unknown"],
                    "observed": np.isfinite(poses).all(-1).tolist(), "inferred": np.zeros(poses.shape[:2], bool).tolist()}
        return (poses, metadata) if return_metadata else poses
    poses = np.full((steps, len(object_names), 7), np.nan)
    observed = np.zeros((steps, len(object_names)), bool)
    inferred = observed.copy()
    gt_use, sources = set(), {}
    row_by_frame = {int(frame): row for row, frame in enumerate(frames)}
    bases = [root / f"e{episode:03d}" / "ego_object_poses" / f"e{episode:03d}",
             root / "ego_object_poses" / f"e{episode:03d}", root / f"e{episode:03d}", root]
    matches = [p for p in bases if any((p / n / kind).is_dir() for n in object_names)]
    if len(matches) != 1:
        raise ValueError(f"expected one unambiguous episode layout, found {len(matches)} under {root}")
    for b, name in enumerate(object_names):
        d = matches[0] / name / kind
        if kind == "object_to_cam":
            d = d / "ego_cam_c"
        seen = set()
        for f in sorted(d.glob("[0-9]*.json")):
            frame = int(f.stem)
            if frame in seen or frame not in row_by_frame:
                raise ValueError(f"duplicate or unknown frame {frame}: {f}")
            seen.add(frame)
            t = row_by_frame[frame]
            j = json.loads(f.read_text())
            if j.get("frame_index", frame) != frame or j.get("episode", episode) != episode:
                raise ValueError(f"frame/episode metadata mismatch: {f}")
            if not np.allclose(j.get("scale", 1.0), 1.0):
                raise ValueError("pose diagnostic requires metric CAD scale 1")
            if j.get("status", "observed") not in ("observed", "inferred"):
                continue
            if kind == "object_to_world" and j.get("coordinate_frame", "fixed_world") != "fixed_world":
                raise ValueError("candidate is not in a fixed world frame")
            poses[t, b, :3] = j["translation"]
            q = _q2m_wxyz_to_xyzw(j["rotation"]) if "quat_xyzw" not in j else np.asarray(j["quat_xyzw"])
            if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12 or not np.isfinite(poses[t, b, :3]).all():
                raise ValueError(f"invalid visible pose: {f}")
            poses[t, b, 3:] = q / np.linalg.norm(q)
            observed[t, b] = j.get("status", "observed") == "observed"
            inferred[t, b] = not observed[t, b]
            gt_use.add(j.get("gt_use", "unknown"))
            if "camera_source" in j:
                src = j["camera_source"]
                if frame in sources and sources[frame] != src:
                    raise ValueError(f"objects use different camera sources at frame {frame}")
                sources[frame] = src
    metadata = {"observed": observed.tolist(), "inferred": inferred.tolist(),
                "gt_use": sorted(gt_use), "camera_sources": sources}
    return (poses, metadata) if return_metadata else poses


def gap_fill(poses: np.ndarray) -> tuple[np.ndarray, float]:
    """hold-last forward, first-pose backward. Returns (filled, coverage_before_fill)."""
    out = poses.copy()
    total = valid = 0
    for b in range(out.shape[1]):
        p = out[:, b]
        ok = np.isfinite(p).all(axis=1)
        missing = np.isnan(p).all(axis=1)
        if ((~ok) & (~missing)).any() or (np.linalg.norm(p[ok, 3:], axis=1) < 1e-12).any():
            raise ValueError(f"body {b} has malformed predicted rows")
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
    ap.add_argument("--kind", default="object_to_world",
                    choices=["object_to_cam", "object_to_world"])
    ap.add_argument("--episodes", default=DEFAULT_EPISODES)
    ap.add_argument("--data", required=True, type=Path,
                    help="track_3/public root (data/, mesh/, meta/)")
    ap.add_argument("--vertices", type=int, default=500)
    ap.add_argument("--oracle", action="store_true",
                    help="also score GT against itself, as a sanity anchor (expect 1.0)")
    ap.add_argument("--report", type=Path, help="write structured diagnostic results and raw/fill masks")
    args = ap.parse_args()
    if args.kind != "object_to_world":
        ap.error("convert moving-camera observations with convert_camera_poses.py before fixed-world scoring")

    episodes = [int(e) for e in args.episodes.split(",") if e.strip()]
    agg: dict[str, list[float]] = {}
    rows = []

    for ep in episodes:
        ref = load_reference(ep, args.data)
        names = list(ref.object_names)
        clouds = load_vertex_clouds(names, args.data, num_points=args.vertices, seed=0)
        verts = batch_vertices_for(ref, clouds)

        pred, provenance = load_candidate_json(args.predictions, ep, args.kind, names, ref.steps, ref.frame_index, True)
        raw_valid = np.isfinite(pred).all(axis=-1)
        pred, coverage = gap_fill(pred)
        reference = ref.pose_xyzw.copy()
        reference[~ref.visible] = np.nan
        reference, reference_coverage = gap_fill(reference)
        got = score_episode(pred[:, None, :, :], reference, verts)
        rows.append({"episode": ep, "objects": names, "metrics": got,
            "frame_index": ref.frame_index.tolist(), "capture_time": ref.capture_time.tolist(),
            "stored_coverage": coverage,
            "raw_observed_coverage_by_object": dict(zip(names, np.asarray(provenance["observed"]).mean(0).tolist())),
            "gap_fill_mask": (~raw_valid).tolist(), "reference_visible_mask": ref.visible.tolist(),
            "provenance": provenance, "reference_visible_coverage": reference_coverage})

        if args.oracle:
            orc = score_episode(reference[:, None, :, :], reference, verts)
            assert abs(orc["add_auc"] - 1.0) < 1e-9, orc
        for k, v in got.items():
            agg.setdefault(k, []).append(v)

        print(f"  ep {ep:>3}  {','.join(names)[:44]:44s} cov={coverage:5.1%}  "
              f"AUC={got['add_auc']:.4f}  MP-SR={got['maniptrans_sr']:.2f}  "
              f"SP-SR={got['spider_sr']:.2f}  RPE={got['rpe_cm']:6.2f}cm  "
              f"MPPE={got['mppe_cm']:6.2f}cm")
        observed_fraction = np.asarray(provenance["observed"]).mean(0).tolist()
        inferred_fraction = np.asarray(provenance["inferred"]).mean(0).tolist()
        print(f"    visually observed by object: {dict(zip(names, observed_fraction))}; "
              f"inferred: {dict(zip(names, inferred_fraction))}; "
              f"reference visible coverage={reference_coverage:.1%}; remaining gaps held/backfilled")

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
    if args.report:
        if args.report.exists():
            raise ValueError("report already exists; choose a new diagnostic report path")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"kind": "reconstruction_diagnostic",
            "alignment": "one constant SE(3), slot 0/frame 0, shared across bodies and times",
            "fill_policy": "hold-last then head backfill; candidate and invisible reference rows are flagged",
            "episodes": rows, "mean_metrics": {k: float(np.mean(v)) for k, v in agg.items()}}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
