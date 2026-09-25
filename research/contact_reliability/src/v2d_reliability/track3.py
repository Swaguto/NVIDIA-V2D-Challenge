"""Track 3 boundary checks. This module is not an official scoring implementation."""
from pathlib import Path
import json
import re
import numpy as np
from .io import identity, sha256
from .core import aligned_weights

PROXY_EPISODES = (0, 11, 23, 31, 39)
VARIANTS = ("baseline", "force_closure", "reliability", "reliability_force_closure")


def audit_public(public_root):
    """Check real public tables without fitting models or exposing pose values."""
    import pyarrow.parquet as pq
    split = discover_split(public_root)
    reports = []
    for episode, entry in split["files"].items():
        table = pq.read_table(Path(public_root) / entry["path"])
        meta = table.schema.metadata or {}
        if meta.get(b"pose_convention") != b"world_T_object":
            raise ValueError("unexpected public pose convention")
        names = json.loads(meta[b"objects"])
        rows = table.to_pylist()
        times = _times([r["capture_time"] for r in rows], "capture_time")
        frames = np.asarray([r["frame_index"] for r in rows])
        if not np.array_equal(frames, np.arange(len(rows))):
            raise ValueError(f"episode {episode}: noncontiguous public frames")
        if any(r["episode_index"] != int(episode) for r in rows):
            raise ValueError("episode identity mismatch")
        source_names = [o["name"] for o in rows[0]["observation.objects"]]
        if len(set(source_names)) != len(names) or set(source_names) != set(names):
            raise ValueError("public object membership differs from metadata")
        if any([o["name"] for o in r["observation.objects"]] != source_names for r in rows):
            raise ValueError("public object order changes within episode")
        # Metadata membership is sorted; row storage order need not be sorted.
        ordered = [{o["name"]: o for o in r["observation.objects"]} for r in rows]
        poses = np.asarray([[r[n]["pose"] for n in names] for r in ordered], float)
        visible = np.asarray([[r[n]["visible"] for n in names] for r in ordered], bool)
        if not np.isfinite(poses).all() or not np.allclose(np.linalg.norm(poses[visible, 3:], axis=-1), 1, atol=1e-4, rtol=0):
            raise ValueError("invalid visible public quaternion or nonfinite pose")
        if np.any(poses[~visible] != 0):
            raise ValueError("invisible public pose convention changed")
        reports.append({"episode_id": int(episode), "frames": len(rows), "objects": names,
                        "source_object_order": source_names, "canonical_order": names,
                        "source_to_canonical_indices": [source_names.index(n) for n in names],
                        "visible_poses": int(visible.sum()), "invisible_poses": int((~visible).sum()),
                        "duration_seconds": float(times[-1] - times[0]), "sha256": entry["sha256"]})
    return {"schema_version": "public_audit_v1", "episodes": reports,
            "total_frames": sum(r["frames"] for r in reports), "split_hash": split["split_hash"],
            "evidence_level": "real_public_data_schema_validation", "official_score": False}


def discover_split(public_root):
    """Discover actual public parquet files; never manufacture episode IDs."""
    root = Path(public_root)
    files = {}
    for path in sorted((root / "data").rglob("episode_*.parquet")):
        match = re.fullmatch(r"episode_(\d+)\.parquet", path.name)
        if not match:
            continue
        episode = int(match[1])
        if episode in files:
            raise ValueError(f"duplicate episode {episode}")
        files[episode] = {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
    missing = set(PROXY_EPISODES) - files.keys()
    if missing:
        raise ValueError(f"missing proxy episodes: {sorted(missing)}")
    development = sorted(files.keys() - set(PROXY_EPISODES))
    if not development:
        raise ValueError("no development episodes found")
    result = {"schema_version": "track3_split_v1", "development": development,
              "holdout": list(PROXY_EPISODES), "pilot": development[:4],
              "files": {str(k): v for k, v in files.items()}}
    result["split_hash"] = identity(result)
    return result


def validate_split(split):
    payload = {k: v for k, v in split.items() if k != "split_hash"}
    if split.get("schema_version") != "track3_split_v1" or identity(payload) != split.get("split_hash"):
        raise ValueError("invalid split identity")
    if split["holdout"] != list(PROXY_EPISODES):
        raise ValueError("team proxy split changed")
    if set(split["development"]) & set(PROXY_EPISODES):
        raise ValueError("proxy leakage")
    if not set(split["pilot"]) <= set(split["development"]):
        raise ValueError("pilot must belong to development")


def experiment_matrix(split, steps, phase="pilot", floor=.25):
    validate_split(split)
    if type(steps) is not int or steps <= 0:
        raise ValueError("positive measured environment-step budget required")
    if phase not in ("pilot", "holdout") or floor not in (0, .25, .5):
        raise ValueError("invalid phase or confidence floor")
    return [{"episode_id": ep, "sequence_id": f"episode_{ep:06d}", "seed": seed,
             "variant": variant, "environment_steps": steps, "split_hash": split["split_hash"],
             "weight_floor": floor if variant.startswith("reliability") else None,
             "status": "planned", "score_kind": "proxy"}
            for ep in split[phase] for seed in (17, 29, 43) for variant in VARIANTS]


def summarize_matched(rows, episodes, metric="add_auc"):
    """Require all planned cells; retain failures instead of averaging survivors."""
    expected = {(ep, seed, variant) for ep in episodes for seed in (17, 29, 43) for variant in VARIANTS}
    cells = {}
    for row in rows:
        key = (row["episode_id"], row["seed"], row["variant"])
        if key not in expected or key in cells:
            raise ValueError("unexpected or duplicate experiment result")
        if type(row.get("environment_steps")) is not int or row["environment_steps"] <= 0:
            raise ValueError("actual positive environment steps required")
        if row.get("status") not in ("completed", "failed") or type(row.get("rollout_complete")) is not bool:
            raise ValueError("explicit run status and actual rollout completion required")
        if row.get("score_kind") != "proxy":
            raise ValueError("this comparison is proxy-only")
        value = row.get("metrics", {}).get(metric)
        if value is not None and not np.isfinite(value):
            raise ValueError("nonfinite metric")
        if row["status"] == "completed" and value is None:
            raise ValueError("completed evaluation requires metric")
        cells[key] = row
    if set(cells) != expected:
        raise ValueError("missing experiment cells; record failed runs explicitly")
    for ep in episodes:
        for seed in (17, 29, 43):
            group = [cells[ep, seed, v] for v in VARIANTS]
            for field in ("environment_steps", "reference_sha256", "physics_config_hash", "evaluation_config_hash"):
                if any(not r.get(field) for r in group) or len({r[field] for r in group}) != 1:
                    raise ValueError(f"unmatched {field} for episode {ep}, seed {seed}")
    summaries = []
    for variant in VARIANTS:
        group = [r for (ep, seed, v), r in cells.items() if v == variant]
        values = [r.get("metrics", {}).get(metric) for r in group]
        summaries.append({"variant": variant, "runs": len(group),
                          "failed_runs": sum(r["status"] == "failed" for r in group),
                          "incomplete_rollouts": sum(not r["rollout_complete"] for r in group),
                          "metric_mean": float(np.mean(values)) if all(v is not None for v in values) else None})
    return {"metric": metric, "score_kind": "proxy", "summary": summaries, "per_run": rows,
            "selection": "No automatic promotion; inspect failures and per-episode/seed results."}


def _times(values, label):
    t = np.asarray(values, dtype=float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f"{label} must be finite and strictly increasing")
    return t


def validate_trajectory(doc):
    """Validate a portable audit/export envelope, not a replacement motion format.

    Invisible poses may be all zero. They are excluded via the returned mask.
    Quaternion convention is declared, never inferred from numeric values.
    """
    if doc.get("schema_version") != "track3_trajectory_v1":
        raise ValueError("unsupported trajectory schema")
    if type(doc.get("episode_id")) is not int or doc["episode_id"] < 0:
        raise ValueError("nonnegative integer episode_id required")
    if doc.get("units") != "m" or doc.get("quaternion_order") != "wxyz":
        raise ValueError("explicit meters and wxyz required")
    if doc.get("coordinate_frame") != "episode_world":
        raise ValueError("transform into episode_world explicitly before validation")
    if not isinstance(doc.get("frame_provenance"), str) or not doc["frame_provenance"].strip():
        raise ValueError("frame_provenance required")
    names = doc.get("object_ids", [])
    if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("nonempty unique object_ids required")
    assets = doc.get("asset_sha256", {})
    if set(assets) != set(names) or any(not re.fullmatch(r"[0-9a-f]{64}", str(h)) for h in assets.values()):
        raise ValueError("one SHA256 per object asset required")
    times = _times(doc["timestamps"], "timestamps")
    frames = np.asarray(doc["frame_index"])
    if frames.shape != times.shape or frames.dtype.kind not in "iu" or np.any(frames < 0) or np.any(np.diff(frames.astype(np.int64)) != 1):
        raise ValueError("contiguous nonnegative frame_index required")
    poses = np.asarray(doc["poses"], dtype=float)
    visible = np.asarray(doc["visible"])
    if poses.shape != (len(times), len(names), 7) or visible.shape != poses.shape[:-1] or visible.dtype != bool:
        raise ValueError("poses[T,B,7] and boolean visible[T,B] required")
    if not np.isfinite(poses).all():
        raise ValueError("nonfinite pose")
    if not visible.any():
        raise ValueError("no visible poses to validate")
    if not np.allclose(np.linalg.norm(poses[visible, 3:], axis=-1), 1, atol=1e-4, rtol=0):
        raise ValueError("visible quaternion must be unit length")
    return poses, visible


def validate_pair(reference, candidate):
    """Reject silent truncation, reorder, shifted frames and asset mismatches."""
    ref, valid = validate_trajectory(reference)
    got, candidate_valid = validate_trajectory(candidate)
    for key in ("episode_id", "object_ids", "asset_sha256", "frame_index", "coordinate_frame"):
        if reference[key] != candidate[key]:
            raise ValueError(f"candidate/reference {key} mismatch")
    if np.shape(reference["timestamps"]) != np.shape(candidate["timestamps"]) or not np.allclose(reference["timestamps"], candidate["timestamps"], atol=1e-8, rtol=0):
        raise ValueError("candidate/reference timestamp mismatch")
    if np.any(valid & ~candidate_valid):
        raise ValueError("candidate missing a required visible pose")
    return ref, got, valid


def validate_time_mapping(mapping):
    """Require per-sample correspondence; do not guess from nominal FPS."""
    source = _times(mapping["source_timestamps"], "source timestamps")
    motion = _times(mapping["motion_timestamps"], "motion timestamps")
    simulation = _times(mapping["simulation_timestamps"], "simulation timestamps")
    sample = np.asarray(mapping["source_time_at_motion"], float)
    warmup = mapping["warmup_seconds"]
    if not np.isfinite(warmup) or warmup < 0 or motion[0] != 0:
        raise ValueError("nonnegative warmup and zero-origin motion required")
    if sample.shape != motion.shape or simulation.shape != motion.shape or not np.isfinite(sample).all():
        raise ValueError("one source and simulation time per motion sample required")
    if np.any(np.diff(sample) < 0) or sample.min() < source[0] or sample.max() > source[-1]:
        raise ValueError("source mapping must be monotone and within source range")
    if not np.allclose(simulation, motion + warmup, atol=1e-8, rtol=0):
        raise ValueError("simulation time must equal motion time plus warmup")
    return True


def validate_bound_sidecar(sidecar, reference_path, sequence_id, timestamps, hands, objects):
    if sidecar.get("reference_sha256") != sha256(reference_path):
        raise ValueError("sidecar/reference hash mismatch")
    return aligned_weights(sidecar, sequence_id, timestamps, hands, objects)


def transform_poses(poses, rotation_wxyz, translation):
    """Apply a known rigid transform to BOTH position and orientation."""
    result = np.array(poses, dtype=float, copy=True)
    q = np.asarray(rotation_wxyz, float)
    t = np.asarray(translation, float)
    if result.shape[-1:] != (7,) or not np.isfinite(result).all():
        raise ValueError("finite poses[...,7] required")
    if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all() or not np.isclose(np.linalg.norm(q), 1):
        raise ValueError("finite unit transform quaternion and translation required")
    if not np.allclose(np.linalg.norm(result[..., 3:], axis=-1), 1, atol=1e-4, rtol=0):
        raise ValueError("transform only valid unit-quaternion poses")
    w, v = q[0], q[1:]
    p = result[..., :3].copy()
    result[..., :3] = p + 2 * np.cross(v, w * p + np.cross(v, p)) + t
    a, b = result[..., 3].copy(), result[..., 4:].copy()
    result[..., 3] = w * a - np.sum(v * b, axis=-1)
    result[..., 4:] = w * b + a[..., None] * v + np.cross(v, b)
    return result
