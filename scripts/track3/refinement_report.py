#!/usr/bin/env python3
"""Per-episode quality report for the gsplat joint refinement (issue #7).

Reads the ``--scene_export_path`` archive written by ``refine_simple`` (initial
and refined camera, object and hand state over the same frames) and writes:

* ``report.json`` / ``report.md``: jitter, drift and contact-precision numbers
  for the initial and refined states, side by side.
* ``viewer.html``: a self-contained Three.js viewer of the episode
  (background points, CAD objects, MANO hands, camera path), initial vs refined.
* ``scoring/{initial,refined}/``: object_to_world poses in the Track 3 layout,
  so ``scripts/eval/score_track3_official.py`` can score both against the
  public reference. With ``--data`` the scorer is run here as well.

Every number except the optional official score is GT-free:

* **Jitter**: RMS / median / p95 of acceleration from second differences over
  consecutive frames (m/s^2, deg/s^2), and the RMS of the high-frequency
  residual after a centred 5-frame moving average (mm).
* **Drift**: while no hand is within ``--rest-clearance`` of an object, the
  real object is resting and must stay fixed in the world. Its apparent world
  motion over each rest segment (start-to-end, and max deviation from the
  segment median) measures camera/world drift plus object-pose wander.
* **Contact precision**: during each contact run (from issue-6
  ``contacts.json`` when given, otherwise fingertip within 2 cm in the refined
  state), the touch point on the object surface should stay put. We report its
  RMS spread in the object frame, the normal spread, the median fingertip-to-
  surface distance, and the penetration rate (signed distance < -5 mm).
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# MANO fingertip vertices (manopth/manotorch convention), thumb..little.
MANO_TIP_VERTS = {"thumb": 745, "index": 317, "middle": 444, "ring": 556, "little": 673}
STATES = ("initial", "refined")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def invert_rigid(M: np.ndarray) -> np.ndarray:
    """Inverse of (...,4,4) rigid transforms."""
    R = M[..., :3, :3]
    t = M[..., :3, 3]
    out = np.zeros_like(M)
    out[..., :3, :3] = np.swapaxes(R, -1, -2)
    out[..., :3, 3] = -np.einsum("...ji,...j->...i", R, t)
    out[..., 3, 3] = 1.0
    return out


def apply(M: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply (4,4) to (...,3) points, or (T,4,4) to (T,N,3)."""
    if M.ndim == 2:
        return points @ M[:3, :3].T + M[:3, 3]
    return np.einsum("tij,tnj->tni", M[:, :3, :3], points) + M[:, None, :3, 3]


def rotation_log(R: np.ndarray) -> np.ndarray:
    """(...,3,3) rotation matrices -> (...,3) rotation vectors (radians)."""
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    angle = np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0))
    w = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=-1)
    sin = np.sin(angle)
    small = sin < 1e-6
    out = w * np.where(small, 0.5, angle / (2.0 * np.where(small, 1.0, sin)))[:, None]
    # Near pi the antisymmetric part vanishes; recover the axis from (R + I) / 2.
    for i in np.nonzero(angle > np.pi - 1e-4)[0]:
        B = (R[i] + np.eye(3)) / 2.0
        axis = B[int(np.argmax(np.diag(B)))]
        out[i] = axis / max(np.linalg.norm(axis), 1e-12) * angle[i]
    return out.reshape(*shape, 3)


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def orthonormalize(M: np.ndarray) -> np.ndarray:
    """Strip a uniform scale from the rotation block of (...,4,4) poses."""
    out = M.copy()
    s = np.cbrt(np.linalg.det(M[..., :3, :3]))
    out[..., :3, :3] = M[..., :3, :3] / s[..., None, None]
    return out


def sample_surface(verts: np.ndarray, faces: np.ndarray, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Area-weighted surface samples, their face normals, and the mean sample spacing."""
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(area, 1e-12)[:, None]
    rng = np.random.default_rng(seed)
    face = rng.choice(len(faces), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    t = tri[face]
    pts = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    return pts.astype(np.float64), normals[face].astype(np.float64), float(np.sqrt(area.sum() / 2.0 / n))


def nearest_surface(points: np.ndarray, surface: np.ndarray, normals: np.ndarray, spacing: float, chunk: int = 256):
    """Closest surface point, its normal, and unsigned / signed distance.

    Uses point-to-plane distance against the nearest sample's face, with the
    in-plane offset beyond one sample spacing added back, so flat regions are
    measured exactly rather than to the sample grid.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    idx = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        d2 = ((p[:, None, :] - surface[None, :, :]) ** 2).sum(-1)
        idx[start:start + chunk] = d2.argmin(1)
    q, nrm = surface[idx], normals[idx]
    diff = points - q
    plane = (diff * nrm).sum(1)
    lateral = np.linalg.norm(diff - plane[:, None] * nrm, axis=1)
    dist = np.sqrt(plane ** 2 + np.maximum(lateral - spacing, 0.0) ** 2)
    touch = points - plane[:, None] * nrm
    return touch, nrm, dist, np.where(plane < 0, -dist, dist)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def consecutive_runs(frames: np.ndarray) -> list[np.ndarray]:
    """Split positional indices into runs of consecutive frame numbers."""
    frames = np.asarray(frames)
    if len(frames) == 0:
        return []
    breaks = np.nonzero(np.diff(frames) != 1)[0] + 1
    return np.split(np.arange(len(frames)), breaks)


def _stats(values: np.ndarray, prefix: str, unit: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {f"{prefix}_rms_{unit}": None, f"{prefix}_median_{unit}": None, f"{prefix}_p95_{unit}": None, "n": 0}
    return {
        f"{prefix}_rms_{unit}": float(np.sqrt(np.mean(values ** 2))),
        f"{prefix}_median_{unit}": float(np.median(values)),
        f"{prefix}_p95_{unit}": float(np.percentile(values, 95)),
        "n": int(values.size),
    }


def translation_jitter(positions: np.ndarray, frames: np.ndarray, fps: float, window: int = 5) -> dict:
    """Acceleration and high-frequency residual of a (T,3) trajectory."""
    acc, resid = [], []
    half = window // 2
    for run in consecutive_runs(frames):
        p = positions[run]
        if len(p) >= 3:
            acc.append(np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1) * fps * fps)
        if len(p) >= window:
            kernel = np.ones(window) / window
            smooth = np.stack([np.convolve(p[:, k], kernel, mode="valid") for k in range(3)], axis=1)
            resid.append(np.linalg.norm(p[half:len(p) - half] - smooth, axis=1) * 1000.0)
    out = _stats(np.concatenate(acc) if acc else np.zeros(0), "accel", "m_s2")
    r = np.concatenate(resid) if resid else np.zeros(0)
    out["highfreq_rms_mm"] = float(np.sqrt(np.mean(r ** 2))) if r.size else None
    return out


def rotation_jitter(rotations: np.ndarray, frames: np.ndarray, fps: float) -> dict:
    """Angular acceleration (deg/s^2) of a (T,3,3) rotation sequence (world frame)."""
    acc = []
    for run in consecutive_runs(frames):
        R = rotations[run]
        if len(R) < 3:
            continue
        omega = rotation_log(R[1:] @ np.swapaxes(R[:-1], -1, -2))  # world-frame increments
        acc.append(np.degrees(np.linalg.norm(omega[1:] - omega[:-1], axis=1)) * fps * fps)
    return _stats(np.concatenate(acc) if acc else np.zeros(0), "ang_accel", "deg_s2")


def rest_segment_drift(
    world_poses: np.ndarray,
    centroid_obj: np.ndarray,
    rest: np.ndarray,
    frames: np.ndarray,
    min_len: int,
) -> dict:
    """Apparent motion of a resting object over each rest segment."""
    segments = []
    for run in consecutive_runs(frames):
        flags = rest[run]
        starts = np.nonzero(np.diff(np.concatenate([[0], flags.astype(int), [0]])) == 1)[0]
        ends = np.nonzero(np.diff(np.concatenate([[0], flags.astype(int), [0]])) == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= min_len:
                segments.append(run[s:e])
    rows = []
    for seg in segments:
        P = world_poses[seg]
        c = np.einsum("tij,j->ti", P[:, :3, :3], centroid_obj) + P[:, :3, 3]
        med = np.median(c, axis=0)
        R = orthonormalize(P)[:, :3, :3]
        rows.append({
            "frames": [int(frames[seg[0]]), int(frames[seg[-1]])],
            "length": int(len(seg)),
            "endpoint_drift_cm": float(np.linalg.norm(c[-1] - c[0]) * 100.0),
            "max_deviation_cm": float(np.linalg.norm(c - med, axis=1).max() * 100.0),
            "endpoint_rot_drift_deg": float(rotation_angle_deg(R[-1] @ R[0].T)),
        })
    total = sum(r["length"] for r in rows)
    summary = {"segments": rows, "rest_frames": int(total)}
    for key in ("endpoint_drift_cm", "max_deviation_cm", "endpoint_rot_drift_deg"):
        vals = [r[key] for r in rows]
        summary[f"{key}_max"] = float(max(vals)) if vals else None
        summary[f"{key}_mean"] = float(np.average(vals, weights=[r["length"] for r in rows])) if vals else None
    return summary


def contact_runs_from_labels(contacts: list[dict], frames: set[int], min_len: int) -> dict:
    """{(side, tip, object): [frame arrays]} from issue-6 contacts.json rows."""
    hits = defaultdict(list)
    for r in contacts:
        label = r.get("window_contact", r.get("raw_contact"))
        if label is True and r["frame_id"] in frames:
            hits[(r["hand"], r["fingertip"], r["object_id"])].append(int(r["frame_id"]))
    runs = {}
    for key, fs in hits.items():
        fs = np.unique(fs)
        runs[key] = [fs[r] for r in consecutive_runs(fs) if len(r) >= min_len]
    return runs


def contact_precision(tip_obj: dict, runs: dict, penetration_m: float) -> dict:
    """GT-free contact precision from fingertip positions in each object frame.

    ``tip_obj[(side, tip, object)]`` maps frame -> (point, normal, dist, signed).
    """
    spreads, normal_spreads, dists, signed_all, run_count = [], [], [], [], 0
    for key, key_runs in runs.items():
        table = tip_obj.get(key)
        if table is None:
            continue
        for fs in key_runs:
            rows = [table[f] for f in fs if f in table]
            if len(rows) < 2:
                continue
            run_count += 1
            q = np.array([r[0] for r in rows])
            n = np.array([r[1] for r in rows])
            spreads.append(np.sqrt(np.mean(np.sum((q - q.mean(0)) ** 2, axis=1))) * 1000.0)
            mean_n = n.mean(0)
            mean_n /= max(np.linalg.norm(mean_n), 1e-12)
            normal_spreads.append(np.degrees(np.arccos(np.clip(n @ mean_n, -1, 1))).mean())
            dists.extend(r[2] for r in rows)
    for table in tip_obj.values():
        signed_all.extend(r[3] for r in table.values())
    signed_all = np.asarray(signed_all)
    return {
        "contact_runs": run_count,
        "contact_point_spread_rms_mm": float(np.mean(spreads)) if spreads else None,
        "contact_normal_spread_deg": float(np.mean(normal_spreads)) if normal_spreads else None,
        "contact_surface_distance_median_mm": float(np.median(dists) * 1000.0) if dists else None,
        "contact_within_2cm_fraction": float(np.mean(np.asarray(dists) < 0.02)) if dists else None,
        "penetration_rate": float(np.mean(signed_all < -penetration_m)) if signed_all.size else None,
        "penetration_depth_p95_mm": float(np.percentile(np.maximum(-signed_all, 0), 95) * 1000.0) if signed_all.size else None,
    }


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def load_scene(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        scene = {k: z[k] for k in z.files}
    for state in STATES:
        if f"{state}_camera_w2c" not in scene:
            raise ValueError(f"{path} has no {state} state; rerun refine_simple with --scene_export_path")
    return scene


def world_state(scene: dict, state: str) -> dict:
    """Everything in the refinement world frame (reference camera)."""
    frames = scene["frame_indices"]
    c2w = invert_rigid(scene[f"{state}_camera_w2c"])
    out = {"c2w": c2w, "objects": [], "hands": []}
    for i, _ in enumerate(scene["object_names"]):
        o2c = scene[f"{state}_object_{i}_o2c"]
        out["objects"].append({"o2c": o2c, "o2w": c2w @ o2c})
    pos = {int(f): t for t, f in enumerate(frames)}
    for j, _ in enumerate(scene["hand_sides"]):
        hf = scene[f"hand_{j}_frame_indices"]
        t_idx = np.array([pos[int(f)] for f in hf])
        verts_cam = scene[f"{state}_hand_{j}_verts"].astype(np.float64)
        joints_cam = scene[f"{state}_hand_{j}_joints"].astype(np.float64)
        out["hands"].append({
            "frames": hf,
            "t_idx": t_idx,
            "verts_cam": verts_cam,
            "verts_world": apply(c2w[t_idx], verts_cam),
            "joints_world": apply(c2w[t_idx], joints_cam),
        })
    return out


def evaluate(
    scene: dict,
    contacts: list[dict] | None,
    rest_clearance: float,
    min_rest_len: int,
    penetration_m: float,
    surface_samples: int,
) -> dict:
    frames = scene["frame_indices"]
    fps = float(scene["fps"])
    names = [str(n) for n in scene["object_names"]]
    sides = [str(s) for s in scene["hand_sides"]]
    states = {s: world_state(scene, s) for s in STATES}
    surfaces = []
    for i in range(len(names)):
        pts, nrm, spacing = sample_surface(scene[f"object_{i}_verts"].astype(np.float64), scene[f"object_{i}_faces"], surface_samples, seed=i)
        surfaces.append((pts, nrm, pts.mean(0), spacing))

    report: dict = {
        "frames": [int(frames[0]), int(frames[-1])],
        "n_frames": int(len(frames)),
        "fps": fps,
        "objects": names,
        "hands": sides,
        "world": "refinement reference camera (OpenCV axes), metres",
        "camera": {}, "object_jitter": {}, "hand_jitter": {}, "drift": {}, "contact": {},
    }

    # Camera path.
    for s in STATES:
        c2w = states[s]["c2w"]
        report["camera"][s] = {
            "translation": translation_jitter(c2w[:, :3, 3], frames, fps),
            "rotation": rotation_jitter(c2w[:, :3, :3], frames, fps),
        }
    d_t = np.linalg.norm(states["refined"]["c2w"][:, :3, 3] - states["initial"]["c2w"][:, :3, 3], axis=1)
    d_r = rotation_angle_deg(states["refined"]["c2w"][:, :3, :3] @ np.swapaxes(states["initial"]["c2w"][:, :3, :3], -1, -2))
    report["camera"]["correction"] = {
        "translation_mean_cm": float(d_t.mean() * 100), "translation_max_cm": float(d_t.max() * 100),
        "rotation_mean_deg": float(d_r.mean()), "rotation_max_deg": float(d_r.max()),
    }

    # Objects: jitter in world, and rest-segment drift.
    # A frame is "rest" for an object when, in both states, no hand vertex is
    # within rest_clearance of its surface.
    rest = {name: np.ones(len(frames), bool) for name in names}
    for s in STATES:
        st = states[s]
        for i, name in enumerate(names):
            pts = surfaces[i][0][:: max(1, surface_samples // 2000)]
            for hand in st["hands"]:
                o2c_inv = invert_rigid(orthonormalize(st["objects"][i]["o2c"][hand["t_idx"]]))
                scale = np.cbrt(np.linalg.det(st["objects"][i]["o2c"][hand["t_idx"], :3, :3]))
                local = apply(o2c_inv, hand["verts_cam"][:, ::8]) / scale[:, None, None]
                for k, t in enumerate(hand["t_idx"]):
                    lo, hi = pts.min(0) - rest_clearance, pts.max(0) + rest_clearance
                    near = local[k][np.all((local[k] > lo) & (local[k] < hi), axis=1)]
                    if len(near) == 0:
                        continue
                    d = np.sqrt(((near[:, None] - pts[None]) ** 2).sum(-1)).min() * scale[k]
                    if d < rest_clearance:
                        rest[name][t] = False

    for i, name in enumerate(names):
        report["object_jitter"][name] = {}
        report["drift"][name] = {}
        for s in STATES:
            o2w = states[s]["objects"][i]["o2w"]
            centre = apply(o2w, surfaces[i][2][None].repeat(len(frames), 0)[:, None])[:, 0]
            report["object_jitter"][name][s] = {
                "translation": translation_jitter(centre, frames, fps),
                "rotation": rotation_jitter(orthonormalize(o2w)[:, :3, :3], frames, fps),
            }
            report["drift"][name][s] = rest_segment_drift(o2w, surfaces[i][2], rest[name], frames, min_rest_len)

    # Hands: wrist (joint 0) and mean fingertip jitter in world.
    tips = list(MANO_TIP_VERTS.values())
    for j, side in enumerate(sides):
        report["hand_jitter"][side] = {}
        for s in STATES:
            h = states[s]["hands"][j]
            report["hand_jitter"][side][s] = {
                "wrist": translation_jitter(h["joints_world"][:, 0], h["frames"], fps),
                "fingertips": translation_jitter(h["verts_world"][:, tips].mean(1), h["frames"], fps),
            }

    # Contacts: fingertip positions expressed in each object frame.
    frame_set = {int(f) for f in frames}
    tip_tables = {}
    for s in STATES:
        table = {}
        st = states[s]
        for j, side in enumerate(sides):
            h = st["hands"][j]
            for i, name in enumerate(names):
                o2c = st["objects"][i]["o2c"][h["t_idx"]]
                inv = np.linalg.inv(o2c)
                for tip, vid in MANO_TIP_VERTS.items():
                    local = apply(inv, h["verts_cam"][:, vid][:, None])[:, 0]
                    q, n, dist, signed = nearest_surface(local, *surfaces[i][:2], surfaces[i][3])
                    scale = np.cbrt(np.linalg.det(o2c[:, :3, :3]))
                    table[(side, tip, name)] = {
                        int(f): (q[k], n[k], dist[k] * scale[k], signed[k] * scale[k])
                        for k, f in enumerate(h["frames"])
                    }
        tip_tables[s] = table
    if contacts is not None:
        runs = contact_runs_from_labels(contacts, frame_set, min_len=3)
        source = "issue-6 contacts.json (window_contact)"
    else:
        runs = defaultdict(list)
        for key, table in tip_tables["refined"].items():
            fs = np.array(sorted(f for f, r in table.items() if r[2] < 0.02))
            runs[key] = [fs[r] for r in consecutive_runs(fs) if len(r) >= 3]
        source = "refined fingertip within 2 cm"
    report["contact"]["run_source"] = source
    for s in STATES:
        report["contact"][s] = contact_precision(tip_tables[s], runs, penetration_m)
    return report


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def export_scoring_layout(scene: dict, out: Path, episode: int, provenance: dict | None) -> dict[str, Path]:
    """object_to_world JSONs for score_track3_official.py, one tree per state."""
    roots = {}
    names = [str(n) for n in scene["object_names"]]
    ep = f"e{episode:03d}"
    for s in STATES:
        st = world_state(scene, s)
        root = out / "scoring" / s
        for i, name in enumerate(names):
            d = root / ep / "ego_object_poses" / ep / name / "object_to_world"
            d.mkdir(parents=True, exist_ok=True)
            for t, f in enumerate(scene["frame_indices"]):
                M = orthonormalize(st["objects"][i]["o2w"][t])
                status = "observed"
                if provenance is not None and not str(provenance.get(name, {}).get(str(int(f)), "observed")).startswith("observed"):
                    status = "inferred"
                (d / f"{int(f):06d}.json").write_text(json.dumps({
                    "episode": episode, "frame_index": int(f),
                    "rotation": _quat_wxyz(M[:3, :3]), "translation": M[:3, 3].tolist(), "scale": [1.0, 1.0, 1.0],
                    "status": status, "coordinate_frame": "fixed_world", "gt_use": "none",
                    "method": f"gsplat_refine_simple_{s}",
                }))
        roots[s] = root
    return roots


def _quat_wxyz(R: np.ndarray) -> list[float]:
    w = np.sqrt(max(0.0, 1.0 + np.trace(R))) / 2.0
    if w > 1e-6:
        q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i], q[1 + j], q[1 + k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    q = q / np.linalg.norm(q)
    return (q if q[0] >= 0 else -q).tolist()


def run_official_scorer(roots: dict[str, Path], episode: int, data: Path, out: Path) -> dict:
    repo = Path(__file__).resolve().parents[2]
    scores = {}
    for s, root in roots.items():
        report = out / "scoring" / f"official_{s}.json"
        report.unlink(missing_ok=True)  # the scorer refuses to overwrite
        subprocess.run([
            sys.executable, str(repo / "scripts/eval/score_track3_official.py"),
            "--predictions", str(root), "--kind", "object_to_world",
            "--episodes", str(episode), "--data", str(data), "--report", str(report),
        ], check=True)
        payload = json.loads(report.read_text())
        rows = payload.get("episodes", payload.get("rows", []))
        scores[s] = rows[0]["metrics"] if rows else payload.get("mean_metrics")
    return scores


def _fmt(v, digits=2) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def write_markdown(report: dict, path: Path, episode: int) -> None:
    L = [f"# Episode {episode:03d} joint refinement quality report", ""]
    L.append(f"Frames {report['frames'][0]}–{report['frames'][1]} ({report['n_frames']} frames, {report['fps']:g} fps). "
             f"Objects: {', '.join(report['objects'])}. Hands: {', '.join(report['hands']) or 'none'}.")
    L.append("Every number except the official score is GT-free. Lower is better unless noted.")
    L.append("")
    L.append("## Jitter")
    L.append("")
    L.append("| Track | Acceleration RMS (m/s²) initial → refined | High-frequency residual (mm) initial → refined | Angular acceleration RMS (deg/s²) initial → refined |")
    L.append("|---|---|---|---|")

    def row(label, ini, ref, rot_ini=None, rot_ref=None):
        a = f"{_fmt(ini['accel_rms_m_s2'])} → {_fmt(ref['accel_rms_m_s2'])}"
        h = f"{_fmt(ini['highfreq_rms_mm'])} → {_fmt(ref['highfreq_rms_mm'])}"
        r = "—" if rot_ini is None else f"{_fmt(rot_ini['ang_accel_rms_deg_s2'], 0)} → {_fmt(rot_ref['ang_accel_rms_deg_s2'], 0)}"
        L.append(f"| {label} | {a} | {h} | {r} |")

    cam = report["camera"]
    row("Camera", cam["initial"]["translation"], cam["refined"]["translation"], cam["initial"]["rotation"], cam["refined"]["rotation"])
    for name, v in report["object_jitter"].items():
        row(f"Object `{name}`", v["initial"]["translation"], v["refined"]["translation"], v["initial"]["rotation"], v["refined"]["rotation"])
    for side, v in report["hand_jitter"].items():
        row(f"{side.title()} wrist", v["initial"]["wrist"], v["refined"]["wrist"])
        row(f"{side.title()} fingertips", v["initial"]["fingertips"], v["refined"]["fingertips"])
    L.append("")
    c = cam["correction"]
    L.append(f"Camera correction applied by refinement: mean {c['translation_mean_cm']:.2f} cm / {c['rotation_mean_deg']:.2f}°, "
             f"max {c['translation_max_cm']:.2f} cm / {c['rotation_max_deg']:.2f}°.")
    L.append("")
    L.append("## Drift (resting objects must stay fixed in the world)")
    L.append("")
    L.append("| Object | Rest frames | Endpoint drift (cm) initial → refined | Max deviation (cm) initial → refined | Rotation drift (deg) initial → refined |")
    L.append("|---|---|---|---|---|")
    for name, v in report["drift"].items():
        i, r = v["initial"], v["refined"]
        L.append(f"| `{name}` | {r['rest_frames']} | {_fmt(i['endpoint_drift_cm_mean'])} → {_fmt(r['endpoint_drift_cm_mean'])} "
                 f"| {_fmt(i['max_deviation_cm_max'])} → {_fmt(r['max_deviation_cm_max'])} "
                 f"| {_fmt(i['endpoint_rot_drift_deg_mean'])} → {_fmt(r['endpoint_rot_drift_deg_mean'])} |")
    L.append("")
    L.append("Endpoint drift and rotation drift are length-weighted means over rest segments; max deviation is the worst segment.")
    L.append("")
    L.append(f"## Contact precision (runs from {report['contact']['run_source']})")
    L.append("")
    L.append("| Measure | Initial | Refined |")
    L.append("|---|---|---|")
    ci, cr = report["contact"]["initial"], report["contact"]["refined"]
    for key, label, d in (
        ("contact_runs", "Contact runs", 0),
        ("contact_point_spread_rms_mm", "Touch-point spread on object (mm RMS)", 1),
        ("contact_normal_spread_deg", "Touch normal spread (deg)", 1),
        ("contact_surface_distance_median_mm", "Fingertip–surface distance in contact, median (mm)", 1),
        ("contact_within_2cm_fraction", "Labelled contact frames within 2 cm (higher is better)", 3),
        ("penetration_rate", "Fingertip penetration rate (> 5 mm)", 3),
        ("penetration_depth_p95_mm", "Penetration depth p95 (mm)", 1),
    ):
        L.append(f"| {label} | {_fmt(ci[key], d)} | {_fmt(cr[key], d)} |")
    if report.get("official"):
        L.append("")
        L.append("## Official reconstruction diagnostic (uses public reference poses)")
        L.append("")
        L.append("| Metric | Initial | Refined |")
        L.append("|---|---|---|")
        for k in ("add_auc", "maniptrans_sr", "spider_sr", "rpe_cm", "mppe_cm"):
            L.append(f"| {k} | {_fmt(report['official']['initial'].get(k), 3)} | {_fmt(report['official']['refined'].get(k), 3)} |")
    path.write_text("\n".join(L) + "\n")


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _quantize(points: np.ndarray, step: float) -> np.ndarray:
    return np.clip(np.round(points / step), -32767, 32767).astype(np.int16)


def decimate(verts: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering decimation for display only."""
    if len(faces) <= max_faces:
        return verts, faces
    extent = float(np.ptp(verts, axis=0).max())
    cell = extent / 256.0
    while True:
        keys = np.floor(verts / cell).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        f = inverse[faces]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        f = np.unique(np.sort(f, axis=1), axis=0) if len(f) else f
        if len(f) <= max_faces or cell > extent:
            break
        cell *= 1.4
    n = inverse.max() + 1
    v = np.zeros((n, 3))
    np.add.at(v, inverse, verts)
    v /= np.bincount(inverse, minlength=n)[:, None]
    # Recompute faces with original winding after clustering.
    f = inverse[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return v, f


def write_viewer(scene: dict, path: Path, title: str, max_object_faces: int = 20000) -> None:
    step = 2e-4  # metres per int16 unit for streamed geometry
    frames = scene["frame_indices"]
    states = {s: world_state(scene, s) for s in STATES}
    payload = {
        "title": title,
        "fps": float(scene["fps"]),
        "frames": [int(f) for f in frames],
        "step": step,
        "cameras": {s: _b64(states[s]["c2w"].astype(np.float32)) for s in STATES},
        "objects": [],
        "hands": [],
        "background": None,
    }
    if "background_points" in scene:
        payload["background"] = {
            "pos": _b64(_quantize(scene["background_points"], step)),
            "col": _b64((np.clip(scene["background_colors"], 0, 1) * 255).astype(np.uint8)),
        }
    palette = ["#e4572e", "#2e86ab", "#f3a712", "#8cb369"]
    for i, name in enumerate(scene["object_names"]):
        v, f = decimate(scene[f"object_{i}_verts"].astype(np.float64), scene[f"object_{i}_faces"], max_object_faces)
        payload["objects"].append({
            "name": str(name),
            "color": palette[i % len(palette)],
            "verts": _b64(v.astype(np.float32)),
            "faces": _b64(f.astype(np.uint32)),
            "poses": {s: _b64(states[s]["objects"][i]["o2w"].astype(np.float32)) for s in STATES},
        })
    for j, side in enumerate(scene["hand_sides"]):
        payload["hands"].append({
            "side": str(side),
            "frames": [int(f) for f in scene[f"hand_{j}_frame_indices"]],
            "faces": _b64(scene[f"hand_{j}_faces"].astype(np.uint32)),
            "verts": {s: _b64(_quantize(states[s]["hands"][j]["verts_world"], step)) for s in STATES},
        })
    html = VIEWER_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", json.dumps(payload))
    path.write_text(html)


VIEWER_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; --bg:#14161a; --panel:#1f2329; --text:#e8eaed; --muted:#9aa0a6; --accent:#7cc4fa; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--text); font:13px/1.4 system-ui,sans-serif; }
  #view { position:fixed; inset:0; }
  #panel { position:fixed; left:12px; right:12px; bottom:12px; background:var(--panel); border-radius:8px;
           padding:10px 14px; display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; }
  #panel h1 { font-size:13px; margin:0; font-weight:600; }
  #slider { flex:1 1 240px; accent-color:var(--accent); }
  label { color:var(--muted); white-space:nowrap; } button { background:#2b3138; color:var(--text); border:1px solid #3c444d;
  border-radius:6px; padding:4px 10px; cursor:pointer; } #frame { font-variant-numeric:tabular-nums; min-width:9ch; }
  .legend { color:var(--muted); }
</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<div id="view"></div>
<div id="panel">
  <h1>__TITLE__</h1>
  <button id="play">Pause</button>
  <input id="slider" type="range" min="0" value="0">
  <span id="frame"></span>
  <label><input type="checkbox" id="refined" checked> Refined</label>
  <label><input type="checkbox" id="initial"> Initial (ghost)</label>
  <label><input type="checkbox" id="bg" checked> Background</label>
  <label><input type="checkbox" id="cam" checked> Camera</label>
  <span class="legend">Drag to orbit · scroll to zoom</span>
</div>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const D = __DATA__;
const dec = (b64, T) => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); };
const T = D.frames.length;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio); document.getElementById("view").appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x14161a);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 50);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(1, 2, 1); scene.add(sun);
// World is the OpenCV reference camera (y down, z forward); flip to y up.
const world = new THREE.Group(); world.rotation.x = Math.PI; scene.add(world);
const STATES = ["initial", "refined"];
const layers = { initial: new THREE.Group(), refined: new THREE.Group() };
world.add(layers.initial, layers.refined);
const toMat = (arr, t) => new THREE.Matrix4().fromArray(arr.subarray(16 * t, 16 * t + 16)).transpose();

let bg = null;
if (D.background) {
  const q = dec(D.background.pos, Int16Array), c = dec(D.background.col, Uint8Array);
  const pos = new Float32Array(q.length); for (let i = 0; i < q.length; i++) pos[i] = q[i] * D.step;
  const col = new Float32Array(c.length); for (let i = 0; i < c.length; i++) col[i] = c[i] / 255;
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  bg = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.006, vertexColors: true })); world.add(bg);
}
const objects = D.objects.map(o => {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(dec(o.verts, Float32Array), 3));
  g.setIndex(new THREE.BufferAttribute(dec(o.faces, Uint32Array), 1)); g.computeVertexNormals();
  const meshes = {}, poses = {};
  for (const s of STATES) {
    const ghost = s === "initial";
    const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: o.color, transparent: ghost, opacity: ghost ? 0.35 : 1, side: THREE.DoubleSide, roughness: 0.6 }));
    m.matrixAutoUpdate = false; layers[s].add(m); meshes[s] = m; poses[s] = dec(o.poses[s], Float32Array);
  }
  return { meshes, poses };
});
const hands = D.hands.map(h => {
  const pos = {}, meshes = {}, map = new Map(h.frames.map((f, i) => [f, i]));
  const faces = dec(h.faces, Uint32Array);
  for (const s of STATES) {
    pos[s] = dec(h.verts[s], Int16Array);
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(778 * 3), 3)); g.setIndex(new THREE.BufferAttribute(faces, 1));
    const ghost = s === "initial";
    const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: h.side === "left" ? 0xf2b5a0 : 0xe8a28a, transparent: ghost, opacity: ghost ? 0.35 : 1, side: THREE.DoubleSide }));
    m.frustumCulled = false; layers[s].add(m); meshes[s] = m;
  }
  return { pos, meshes, map };
});
const cams = {}, paths = {};
for (const s of STATES) {
  const arr = dec(D.cameras[s], Float32Array);
  const pts = []; for (let t = 0; t < T; t++) pts.push(new THREE.Vector3(arr[16 * t + 3], arr[16 * t + 7], arr[16 * t + 11]));
  const color = s === "refined" ? 0x7cc4fa : 0x888888;
  paths[s] = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color }));
  layers[s].add(paths[s]);
  const z = 0.08, x = 0.05, y = 0.035, V = (a, b, c) => new THREE.Vector3(a, b, c);
  const o = V(0, 0, 0), corners = [V(x, y, z), V(-x, y, z), V(-x, -y, z), V(x, -y, z)];
  const segs = [];
  for (let k = 0; k < 4; k++) segs.push(o, corners[k], corners[k], corners[(k + 1) % 4]);
  const fr = new THREE.BufferGeometry().setFromPoints(segs);
  const frustum = new THREE.LineSegments(fr, new THREE.LineBasicMaterial({ color })); frustum.matrixAutoUpdate = false;
  layers[s].add(frustum); cams[s] = { arr, frustum };
}
function show(t) {
  for (const s of STATES) {
    for (const o of objects) o.meshes[s].matrix.copy(toMat(o.poses[s], t));
    for (const h of hands) {
      const i = h.map.get(D.frames[t]), m = h.meshes[s]; m.visible = i !== undefined && layers[s].visible;
      if (i === undefined) continue;
      const src = h.pos[s], dst = m.geometry.attributes.position.array, off = i * 778 * 3;
      for (let k = 0; k < 778 * 3; k++) dst[k] = src[off + k] * D.step;
      m.geometry.attributes.position.needsUpdate = true; m.geometry.computeVertexNormals();
    }
    cams[s].frustum.matrix.copy(toMat(cams[s].arr, t));
  }
  document.getElementById("frame").textContent = `frame ${D.frames[t]}`;
}
const slider = document.getElementById("slider"); slider.max = T - 1;
let t = 0, playing = true, last = 0;
slider.oninput = () => { t = +slider.value; show(t); };
document.getElementById("play").onclick = e => { playing = !playing; e.target.textContent = playing ? "Pause" : "Play"; };
const bind = (id, fn) => { const el = document.getElementById(id); el.onchange = () => { fn(el.checked); show(t); }; fn(el.checked); };
bind("refined", v => layers.refined.visible = v); bind("initial", v => layers.initial.visible = v);
bind("bg", v => bg && (bg.visible = v)); bind("cam", v => { for (const s of STATES) { cams[s].frustum.visible = v; paths[s].visible = v; } });
// Frame the first object.
const box = new THREE.Box3(); for (const o of objects) { o.meshes.refined.matrix.copy(toMat(o.poses.refined, 0)); o.meshes.refined.updateMatrixWorld(true); box.expandByObject(o.meshes.refined); }
world.updateMatrixWorld(true); const c = box.isEmpty() ? new THREE.Vector3(0, 0, -0.6) : box.getCenter(new THREE.Vector3());
controls.target.copy(c); camera.position.copy(c).add(new THREE.Vector3(0.5, 0.45, 0.7));
function resize() { renderer.setSize(innerWidth, innerHeight); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize(); show(0);
renderer.setAnimationLoop(now => {
  if (playing && now - last > 1000 / D.fps) { last = now; t = (t + 1) % T; slider.value = t; show(t); }
  controls.update(); renderer.render(scene, camera);
});
</script></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, required=True, help="refine_simple --scene_export_path archive")
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--inputs", type=Path, default=None, help="gsplat_inputs.py inputs.json (pose provenance)")
    p.add_argument("--contacts", type=Path, default=None, help="issue-6 contacts/contacts.json")
    p.add_argument("--data", type=Path, default=None, help="public dataset root; runs the official scorer")
    p.add_argument("--rest-clearance", type=float, default=0.05, help="metres between hand and object for 'rest'")
    p.add_argument("--min-rest-len", type=int, default=10)
    p.add_argument("--penetration-m", type=float, default=0.005)
    p.add_argument("--surface-samples", type=int, default=20000)
    p.add_argument("--no-viewer", action="store_true")
    args = p.parse_args()

    scene = load_scene(args.scene)
    contacts = json.loads(args.contacts.read_text()) if args.contacts else None
    report = evaluate(scene, contacts, args.rest_clearance, args.min_rest_len, args.penetration_m, args.surface_samples)
    report["episode"] = args.episode
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(args.inputs.read_text())["object_provenance"] if args.inputs else None
    roots = export_scoring_layout(scene, args.out, args.episode, provenance)
    if args.data is not None:
        report["official"] = run_official_scorer(roots, args.episode, args.data, args.out)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    write_markdown(report, args.out / "report.md", args.episode)
    if not args.no_viewer:
        write_viewer(scene, args.out / "viewer.html", f"Episode {args.episode:03d} — joint refinement")
    print((args.out / "report.md").read_text())


if __name__ == "__main__":
    main()
