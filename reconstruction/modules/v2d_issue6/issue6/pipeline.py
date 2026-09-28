"""Restartable stages with immutable input and output receipts."""

from __future__ import annotations

import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from . import backend
from .geometry import (
    bridge_contacts,
    decode_depth,
    reproject_depth,
    surface_contacts,
)
from .manifest import (
    camera_from_object,
    digest,
    file_hash,
    fingerprint,
    inspect_inputs,
    read_json,
    real_file,
    source_state,
    write_json,
)

TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "little": 20}
DEPENDENCIES = {
    "validate": [],
    "hands": [],
    "depth": [],
    "align": ["hands", "depth"],
    "contacts": ["align"],
    "report": ["contacts", "align"],
}


def output_hashes(folder):
    return {
        str(p.relative_to(folder)): file_hash(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "receipt.json"
    }


def check_stage(root, stage, config_hash):
    folder = root / stage
    receipt = read_json(folder / "receipt.json")
    if (
        receipt["status"] != "complete"
        or receipt["config_sha256"] != config_hash
        or receipt["outputs"] != output_hashes(folder)
    ):
        raise ValueError(f"Stale, modified or incomplete {stage} outputs; use a fresh run directory")


def execute(m, root, stage):
    root = Path(root).resolve()
    if stage not in DEPENDENCIES:
        raise ValueError("Unknown stage")
    root.mkdir(parents=True, exist_ok=True)
    inspect_inputs(m)
    hashes = fingerprint(m)
    implementation = {
        str(p.relative_to(backend.MODULES)): file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    # The worker deliberately reuses these pinned implementations.
    for rel in (
        "v2d_hamer/lib/align_hands.py",
        "v2d_hamer/lib/render_hands_aligned_video.py",
        "v2d_hamer/lib/masks_to_hands.py",
        "v2d_hand_alignment/lib/dynhamr_to_hamer_tracks.py",
        "v2d_pipelines/run_hand_masks.py",
    ):
        p = backend.MODULES / rel
        if p.exists():
            implementation[rel] = file_hash(p)
    lock = {"manifest": m, "inputs": hashes, "implementation": implementation}
    lock_path = root / "inputs.lock.json"
    if lock_path.exists():
        if read_json(lock_path) != lock:
            raise ValueError(
                "Inputs/settings/code changed; start a new run (import prior hand records to reuse inference)"
            )
    else:
        write_json(lock_path, lock)
    config_hash = digest(lock)
    for prerequisite in DEPENDENCIES[stage]:
        check_stage(root, prerequisite, config_hash)
    out = root / stage
    if out.exists():
        raise ValueError(f"Stage directory already exists: {out}; use a fresh run directory")
    out.mkdir()
    start = time.time()
    receipt = {
        "stage": stage,
        "config_sha256": config_hash,
        "status": "running",
        "source": source_state(backend.MODULES),
        "started_unix_s": start,
        "mode": m["mode"],
        "camera_provenance": m["camera_provenance"],
        "hand_provenance": m["hand_provenance"],
        "settings": m.get("settings", {}),
    }
    write_json(out / "receipt.json", receipt)
    try:
        {
            "validate": validate_stage,
            "hands": hands_stage,
            "depth": depth_stage,
            "align": align_stage,
            "contacts": contacts_stage,
            "report": report_stage,
        }[stage](m, root, out)
        receipt["status"] = "complete"
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_s"] = time.time() - start
        receipt["outputs"] = output_hashes(out)
        write_json(out / "receipt.json", receipt)
    return out


def validate_stage(m, root, out):
    write_json(out / "readiness.json", inspect_inputs(m))


def hands_stage(m, root, out):
    backend.reconstruct(m, out)
    write_json(
        out / "frame_map.json",
        [{"inference_index": n, **{k: f[k] for k in ("id", "timestamp_s")}} for n, f in enumerate(m["frames"])],
    )


def read_array(path):
    p = real_file(path)
    return np.load(p, allow_pickle=False) if p.suffix == ".npy" else np.array(Image.open(p))


def depth_stage(m, root, out):
    target = m["color_camera"]
    stats = []
    for f in m["frames"]:
        d = m.get("depth")
        if d is None or not f.get("depth_path"):
            z = np.full((target["height"], target["width"]), np.nan)
        else:
            validity = read_array(f["depth_validity"]) > 0 if f.get("depth_validity") else None
            native = decode_depth(read_array(f["depth_path"]), d["encoding"], validity)
            z = reproject_depth(native, d["camera"], target, d["color_from_depth"])
        np.save(out / f"{f['id']:06d}.npy", z.astype(np.float32))
        stats.append(
            {
                "frame_id": f["id"],
                "valid_pixels": int(np.isfinite(z).sum()),
                "coverage": float(np.isfinite(z).mean()),
            }
        )
    write_json(out / "coverage.json", stats)


def align_stage(m, root, out):
    h, w = m["color_camera"]["height"], m["color_camera"]["width"]
    frames = []
    mesh_cache = {}
    for f in m["frames"]:
        mask = np.zeros((h, w), bool)
        known = True
        projected_objects = []
        for obj in m["objects"]:
            state = f.get("objects", {}).get(obj["id"], {})
            if state.get("mask"):
                a = read_array(state["mask"]) > 0
                if a.shape != mask.shape:
                    raise ValueError("Object mask is not in the color-camera pixel grid")
                mask |= a
            elif state.get("occlusion_clear") is not True:
                object_pose = camera_from_object(f, obj)
                if object_pose is None:
                    known = False
                else:
                    # A valid CAD pose can supply the color-view occlusion mask
                    # when #5 only exports masks in the stereo-camera views.
                    if obj["id"] not in mesh_cache:
                        mesh = load_mesh(obj["mesh"])
                        name = f"object_{len(mesh_cache)}.npz"
                        np.savez_compressed(out / name, vertices=mesh.vertices, faces=mesh.faces)
                        mesh_cache[obj["id"]] = name
                    projected_objects.append(
                        {
                            "mesh": mesh_cache[obj["id"]],
                            "camera_from_object": object_pose.tolist(),
                        }
                    )
        np.save(out / f"occlusion_{f['id']:06d}.npy", mask)
        frames.append(
            {
                "id": f["id"],
                "timestamp_s": f["timestamp_s"],
                "occlusion_known": known,
                "projected_objects": projected_objects,
            }
        )
        target = out / "frames" / f"{f['id']:06d}.png"
        target.parent.mkdir(exist_ok=True)
        with Image.open(real_file(f["color"])) as im:
            im.convert("RGB").save(target)
    write_json(
        out / "job.json",
        {
            "camera": m["color_camera"],
            "frames": frames,
            "minimum_pixels": m.get("settings", {}).get("minimum_pixels", 256),
            "mode": m["mode"],
            "camera_provenance": m["camera_provenance"],
            "hand_provenance": m["hand_provenance"],
        },
    )
    backend.align_worker(m, root, out)
    if not (out / "hands.json").is_file() or not (out / "hands.npz").is_file():
        raise RuntimeError("Alignment worker did not produce required hand artifacts")


def load_mesh(path):
    mesh = trimesh.load(real_file(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("Object asset needs a triangle mesh in metres")
    if not np.isfinite(mesh.vertices).all() or np.any(mesh.area_faces <= 1e-15):
        raise ValueError("Nonfinite or degenerate object mesh")
    return mesh


def load_hands(path, m):
    records = read_json(path)
    expected = {f["id"]: f["timestamp_s"] for f in m["frames"]}
    grouped = defaultdict(list)
    for r in records:
        idx = r["frame_id"]
        if idx not in expected or r["timestamp_s"] != expected[idx]:
            raise ValueError("Hand records do not match manifest frame/time axes")
        if not isinstance(r["is_right"], bool):
            raise ValueError("Invalid handedness")
        joints = np.asarray(r["joints_camera_m"], float)
        if joints.shape != (21, 3) or not np.isfinite(joints).all():
            raise ValueError("Expected finite 21-joint hand geometry")
        grouped[(idx, r["is_right"])].append(r)
    return grouped


def contacts_stage(m, root, out):
    hands = load_hands(root / "align" / "hands.json", m)
    settings = m.get("settings", {})
    threshold = settings.get("contact_threshold_m", 0.02)
    maximum = settings.get("bridge_frames", 15)
    if not np.isfinite(threshold) or threshold <= 0 or not isinstance(maximum, int) or maximum < 0:
        raise ValueError("Invalid contact settings")
    meshes = {o["id"]: load_mesh(o["mesh"]) for o in m["objects"]}
    records, series = [], defaultdict(list)
    for frame in m["frames"]:
        for right in (False, True):
            candidates = hands.get((frame["id"], right), [])
            hand = candidates[0] if len(candidates) == 1 else None
            for obj in m["objects"]:
                pose = camera_from_object(frame, obj)
                valid = hand is not None and hand["metric_valid"] is True and pose is not None
                nearest = distance = normal = inside = None
                if valid:
                    joints = np.array(hand["joints_camera_m"])[list(TIPS.values())]
                    nearest, distance, normal, inside = surface_contacts(joints, meshes[obj["id"]], pose)
                for n, tip in enumerate(TIPS):
                    reason = (
                        None
                        if valid
                        else "ambiguous_hand"
                        if len(candidates) > 1
                        else "missing_hand"
                        if hand is None
                        else "unanchored_hand"
                        if not hand["metric_valid"]
                        else "invalid_object_pose"
                    )
                    record = {
                        "frame_id": frame["id"],
                        "timestamp_s": frame["timestamp_s"],
                        "mode": m["mode"],
                        "camera_provenance": m["camera_provenance"],
                        "object_provenance": obj["provenance"],
                        "hand_provenance": m["hand_provenance"],
                        "hand": "right" if right else "left",
                        "fingertip": tip,
                        "object_id": obj["id"],
                        "valid": bool(valid),
                        "reason": reason,
                        "raw_contact": bool(distance[n] < threshold) if valid else None,
                        "surface_distance_m": float(distance[n]) if valid else None,
                        "point_object_m": nearest[n].tolist() if valid else None,
                        "normal_object": normal[n].tolist() if valid else None,
                        "penetration": inside[n] if valid else None,
                        "geometry_ambiguous": not meshes[obj["id"]].is_volume,
                    }
                    records.append(record)
                    series[(right, obj["id"], tip)].append(record)
    for group in series.values():
        values = bridge_contacts([r["raw_contact"] for r in group], [r["frame_id"] for r in group], maximum)
        for r, value in zip(group, values):
            r["window_contact"] = value
            r["bridged"] = value is True and r["raw_contact"] is False
    write_json(out / "contacts.json", records)
    # Majority summary is deliberately secondary; ties stay explicit.
    dominant = []
    for f in m["frames"]:
        for side in ("left", "right"):
            rows = [r for r in records if r["frame_id"] == f["id"] and r["hand"] == side]
            votes = {
                o["id"]: sum(r["raw_contact"] is True for r in rows if r["object_id"] == o["id"]) for o in m["objects"]
            }
            best = max(votes.values(), default=0)
            winners = [key for key, n in votes.items() if n == best and n > 0]
            dominant.append(
                {
                    "frame_id": f["id"],
                    "hand": side,
                    "object_ids": winners,
                    "complete_observation": bool(rows) and all(r["valid"] for r in rows),
                }
            )
    write_json(out / "dominant_objects.json", dominant)
    times = np.array([f["timestamp_s"] for f in m["frames"]])
    write_json(
        out / "settings.json",
        {
            "threshold_m": threshold,
            "bridge_frames": maximum,
            "bridge_nominal_seconds": float(np.median(np.diff(times)) * maximum) if len(times) > 1 else None,
            "mode": m["mode"],
            "camera_provenance": m["camera_provenance"],
            "contact_precision": None,
        },
    )


def report_stage(m, root, out):
    records = read_json(root / "contacts" / "contacts.json")
    hands = load_hands(root / "align" / "hands.json", m)
    flat = [r for group in hands.values() for r in group]
    timestamps = {f["id"]: f["timestamp_s"] for f in m["frames"]}
    transitions, jumps, switches, residuals = [], [], [], []
    by_track = defaultdict(list)
    for r in flat:
        by_track[r["track_id"]].append(r)
        if r["metric_valid"]:
            residuals.append(r["diagnostics"]["median_abs_residual_m"])
    for track, rows in by_track.items():
        rows.sort(key=lambda r: r["frame_id"])
        for a, b in zip(rows, rows[1:]):
            if b["frame_id"] != a["frame_id"] + 1:
                continue
            if a["is_right"] != b["is_right"]:
                switches.append({"track_id": track, "frame_id": b["frame_id"]})
            if a["metric_valid"] and b["metric_valid"]:
                displacement = float(
                    np.linalg.norm(np.array(b["joints_camera_m"])[0] - np.array(a["joints_camera_m"])[0])
                )
                jumps.append(
                    {
                        "track_id": track,
                        "frame_id": b["frame_id"],
                        "wrist_displacement_camera_m": displacement,
                        "wrist_speed_camera_m_s": displacement
                        / (timestamps[b["frame_id"]] - timestamps[a["frame_id"]]),
                    }
                )
    last = {}
    for r in records:
        key = (r["hand"], r["fingertip"], r["object_id"])
        a = last.get(key)
        if (
            a is not None
            and a["raw_contact"] is not None
            and r["raw_contact"] is not None
            and a["raw_contact"] != r["raw_contact"]
        ):
            transitions.append(
                {
                    k: r[k]
                    for k in (
                        "frame_id",
                        "hand",
                        "fingertip",
                        "object_id",
                        "raw_contact",
                    )
                }
            )
        last[key] = r
    reference = {k: v for k, v in m.items() if k not in ("hand_source", "hand_provenance")}
    report = {
        "episode": m["episode"],
        "mode": m["mode"],
        "camera_provenance": m["camera_provenance"],
        "comparison_sha256": digest({"manifest": reference, "inputs": fingerprint(reference)}),
        "frame_axis": [{"id": f["id"], "timestamp_s": f["timestamp_s"]} for f in m["frames"]],
        "frame_count": len(m["frames"]),
        "observed_hand_slots": len(hands),
        "expected_hand_slots": 2 * len(m["frames"]),
        "metric_valid_hand_records": sum(r["metric_valid"] for r in flat),
        "contact_valid_count": sum(r["valid"] for r in records),
        "contact_unknown_count": sum(not r["valid"] for r in records),
        "raw_contact_count": sum(r["raw_contact"] is True for r in records),
        "bridged_count": sum(r["bridged"] for r in records),
        "alignment_median_residual_m": float(np.median(residuals)) if residuals else None,
        "handedness_switches": switches,
        "wrist_motion_camera_frame": jumps,
        "contact_transitions": transitions,
        "contact_precision": None,
        "reprojection_error_px": None,
        "visual_review_complete": False,
        "issue6_complete": False,
        "notes": [
            "Camera-frame motion includes camera motion; it is not world-frame jitter.",
            "Reprojection accuracy and contact precision require independent labels.",
            "Review all overlays; automatic report creation does not approve an episode.",
        ],
    }
    write_json(out / "report.json", report)
    c = m["color_camera"]
    overlays = out / "overlays"
    overlays.mkdir()
    for f in m["frames"]:
        im = Image.open(real_file(f["color"])).convert("RGB")
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, im.width, 42), fill="black")
        draw.text((4, 4), f"ep {m['episode']} frame {f['id']} | {m['mode']}", fill="white")
        draw.text((4, 22), "red=raw touch yellow=bridged green=far gray=unknown", fill="white")
        for right in (False, True):
            hs = hands.get((f["id"], right), [])
            if len(hs) != 1:
                continue
            h = hs[0]
            joints = np.asarray(h["joints_camera_m"])
            for tip, idx in TIPS.items():
                p = joints[idx]
                if p[2] <= 0:
                    continue
                u, v = c["fx"] * p[0] / p[2] + c["cx"], c["fy"] * p[1] / p[2] + c["cy"]
                rows = [
                    r
                    for r in records
                    if r["frame_id"] == f["id"]
                    and r["fingertip"] == tip
                    and r["hand"] == ("right" if right else "left")
                ]
                color = (
                    "red"
                    if any(r["raw_contact"] is True for r in rows)
                    else "yellow"
                    if any(r["bridged"] for r in rows)
                    else "lime"
                    if rows and all(r["valid"] for r in rows)
                    else "gray"
                )
                if 0 <= u < im.width and 0 <= v < im.height:
                    draw.ellipse((u - 4, v - 4, u + 4, v + 4), fill=color)
                    draw.text((u + 5, v), ("R" if right else "L") + ":" + tip, fill=color)
        im.save(overlays / f"{f['id']:06d}.png")
    # Browser contact sheet preserves exact original frame IDs and timestamps.
    items = "\n".join(
        f'<figure><img loading="lazy" src="overlays/{f["id"]:06d}.png">'
        f"<figcaption>frame {f['id']}, {f['timestamp_s']:.6f}s</figcaption></figure>"
        for f in m["frames"]
    )
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Issue 6 review</title>'
        "<style>body{background:#171717;color:white;font:16px sans-serif}"
        "figure{display:inline-block;width:46%;margin:1%}img{width:100%}</style>"
        "<h1>Hand/contact diagnostics — visual review pending</h1>" + items,
        encoding="utf-8",
    )
    report["contact_video_status"] = "ffmpeg_unavailable"
    if shutil.which("ffmpeg"):
        try:
            rate = backend.fps(m)
        except ValueError:
            report["contact_video_status"] = "irregular_or_single_frame_use_timestamped_gallery"
        else:
            backend.command(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-framerate",
                    rate,
                    "-start_number",
                    m["frames"][0]["id"],
                    "-i",
                    overlays / "%06d.png",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    out / "contact_overlay.mp4",
                ],
                out,
            )
            report["contact_video_status"] = "encoded"
    write_json(out / "report.json", report)


def compare_reports(first, second):
    a, b = read_json(first), read_json(second)
    if a["comparison_sha256"] != b["comparison_sha256"] or a["frame_axis"] != b["frame_axis"]:
        raise ValueError("Comparison needs matching frames, calibration, object inputs and settings")
    keys = (
        "observed_hand_slots",
        "metric_valid_hand_records",
        "alignment_median_residual_m",
        "contact_unknown_count",
        "bridged_count",
    )
    return {
        "first": {k: a[k] for k in keys},
        "second": {k: b[k] for k in keys},
        "note": "Diagnostic comparison only; no accuracy claim without independent labels.",
    }
