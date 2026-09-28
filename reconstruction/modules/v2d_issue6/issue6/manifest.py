"""Explicit frame/pose contracts and content-addressed stage receipts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import camera, rigid

PATH_KEYS = {
    "color",
    "depth_path",
    "depth_validity",
    "mask",
    "pose_path",
    "mesh",
    "records_dir",
    "masks_dir",
    "tracks_path",
    "mano_assets_root",
    "weights_dir",
    "sam2_weights",
    "dynhamr_weights",
    "world_results",
    "camera_pose_path",
}
PROVENANCE = {"predicted", "ground_truth_assisted", "synthetic"}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def real_file(path):
    p = Path(path)
    if not p.is_file() or not p.stat().st_size:
        raise ValueError(f"Missing or empty input: {p}")
    with p.open("rb") as f:
        if f.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Git LFS pointer, not a materialized asset: {p}")
    return p


def load(path, limit=None):
    path = Path(path).resolve()
    m = read_json(path)

    def resolve(obj):
        if isinstance(obj, list):
            return [resolve(v) for v in obj]
        if not isinstance(obj, dict):
            return obj
        out = {}
        for key, value in obj.items():
            if key in PATH_KEYS and value is not None:
                out[key] = str((path.parent / value).resolve())
            elif key == "model_files":
                out[key] = [str((path.parent / v).resolve()) for v in value]
            else:
                out[key] = resolve(value)
        return out

    m = resolve(m)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        m["frames"] = m["frames"][:limit]
    validate(m)
    return m


def pose(record):
    """Explicit directions; supports #3/#5 Transform3d JSON (wxyz)."""
    if "matrix" in record and "pose_path" in record:
        raise ValueError("Provide exactly one pose representation")
    if "matrix" in record:
        return rigid(record["matrix"])
    data = read_json(real_file(record["pose_path"]))
    if not np.allclose(np.asarray(data.get("scale", 1), float), 1):
        raise ValueError("Pose scale must be one; use the original metric CAD mesh")
    q = np.asarray(data["rotation"], float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-4):
        raise ValueError("Transform3d rotation must be normalized wxyz")
    w, x, y, z = q
    t = np.eye(4)
    t[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    t[:3, 3] = data["translation"]
    return rigid(t)


def camera_from_object(frame, obj):
    state = frame.get("objects", {}).get(obj["id"], {})
    if state.get("valid") is not True:
        return None
    p = pose(state)
    if state["direction"] == "camera_from_object":
        return p
    if state["direction"] != "world_from_object":
        raise ValueError("Object pose direction must be camera_from_object or world_from_object")
    if frame.get("camera_valid") is not True:
        return None
    c = (
        rigid(frame["camera_from_world"])
        if "camera_from_world" in frame
        else pose({"pose_path": frame["camera_pose_path"]})
    )
    return c @ p


def validate(m):
    if m.get("schema_version") != 1 or m.get("units") != "metres":
        raise ValueError("Expected schema_version=1 and units=metres")
    if m.get("mode") not in {
        "gt_assisted_development",
        "predicted_object_integration",
        "synthetic",
    }:
        raise ValueError("Declare a supported provenance mode")
    if not isinstance(m.get("episode"), int) or m["episode"] < 0:
        raise ValueError("Episode must be a nonnegative integer")
    if m["episode"] in (0, 11, 23, 31, 39) and m["mode"] != "synthetic":
        raise ValueError("Proxy holdout is excluded from issue-6 development runs")
    camera(m["color_camera"])
    if m["color_camera"].get("id") != "ego_cam_a":
        raise ValueError("v1 consumes color camera ego_cam_a")
    for key in ("camera_provenance", "hand_provenance"):
        if m.get(key) not in PROVENANCE:
            raise ValueError(f"Declare {key}")
    depth = m.get("depth")
    if depth is not None:
        camera(depth["camera"])
        rigid(depth["color_from_depth"])
        if depth["provenance"] not in PROVENANCE:
            raise ValueError("Declare depth provenance")
        if depth["encoding"] not in {"inverse_u16", "millimetres_u16", "metres"}:
            raise ValueError("Unsupported depth encoding")
    objects = m.get("objects", [])
    ids = [o["id"] for o in objects]
    if len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Object IDs must be unique nonempty names")
    for o in objects:
        if o.get("provenance") not in PROVENANCE:
            raise ValueError("Declare object-pose provenance")
        if m["mode"] == "predicted_object_integration" and o["provenance"] != "predicted":
            raise ValueError("Predicted-object mode requires predicted object poses")
    if m["mode"] != "synthetic" and (
        m["hand_provenance"] == "synthetic"
        or m["camera_provenance"] == "synthetic"
        or any(o["provenance"] == "synthetic" for o in objects)
        or (depth and depth["provenance"] == "synthetic")
    ):
        raise ValueError("Synthetic inputs must use synthetic mode")
    frames = m["frames"]
    if not frames:
        raise ValueError("Manifest needs at least one explicit frame")
    indexes = [f["id"] for f in frames]
    times = np.asarray([f["timestamp_s"] for f in frames], float)
    if (
        any(not isinstance(i, int) or i < 0 for i in indexes)
        or (len(indexes) > 1 and not np.all(np.diff(indexes) == 1))
        or not np.isfinite(times).all()
        or (len(times) > 1 and not np.all(np.diff(times) > 0))
    ):
        raise ValueError("Frames must be contiguous original IDs with increasing finite timestamps")
    for f in frames:
        if set(f.get("objects", {})) - set(ids):
            raise ValueError("Frame contains undeclared object")
        if f.get("depth_path") and depth is None:
            raise ValueError("Depth file needs depth calibration")
        if f.get("camera_valid") is True:
            if "camera_from_world" in f:
                rigid(f["camera_from_world"])
            elif f.get("camera_pose_path"):
                pose({"pose_path": f["camera_pose_path"]})
            else:
                raise ValueError("camera_valid requires a world-to-color-camera transform")
        for o in objects:
            camera_from_object(f, o)
    if "hand_source" in m:
        if m["hand_source"]["backend"] not in {"hamer", "dynhamr", "import"}:
            raise ValueError("Unknown hand backend")
    settings = m.get("settings", {})
    threshold = settings.get("contact_threshold_m", 0.02)
    maximum = settings.get("bridge_frames", 15)
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not np.isfinite(threshold)
        or threshold <= 0
        or type(maximum) is not int
        or maximum < 0
    ):
        raise ValueError("Invalid contact settings")
    minimum = settings.get("minimum_pixels", 256)
    if not isinstance(minimum, int) or minimum <= 0:
        raise ValueError("minimum_pixels must be a positive integer")
    if settings.get("reference_frame_id", indexes[0]) not in indexes:
        raise ValueError("reference_frame_id must be in the selected original-frame range")
    for f in frames:
        if "camera_from_world" in f and "camera_pose_path" in f:
            raise ValueError("Provide exactly one camera pose representation")
        if f.get("camera_pose_path"):
            stem = Path(f["camera_pose_path"]).stem
            if not stem.isdigit() or int(stem) != f["id"]:
                raise ValueError("Camera pose filename must match original frame ID")
        for state in f.get("objects", {}).values():
            if state.get("valid") is True and state.get("pose_path"):
                if not Path(state["pose_path"]).stem.isdigit() or int(Path(state["pose_path"]).stem) != f["id"]:
                    raise ValueError("Object pose filename must match original frame ID")


def input_files(m):
    paths = set()

    def walk(x):
        if isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, dict):
            for k, v in x.items():
                if k in PATH_KEYS and v:
                    p = Path(v)
                    if p.is_dir():
                        paths.update(f for f in p.rglob("*") if f.is_file())
                    else:
                        paths.add(p)
                elif k == "model_files":
                    paths.update(Path(f) for f in v)
                else:
                    walk(v)

    walk(m)
    return paths


def fingerprint(m):
    return {str(p): file_hash(real_file(p)) if p.exists() else "MISSING" for p in sorted(input_files(m))}


def inspect_inputs(m):
    pending = []
    w, h = m["color_camera"]["width"], m["color_camera"]["height"]
    for f in m["frames"]:
        p = Path(f["color"])
        if not p.exists():
            pending.append(f"color frame {f['id']}: {p}")
        else:
            with Image.open(real_file(p)) as im:
                if im.size != (w, h):
                    raise ValueError(f"Color dimensions/calibration mismatch at frame {f['id']}")
        if not f.get("depth_path"):
            pending.append(f"depth frame {f['id']}")
    for p in input_files(m):
        if not p.exists():
            pending.append(str(p))
        elif p.is_file():
            real_file(p)
    source = m.get("hand_source", {})
    if source.get("backend") != "import":
        for key in ("mano_assets_root", "model_files", "sam2_weights"):
            if not source.get(key):
                pending.append(f"hand_source.{key}")
    for f in m["frames"]:
        for o in m["objects"]:
            if camera_from_object(f, o) is None:
                pending.append(f"object pose {o['id']} frame {f['id']}")
    return {
        "pending": sorted(set(pending)),
        "complete_inputs": not pending,
        "real_reconstruction_verified": False,
    }


def source_state(root):
    def git(*args):
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unavailable"

    return {
        "revision": git("rev-parse", "HEAD"),
        "working_tree": git("status", "--porcelain"),
    }
