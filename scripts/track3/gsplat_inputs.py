#!/usr/bin/env python3
"""Assemble one Track 3 episode into the gsplat joint-refinement input layout.

Sources, all in the rectified ``ego_cam_c`` pixel grid:

* ``--bundle``: an issue-6 ``track3_adapter`` bundle (``color/``, metre-valued
  stereo ``depth/*.npy``, ``valid/``, ``manifest.json``).
* ``--results``: the Track 3 results root. Uses ``eNNN/camera/ego_cam_c``
  (stereo odometry, camera_from_world), ``eNNN/poses`` (observed FoundationPose
  object_to_cam), ``eNNN/inferred`` (gap-filled object_to_world) and
  ``eNNN/masks``.
* ``--hands-run`` (optional): an issue-6 run directory with
  ``align/records/<track>/*.json`` and ``hands/masks/{left,right}``.
* ``--data``: the public dataset, for the CAD meshes and the frame rate.

Output layout (consumed by ``refine_simple --objects_root``)::

    out/frames/{f:06d}.png        rectified color
    out/depth/{f:06d}.png         stereo depth, inverse_u16 (0 = unknown)
    out/intrinsics.json           {fx, fy, cx, cy, width, height}
    out/camera_c2w/{f:06d}.json   camera->world Transform3d (world = VO frame 0)
    out/objects/<name>/mesh.glb, poses/{f:06d}.json (object->camera), masks/
    out/hands/{left,right}/poses/{f:06d}.json, masks/
    out/inputs.json               sources, frame range and per-frame provenance

No reference (GT) pose is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SLAM_METHODS = {"droid_slam", "stereo_droid_slam", "stereo_depth_droid_slam", "orb_slam3_stereo", "orb_slam3_rgbd"}
ODOMETRY_METHODS = {"stereo_pnp_ransac"}


def camera_provenance(record: dict, source: Path) -> dict:
    """Preserve estimator evidence; frame-to-frame VO is not a SLAM claim."""
    method = record.get("method", "unknown")
    if not isinstance(method, str):
        raise TypeError(f"Camera method must be a string: {source}")
    kind = "slam" if method in SLAM_METHODS else "odometry" if method in ODOMETRY_METHODS else "unknown"
    return {"method": method, "kind": kind, "status": record.get("status", "unknown"),
            "gt_use": record.get("gt_use", "unknown"),
            "slam_run_evidence_sha256": record.get("slam_run_evidence_sha256"),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def quat_wxyz_to_matrix(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Expected a finite nonzero wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_wxyz(R: np.ndarray) -> list[float]:
    # Shepperd's method, numerically stable for every rotation.
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0] * 4
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = np.asarray(q)
    q = q if q[0] >= 0 else -q
    return (q / np.linalg.norm(q)).tolist()


def pose_record_to_matrix(record: dict) -> np.ndarray:
    """Track 3 pose JSON (wxyz rotation, metre translation, unit scale) -> 4x4."""
    scale = record.get("scale", 1.0)
    scale = [scale] * 3 if np.isscalar(scale) else list(scale)
    if len(scale) != 3 or any(not np.isfinite(s) or abs(float(s) - 1.0) > 1e-6 for s in scale):
        raise ValueError(f"Expected unit scale, got {scale}")
    M = np.eye(4)
    M[:3, :3] = quat_wxyz_to_matrix(record["rotation"])
    translation = np.asarray(record["translation"], dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("Expected a finite three-component translation")
    M[:3, 3] = translation
    return M


def matrix_to_transform3d(M: np.ndarray) -> dict:
    return {
        "rotation": matrix_to_quat_wxyz(M[:3, :3]),
        "translation": [float(v) for v in M[:3, 3]],
        "scale": [1.0, 1.0, 1.0],
    }


def invert_rigid(M: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = M[:3, :3].T
    out[:3, 3] = -M[:3, :3].T @ M[:3, 3]
    return out


def depth_to_inverse_u16(depth_m: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Encode metres as the repo's inverse-depth PNG (``65535 / (z + 1)``).

    Non-finite, non-positive or invalid pixels become 0, which the refinement
    loader decodes to +inf (unknown).
    """
    z = np.asarray(depth_m, dtype=np.float64)
    ok = np.isfinite(z) & (z > 0)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    px = np.zeros(z.shape, dtype=np.uint16)
    px[ok] = np.clip(np.round(65535.0 / (z[ok] + 1.0)), 1, 65535).astype(np.uint16)
    return px


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def mesh_for(data: Path, name: str) -> Path:
    # Same lookup order as foundationpose_track.py, so poses match the mesh frame.
    for suffix in ("", "_visual", "_collision"):
        candidate = data / "mesh" / name / f"{name}{suffix}.glb"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Public CAD mesh missing for {name}")


def object_pose_dirs(results: Path, episode: int, name: str) -> tuple[Path, Path]:
    ep = f"e{episode:03d}"
    observed = results / ep / "poses" / ep / "ego_object_poses" / ep / name / "object_to_cam" / "ego_cam_c"
    inferred = results / ep / "inferred" / ep / "ego_object_poses" / ep / name / "object_to_world"
    return observed, inferred


def object_camera_poses(
    results: Path,
    episode: int,
    name: str,
    frames: list[int],
    camera_from_world: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """camera_from_object for every frame that has an observed or inferred pose.

    FoundationPose poses explicitly marked observed and GT-free are used as-is.
    Other frames compose the
    gap-filled world pose with the stereo-odometry camera:
    ``camera_from_object = camera_from_world @ world_from_object``.
    """
    observed_dir, inferred_dir = object_pose_dirs(results, episode, name)
    poses: dict[int, np.ndarray] = {}
    provenance: dict[int, str] = {}
    for f in frames:
        obs = observed_dir / f"{f:06d}.json"
        if obs.is_file():
            record = read_json(obs)
            if record.get("status") == "observed" and record.get("gt_use") == "none":
                poses[f] = pose_record_to_matrix(record)
                provenance[f] = "observed"
                continue
        inf = inferred_dir / f"{f:06d}.json"
        if inf.is_file() and f in camera_from_world:
            record = read_json(inf)
            if record.get("gt_use", "none") not in ("none", "independent"):
                raise ValueError(f"GT-assisted inferred pose cannot initialize refinement: {inf}")
            poses[f] = camera_from_world[f] @ pose_record_to_matrix(record)
            provenance[f] = f"inferred:{record.get('inference_method', record.get('status', 'unknown'))}"
    return poses, provenance


def merge_hand_tracks(records_root: Path, frames: set[int], metric_valid_only: bool = True) -> dict[str, dict[int, dict]]:
    """One record per side per frame; on conflicts prefer the longer track.

    A wearer's hand can be reacquired as a new track id; the refinement takes
    one track per side, so tracks of the same side are merged by frame.
    """
    tracks: dict[str, dict[str, dict[int, dict]]] = defaultdict(dict)
    for track_dir in sorted(p for p in records_root.iterdir() if p.is_dir()):
        per_frame: dict[int, dict] = {}
        side = None
        for path in sorted(track_dir.glob("*.json")):
            record = read_json(path)
            f = int(record.get("frame_idx", path.stem))
            if f not in frames:
                continue
            if metric_valid_only and record.get("diagnostics", {}).get("metric_valid") is not True:
                continue
            record_side = "right" if record["is_right"] else "left"
            if side is not None and side != record_side:
                raise ValueError(f"Mixed handedness in {track_dir}")
            side = record_side
            per_frame[f] = record
        if side is not None and per_frame:
            tracks[side][track_dir.name] = per_frame
    merged: dict[str, dict[int, dict]] = {}
    for side, by_track in tracks.items():
        out: dict[int, dict] = {}
        for _, per_frame in sorted(by_track.items(), key=lambda kv: len(kv[1])):
            out.update(per_frame)  # longer tracks are applied last and win
        merged[side] = out
    return merged


def assemble(
    data: Path,
    results: Path,
    bundle: Path,
    episode: int,
    out: Path,
    hands_run: Path | None = None,
    start: int | None = None,
    frames_limit: int | None = None,
    metric_valid_hands_only: bool = True,
    require_slam: bool = False,
    slam_camera_dir: Path | None = None,
) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory {out} is not empty")
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("episode") != episode:
        raise ValueError(f"Bundle is for episode {manifest.get('episode')}, not {episode}")
    camera = manifest["color_camera"]
    if camera.get("id") != "ego_cam_c":
        raise ValueError(f"Track 3 poses are in rectified ego_cam_c; bundle uses {camera.get('id')}")
    bundle_frames = manifest["frames"]
    source_frames = [int(f["id"]) for f in bundle_frames]
    if start is not None:
        bundle_frames = [f for f in bundle_frames if f["id"] >= start]
    if frames_limit is not None:
        bundle_frames = bundle_frames[:frames_limit]
    frames = [int(f["id"]) for f in bundle_frames]
    if not frames:
        raise ValueError("No frames selected")

    ep_dir = results / f"e{episode:03d}"
    camera_from_world: dict[int, np.ndarray] = {}
    camera_status: dict[int, str] = {}
    camera_evidence: dict[int, dict] = {}
    for f in frames:
        path = ep_dir / "camera" / "ego_cam_c" / f"{f:06d}.json"
        if path.is_file():
            record = read_json(path)
            if record.get("gt_use", "independent") not in ("none", "independent"):
                raise ValueError(f"GT-assisted camera cannot initialize refinement: {path}")
            camera_from_world[f] = pose_record_to_matrix(record)
            camera_status[f] = record.get("status", "unknown")
            camera_evidence[f] = camera_provenance(record, path)
    missing_cameras = sorted(set(frames) - camera_from_world.keys())
    if missing_cameras:
        raise ValueError(f"Missing independent camera initialization for frames {missing_cameras}")
    # Inferred object poses belong to the original results world. Always convert
    # them with that original camera, even when SLAM supplies a different gauge.
    initialization_c2w = {f: invert_rigid(M) for f, M in camera_from_world.items()}
    if slam_camera_dir is not None:
        for f in frames:
            path = slam_camera_dir / f"{f:06d}.json"
            record = read_json(path)
            if record.get("direction") != "camera_to_world" or record.get("camera_frame") != "ego_cam_c":
                raise ValueError(f"SLAM pose must declare camera_to_world in ego_cam_c: {path}")
            evidence = camera_provenance(record, path)
            if evidence["kind"] != "slam" or evidence["gt_use"] not in ("none", "independent"):
                raise ValueError(f"Expected documented GT-free SLAM pose: {path}")
            initialization_c2w[f] = pose_record_to_matrix(record)
            camera_evidence[f] = evidence
            camera_status[f] = record.get("status", "unknown")
    if require_slam and any(p["kind"] != "slam" or p["gt_use"] not in ("none", "independent")
                            for p in camera_evidence.values()):
        raise ValueError("Issue 7 requires documented GT-free SLAM camera inputs; "
                         "odometry or unknown camera provenance cannot satisfy that requirement")

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "intrinsics.json", {k: camera[k] for k in ("fx", "fy", "cx", "cy", "width", "height")})
    for sub in ("frames", "depth", "camera_c2w"):
        (out / sub).mkdir()
    for frame in bundle_frames:
        f = int(frame["id"])
        shutil.copyfile(bundle / frame["color"], out / "frames" / f"{f:06d}.png")
        depth = np.load(bundle / frame["depth_path"], allow_pickle=False)
        valid = None
        if frame.get("depth_validity"):
            valid = np.asarray(Image.open(bundle / frame["depth_validity"])) > 0
        Image.fromarray(depth_to_inverse_u16(depth, valid)).save(out / "depth" / f"{f:06d}.png")
        if f in camera_from_world:
            write_json(out / "camera_c2w" / f"{f:06d}.json", {
                **matrix_to_transform3d(initialization_c2w[f]),
                **camera_evidence[f], "direction": "camera_to_world",
            })

    object_names = [o["id"] for o in manifest["objects"]]
    object_provenance = {}
    for name in object_names:
        obj_out = out / "objects" / name
        (obj_out / "poses").mkdir(parents=True)
        (obj_out / "masks").mkdir()
        shutil.copyfile(mesh_for(data, name), obj_out / "mesh.glb")
        poses, provenance = object_camera_poses(results, episode, name, frames, camera_from_world)
        for f, M in poses.items():
            write_json(obj_out / "poses" / f"{f:06d}.json", matrix_to_transform3d(M))
        for f in frames:
            mask = ep_dir / "masks" / name / f"{f:06d}.png"
            if mask.is_file():
                shutil.copyfile(mask, obj_out / "masks" / f"{f:06d}.png")
        object_provenance[name] = {str(f): provenance.get(f, "missing") for f in frames}

    hand_frames = {}
    if hands_run is not None:
        merged = merge_hand_tracks(hands_run / "align" / "records", set(frames), metric_valid_hands_only)
        for side, per_frame in merged.items():
            side_out = out / "hands" / side
            (side_out / "poses").mkdir(parents=True)
            (side_out / "masks").mkdir()
            for f, record in sorted(per_frame.items()):
                write_json(side_out / "poses" / f"{f:06d}.json", record)
            for f in frames:
                mask = hands_run / "hands" / "masks" / side / f"{f:06d}.png"
                if mask.is_file():
                    shutil.copyfile(mask, side_out / "masks" / f"{f:06d}.png")
            hand_frames[side] = sorted(per_frame)

    fps = None
    info = data / "meta" / "info.json"
    if info.is_file():
        fps = read_json(info).get("fps")

    summary = {
        "episode": episode,
        "frames": frames,
        "source_frames": source_frames,
        "full_episode": frames == source_frames,
        "fps": fps,
        "camera_frame": "ego_cam_c",
        "coordinate_frame": "rectified_camera",
        "world": "camera estimator's world frame; refinement reanchors to its reference camera",
        "gt_use": "none",
        "sources": {
            "bundle": str(bundle.resolve()),
            "results": str(ep_dir.resolve()),
            "hands_run": str(hands_run.resolve()) if hands_run else None,
            "data": str(data.resolve()),
            "slam_camera_dir": str(slam_camera_dir.resolve()) if slam_camera_dir else None,
        },
        "camera_status": {str(f): camera_status.get(f, "missing") for f in frames},
        "camera_provenance": {str(f): camera_evidence[f] for f in frames},
        "camera_is_slam": all(p["kind"] == "slam" and p["gt_use"] in ("none", "independent")
                              for p in camera_evidence.values()),
        "require_slam": require_slam,
        "object_provenance": object_provenance,
        "hand_frames": hand_frames,
        "metric_valid_hands_only": metric_valid_hands_only,
        "counts": {
            "frames": len(frames),
            "camera_poses": len(camera_from_world),
            "objects": {
                name: {
                    "observed": sum(v == "observed" for v in prov.values()),
                    "inferred": sum(v.startswith("inferred") for v in prov.values()),
                    "missing": sum(v == "missing" for v in prov.values()),
                }
                for name, prov in object_provenance.items()
            },
            "hands": {side: len(v) for side, v in hand_frames.items()},
        },
    }
    write_json(out / "inputs.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--data", type=Path, required=True, help="Public Track 3 dataset root")
    p.add_argument("--results", type=Path, required=True, help="Track 3 results root (contains eNNN/)")
    p.add_argument("--bundle", type=Path, required=True, help="issue6.track3_adapter bundle directory")
    p.add_argument("--hands-run", type=Path, default=None, help="issue-6 run directory with align/records")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--require-slam", action="store_true",
                   help="Require declared independent SLAM provenance before writing any inputs")
    p.add_argument("--slam-camera-dir", type=Path, default=None,
                   help="Independent SLAM camera_to_world poses in ego_cam_c; original cameras still convert inferred objects")
    p.add_argument("--metric-valid-hands-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Drop hand frames whose stereo alignment was not metric-valid (default: enabled)")
    args = p.parse_args()
    summary = assemble(
        args.data, args.results, args.bundle, args.episode, args.out,
        hands_run=args.hands_run, start=args.start, frames_limit=args.frames,
        metric_valid_hands_only=args.metric_valid_hands_only,
        require_slam=args.require_slam,
        slam_camera_dir=args.slam_camera_dir,
    )
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
