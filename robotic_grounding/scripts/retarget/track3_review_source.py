# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original CAD/MANO surface fit to locked masks and calibrated stereo depth.

Development diagnostic only. Masks/depth are estimated inputs, not ground truth.
Triangle rasterization uses integer calibrated pixel coordinates, perspective
correct depth, no back-face culling, and nearest visible surface ownership.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numba
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source/robotic_grounding"))
from robotic_grounding.retarget.track3_bundle import validate_bundle  # noqa: E402


def resolve_source(source: str, mappings: list[dict]) -> Path:
    """Apply the longest explicit source-root mapping without changing bytes."""
    matching = [
        entry
        for entry in mappings
        if source == entry["source_prefix"].rstrip("/")
        or source.startswith(entry["source_prefix"].rstrip("/") + "/")
    ]
    if not matching:
        return Path(source)
    matching.sort(
        key=lambda entry: len(entry["source_prefix"].rstrip("/")), reverse=True
    )
    prefix = matching[0]["source_prefix"].rstrip("/")
    relative = source[len(prefix) :].lstrip("/")
    if ".." in Path(relative).parts:
        raise ValueError("Source mapping cannot escape its explicit root")
    return Path(matching[0]["local_prefix"]) / relative


def mask_source_hash(hashes: dict, frame: int, name: str, is_hand: bool) -> str:
    """Require the matching source frame and identity, not any known PNG bytes."""
    filename = f"{frame:06d}.png"
    markers = (
        (f"/hands/masks/{name}/{filename}", f"/hands/{name}/masks/{filename}")
        if is_hand
        else (f"/objects/{name}/masks/{filename}", f"/masks/{name}/{filename}")
    )
    expected = {value for source, value in hashes.items() if source.endswith(markers)}
    if len(expected) != 1:
        raise ValueError(f"Expected one semantic mask lock for {name}/{frame}")
    return expected.pop()


@numba.njit(cache=True)
def raster(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsic: np.ndarray,
    depths: np.ndarray,
    owners: np.ndarray,
    owner: int,
) -> None:
    """Rasterize front-plane triangles with nearest perspective-correct depth."""
    height, width = depths.shape
    for face in faces:
        tri = vertices[face]
        if np.min(tri[:, 2]) <= 0.01:
            continue
        u = intrinsic[0, 0] * tri[:, 0] / tri[:, 2] + intrinsic[0, 2]
        v = intrinsic[1, 1] * tri[:, 1] / tri[:, 2] + intrinsic[1, 2]
        determinant = (v[1] - v[2]) * (u[0] - u[2]) + (u[2] - u[1]) * (v[0] - v[2])
        if abs(determinant) < 1e-10:
            continue
        x0 = max(0, int(np.ceil(np.min(u))))
        x1 = min(width - 1, int(np.floor(np.max(u))))
        y0 = max(0, int(np.ceil(np.min(v))))
        y1 = min(height - 1, int(np.floor(np.max(v))))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                a = (
                    (v[1] - v[2]) * (x - u[2]) + (u[2] - u[1]) * (y - v[2])
                ) / determinant
                b = (
                    (v[2] - v[0]) * (x - u[2]) + (u[0] - u[2]) * (y - v[2])
                ) / determinant
                c = 1 - a - b
                if a >= -1e-8 and b >= -1e-8 and c >= -1e-8:
                    z = 1 / (a / tri[0, 2] + b / tri[1, 2] + c / tri[2, 2])
                    if z < depths[y, x]:
                        depths[y, x] = z
                        owners[y, x] = owner


def digest(path: Path) -> str:
    """Hash source inputs incrementally without changing their bytes."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    """Verify frozen inputs, measure every selected frame and save evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", nargs="+", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--prepared-inputs",
        type=Path,
        required=True,
        help="Frozen adapter inputs with hands/<side>/masks and objects/<name>/masks",
    )
    parser.add_argument(
        "--path-map",
        type=Path,
        help="JSON list of source_prefix/local_prefix mappings for relocated locked rasters",
    )
    parser.add_argument(
        "--witnesses",
        type=Path,
        help="Optional manifest-bound collision witness report containing world points",
    )
    parser.add_argument(
        "--image-every",
        type=int,
        default=10,
        help="Write every Nth reviewed frame plus all supplied witnesses",
    )
    args = parser.parse_args()
    source_sha256 = digest(Path(__file__))
    bundle = args.bundle
    inputs = args.prepared_inputs
    if args.image_every < 1:
        raise ValueError("image-every must be positive")
    validation = validate_bundle(bundle)
    mappings = json.loads(args.path_map.read_text()) if args.path_map else []
    for entry in mappings:
        if not isinstance(entry, dict) or set(entry) != {
            "source_prefix",
            "local_prefix",
        }:
            raise ValueError("Path map entries require source_prefix and local_prefix")
    lock = json.loads((bundle / "source/provenance/inputs-lock.json").read_text())
    hashes = {item["path"]: item["sha256"] for item in lock["source_files"]}
    manifest = json.loads((bundle / "manifest.json").read_text())
    if digest(bundle / "episode.npz") != manifest["payload"]["sha256"]:
        raise ValueError("Source bundle payload changed")
    with np.load(bundle / "episode.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "images").mkdir()
    width, height = map(int, data["image_size"])
    if not np.allclose(
        data["intrinsics"][[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]],
        [0, 0, 0, 0, 1],
        atol=1e-12,
    ):
        raise ValueError("Diagnostic requires rectified zero-skew intrinsics")
    objects = len(manifest["objects"])
    names = [obj["name"] for obj in manifest["objects"]] + ["left", "right"]
    colors = np.array(
        [[35, 180, 80] if i % 2 == 0 else [200, 50, 230] for i in range(objects)]
        + [[35, 100, 255], [255, 110, 25]],
        dtype=np.uint8,
    )
    rows: list[dict] = []
    witness_report = json.loads(args.witnesses.read_text()) if args.witnesses else {}
    if (
        args.witnesses
        and witness_report.get("bundle_manifest_sha256")
        != validation["manifest_sha256"]
    ):
        raise ValueError("Witnesses belong to a different source bundle")
    witnesses = witness_report.get("witnesses", [])
    selected = list(data["frame_indices"]) if args.frames is None else args.frames
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected).issubset(set(data["frame_indices"].tolist()))
    ):
        raise ValueError("Requested frames must be unique source frame IDs")
    verified = []
    for frame in selected:
        row = int(np.flatnonzero(data["frame_indices"] == frame)[0])
        inverse = np.linalg.inv(data["camera_c2w"][row])
        zbuffer = np.full((height, width), np.inf)
        owners = np.full((height, width), -1, dtype=np.int16)
        for oi in range(objects):
            transform = data[f"object_{oi}_world"][row]
            world = (
                data[f"object_{oi}_vertices"]
                @ (transform[:3, :3] * data[f"object_{oi}_scale"][row]).T
                + transform[:3, 3]
            )
            camera = world @ inverse[:3, :3].T + inverse[:3, 3]
            if np.min(camera[:, 2]) <= 0.01:
                raise ValueError("Object crosses the unsupported diagnostic near plane")
            raster(
                camera,
                data[f"object_{oi}_faces"],
                data["intrinsics"],
                zbuffer,
                owners,
                oi,
            )
        for si, side in enumerate(("left", "right")):
            if data["hand_valid"][row, si]:
                camera = (
                    data["hand_vertices_world"][row, si] @ inverse[:3, :3].T
                    + inverse[:3, 3]
                )
                if np.min(camera[:, 2]) <= 0.01:
                    raise ValueError(
                        "Hand crosses the unsupported diagnostic near plane"
                    )
                raster(
                    camera,
                    data[f"hand_{side}_faces"],
                    data["intrinsics"],
                    zbuffer,
                    owners,
                    si + objects,
                )
        sources = {}
        for field, marker in [
            ("rgb", "/color/"),
            ("depth", "/depth/"),
            ("valid", "/valid/"),
        ]:
            suffix = (
                f"{int(frame):06d}.npy" if field == "depth" else f"{int(frame):06d}.png"
            )
            matches = [
                path for path in hashes if marker in path and path.endswith(suffix)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one locked {field} source at frame {frame}; found {matches}"
                )
            source_path = resolve_source(matches[0], mappings)
            if digest(source_path) != hashes[matches[0]]:
                raise ValueError(f"Source raster changed: {source_path}")
            sources[field] = source_path
            verified.append({"path": matches[0], "sha256": hashes[matches[0]]})
        depth = np.load(sources["depth"], allow_pickle=False)
        valid = np.asarray(Image.open(sources["valid"])) > 0
        if valid.ndim == 3:
            valid = valid[:, :, 0]
        if depth.shape != (height, width) or valid.shape != (height, width):
            raise ValueError("Source depth/validity do not match calibrated resolution")
        valid &= np.isfinite(depth) & (depth > 0.01) & (depth < 3.0)
        metrics = []
        for owner, name in enumerate(names):
            path = (
                inputs / f"objects/{name}/masks/{int(frame):06d}.png"
                if owner < objects
                else inputs / f"hands/{name}/masks/{int(frame):06d}.png"
            )
            if not path.exists():
                metrics.append({"name": name, "mask_status": "missing_unknown"})
                continue
            mask_sha = digest(path)
            if mask_sha != mask_source_hash(hashes, int(frame), name, owner >= objects):
                raise ValueError(f"Mask differs from locked source: {path}")
            mask = np.asarray(Image.open(path))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if mask.shape != (height, width):
                raise ValueError(f"Mask does not match calibrated resolution: {path}")
            mask = mask > 0
            predicted = owners == owner
            intersection = predicted & mask
            union = predicted | mask
            support = intersection & valid
            residual = zbuffer[support] - depth[support]
            entry = {
                "name": name,
                "mask_sha256": mask_sha,
                "source_mask_pixels": int(mask.sum()),
                "predicted_visible_pixels": int(predicted.sum()),
                "intersection_pixels": int(intersection.sum()),
                "union_pixels": int(union.sum()),
                "iou": float(intersection.sum() / max(1, union.sum())),
                "rendered_surface_mask_precision": float(
                    intersection.sum() / max(1, predicted.sum())
                ),
                "stereo_valid_overlap_pixels": int(support.sum()),
            }
            if residual.size:
                entry.update(
                    median_signed_depth_error_m=float(np.median(residual)),
                    median_absolute_depth_error_m=float(np.median(np.abs(residual))),
                    p95_absolute_depth_error_m=float(
                        np.quantile(np.abs(residual), 0.95)
                    ),
                )
            metrics.append(entry)
        witness_results = []
        for witness in witnesses:
            if witness["frame"] != frame:
                continue
            point = (
                np.asarray(witness["point_world_m"]) @ inverse[:3, :3].T
                + inverse[:3, 3]
            )
            pixel = data["intrinsics"] @ point
            uv = pixel[:2] / pixel[2]
            x, y = np.rint(uv).astype(int)
            entry = {
                "kind": witness["kind"],
                "side": witness["side"],
                "pixel": uv.tolist(),
                "camera_z_m": float(point[2]),
                "in_image": bool(0 <= x < width and 0 <= y < height),
            }
            if entry["in_image"]:
                entry.update(
                    front_surface_owner=int(owners[y, x]),
                    front_surface_z_m=(
                        float(zbuffer[y, x]) if owners[y, x] >= 0 else None
                    ),
                    depth_behind_front_surface_m=(
                        float(point[2] - zbuffer[y, x]) if owners[y, x] >= 0 else None
                    ),
                    stereo_valid=bool(valid[y, x]),
                    stereo_z_m=float(depth[y, x]) if valid[y, x] else None,
                )
            witness_results.append(entry)
        source = Image.open(sources["rgb"]).convert("RGB")
        overlay = np.asarray(source).copy()
        hit = owners >= 0
        overlay[hit] = (0.5 * overlay[hit] + 0.5 * colors[owners[hit]]).astype(np.uint8)
        error = np.full((height, width, 3), 40, dtype=np.uint8)
        support = valid & hit
        difference = np.clip((zbuffer[support] - depth[support]) / 0.02, -1, 1)
        error[support] = np.stack(
            [
                255 * np.maximum(difference, 0),
                255 * (1 - np.abs(difference)),
                255 * np.maximum(-difference, 0),
            ],
            axis=1,
        ).astype(np.uint8)
        panel = Image.new("RGB", (width * 3, height + 35), (20, 20, 20))
        for i, (im, label) in enumerate(
            zip(
                (source, Image.fromarray(overlay), Image.fromarray(error)),
                (
                    "Source RGB",
                    "Source objects + recovered MANO visible surfaces",
                    "Stereo depth residual: blue closer / red farther; +/-20mm",
                ),
                strict=True,
            )
        ):
            panel.paste(im, (i * width, 35))
            ImageDraw.Draw(panel).text(
                (i * width + 8, 10), f"Frame {frame} | {label}", fill="white"
            )
        if args.frames is not None or row % args.image_every == 0 or witness_results:
            panel.save(args.out / "images" / f"{int(frame):06d}.jpg", quality=92)
        rows.append(
            {
                "frame": int(frame),
                "objects_and_hands": metrics,
                "witnesses": witness_results,
            }
        )
        if args.frames is not None or frame % 25 == 0:
            print(
                json.dumps(
                    {
                        "frame": int(frame),
                        "iou": {
                            m["name"]: round(m["iou"], 3) for m in metrics if "iou" in m
                        },
                        "witnesses": witness_results,
                    }
                ),
                flush=True,
            )
    summary: dict = {}
    frame_rows = {int(frame): row for row, frame in enumerate(data["frame_indices"])}
    for si, side in enumerate(("left", "right")):
        measured = [
            r["objects_and_hands"][si + objects]
            for r in rows
            if data["hand_valid"][frame_rows[r["frame"]], si]
            and "iou" in r["objects_and_hands"][si + objects]
        ]
        summary[side] = {"supported_frames": len(measured)}
        for key in (
            "iou",
            "rendered_surface_mask_precision",
            "median_absolute_depth_error_m",
            "median_signed_depth_error_m",
            "p95_absolute_depth_error_m",
        ):
            values = [r[key] for r in measured if key in r]
            if values:
                summary[side][key] = {
                    "median_over_frames": float(np.median(values)),
                    "p95_over_frames": float(np.quantile(values, 0.95)),
                }
    report = {
        "source_sha256": source_sha256,
        "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "episode": manifest["episode"],
        "selected_frame_ids": [int(f) for f in selected],
        "source_hand_valid_counts": data["hand_valid"].sum(axis=0).tolist(),
        "source_hand_directly_observed_counts": data["hand_directly_observed"]
        .sum(axis=0)
        .tolist(),
        "display_holds_used": False,
        "path_mappings": mappings,
        "prepared_inputs": str(inputs),
        "input_lock_sha256": digest(bundle / "source/provenance/inputs-lock.json"),
        "witnesses_sha256": digest(args.witnesses) if args.witnesses else None,
        "rows": rows,
        "summary": summary,
        "verified_source_files": verified,
        "runtime": {"numpy": np.__version__, "numba": numba.__version__},
        "scope": "visible CAD/MANO triangle surfaces against estimated source masks and calibrated stereo depth; not ground-truth accuracy; masks include forearms absent from MANO, so IoU alone is not a valid whole-hand accuracy score",
        "physical_ground_truth_known": False,
        "physical_task_accepted": False,
        "hand_pose_accuracy_accepted": False,
        "contact_targets_accepted": False,
        "source_observations_modified": False,
    }
    if digest(Path(__file__)) != source_sha256:
        raise ValueError("Review source changed during execution; no report accepted")
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
