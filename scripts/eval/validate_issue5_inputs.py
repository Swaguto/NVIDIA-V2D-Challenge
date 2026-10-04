"""Bounded real-input diagnostics; never a reconstructed-pose accuracy claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/eval"), str(REPO / "scripts/track3"),
                str(REPO / "reconstruction/modules/v2d_stereo_depth/lib")]
from glb_mesh import load_vertices, sample_surface
from reference_loader import load_reference
from stereo_depth import load_rig, stereo_depth
from validate_stereo import lr_stats, plane_inliers, read_frame


def sha(path):
    """Hash inputs without reading an entire mesh/video into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    """Keep operational validation enabled when Python is run with -O."""
    if not condition:
        raise ValueError(message)


def validate_pinned_inputs(data):
    """Verify the complete repository-pinned asset set before opening external data."""
    evidence = REPO / "docs/team/issue5/evidence"
    expected_path = evidence / "asset_download_manifest.json"
    expected = json.loads(expected_path.read_text())
    manifest_path = data / "asset_download_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(all(manifest.get(k) == expected[k] for k in ("dataset", "revision")),
            "input manifest does not match the pinned dataset/revision")
    supplied = {}
    for item in manifest["files"]:
        key = item["path"]
        require(key not in supplied, f"duplicate input manifest path: {key}")
        supplied[key] = item
    for item in expected["files"]:
        key = item["path"]
        require(key in supplied and all(supplied[key].get(k) == item[k]
                for k in ("bytes", "sha256")), f"missing or changed pinned asset metadata: {key}")
        path = data / key
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha(path) == item["sha256"], f"pinned asset size/hash mismatch: {key}")
    prior = json.loads((evidence / "local_results.json").read_text())
    inputs = {"meta/camera_calibration.json": prior["camera_calibration_sha256"]}
    for episode in prior["episodes"]:
        inputs[f'data/chunk-000/episode_{episode["episode"]:06d}.parquet'] = episode["gt_parquet_sha256"]
    for key, digest in inputs.items():
        require(sha(data / key) == digest, f"pinned calibration/reference hash mismatch: {key}")
    return expected, sha(manifest_path), sha(expected_path)


def rates(result):
    values, count = result
    return {"reverse_valid_textured_pixels": count,
            "disagreement_fraction": {str(t): value if np.isfinite(value) else None
                                      for t, value in values.items()}}


def run(DATA, OUT):
    """Run the bounded episode-0/2 diagnostics only after pinned input verification."""
    if OUT.exists() and (not OUT.is_dir() or any(OUT.iterdir())):
        raise ValueError("diagnostic output must be a fresh directory")
    manifest, supplied_hash, expected_hash = validate_pinned_inputs(DATA)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"dataset": manifest["dataset"], "revision": manifest["revision"],
              "input_manifest_sha256": supplied_hash,
              "expected_asset_manifest_sha256": expected_hash,
              "verified_asset_count": len(manifest["files"]),
              "all_14_downloaded_asset_hashes_verified": True,
              "camera_calibration_sha256": sha(DATA / "meta/camera_calibration.json"),
              "mesh_checks": [], "episodes": [],
              "evidence_limits": ["No segmentation or object poses estimated in this diagnostic.",
                "Stereo statistics cover whole images, not a validated object mask.",
                "Plane is measured in camera coordinates; no support-surface identification.",
                "No independent camera trajectory or real official pose accuracy measured."]}
    for key in sorted(item["path"] for item in manifest["files"] if item["path"].startswith("mesh/")):
        path = DATA / key
        with path.open("rb") as file: vertices, indices = load_vertices(file)
        full = trimesh.load(path, process=False).to_mesh()
        bounds = np.stack([vertices.min(0), vertices.max(0)])
        require(np.allclose(bounds, full.bounds, rtol=1e-7, atol=1e-8), f"mesh bounds mismatch: {path}")
        require(len(vertices) == len(full.vertices) and len(indices) == full.faces.size, f"mesh geometry count mismatch: {path}")
        points = sample_surface(vertices, indices, num_points=4096)
        require(np.isfinite(points).all(), f"nonfinite surface samples: {path}")
        result["mesh_checks"].append({"path": str(path.relative_to(DATA)),
            "vertices": len(vertices), "triangles": len(indices) // 3,
            "bounds_m": bounds.tolist(), "full_trimesh_geometry_matches": True})

    rig = load_rig(str(DATA / "meta/camera_calibration.json"))
    result["rectification"] = {"left": rig.left, "right": rig.right,
        "baseline_m": rig.baseline_m, "raw_translation_norm_m": float(np.linalg.norm(rig.T_lr)),
        "fx_rect": rig.fx_rect, "R1": rig.R1.tolist(), "R2": rig.R2.tolist(),
        "P1": rig.P1.tolist(), "P2": rig.P2.tolist(), "Q": rig.Q.tolist()}
    for episode in (0, 2):
        ref = load_reference(episode, DATA)
        require(np.all(np.diff(ref.capture_time) > 0) and np.isfinite(ref.capture_time).all(), f"invalid capture times: episode {episode}")
        paths = {camera: DATA / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"
                 for camera in (rig.left, rig.right)}
        videos = {}
        for camera, path in paths.items():
            cap = cv2.VideoCapture(str(path)); count = 0
            size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            while cap.read()[0]: count += 1
            cap.release()
            require(count == ref.steps and size == list(rig.size), f"video count/resolution mismatch: {path}")
            require(np.array_equal(ref.frame_index, np.arange(count)), f"video/GT source frame ID mismatch: {path}")
            videos[camera] = {"decoded_frames": count, "resolution": size, "container_fps": fps}
        item = {"episode": episode, "objects": list(ref.object_names), "gt_frames": ref.steps,
            "gt_parquet_sha256": sha(DATA / f"data/chunk-000/episode_{episode:06d}.parquet"),
            "capture_time_range_s": [float(ref.capture_time[0]), float(ref.capture_time[-1])],
            "both_videos_exactly_match_gt_source_frame_ids": True, "videos": videos, "frames": []}
        for frame in (0, 120, 200, ref.steps - 1):
            left = read_frame(paths[rig.left], frame); right = read_frame(paths[rig.right], frame)
            pl, pr = rig.rectify_pair(left, right)
            depth, valid = stereo_depth(rig, left, right)
            pixels = depth[valid]
            ratio, rms, n = plane_inliers(depth, valid, rig.P1)
            check = {"frame_index": frame, "capture_time_s": float(ref.capture_time[frame]),
                "valid_depth_pixels": int(valid.sum()), "valid_depth_fraction": float(valid.mean()),
                "whole_image_depth_quantiles_m": np.quantile(pixels, [.05, .5, .95]).tolist() if len(pixels) else [],
                "left_right_consistency": rates(lr_stats(pl, pr)),
                "vertical_shift_controls": {str(s): rates(lr_stats(pl, np.roll(pr, s, axis=0))) for s in (2, 4)},
                "largest_plane": {"sampled_points": n, "inlier_fraction_at_10mm": ratio,
                                  "rms_mm": rms * 1000 if np.isfinite(rms) else None}}
            item["frames"].append(check)
            if frame == 120:
                preview = cv2.applyColorMap(np.uint8(np.clip((depth - .2) / 1.8, 0, 1) * 255), cv2.COLORMAP_TURBO)
                preview[~valid] = 0
                cv2.imwrite(str(OUT / f"ep{episode}_rectified_stereo_frame120.png"), np.hstack([pl, pr]))
                cv2.imwrite(str(OUT / f"ep{episode}_depth_frame120.png"), preview)
            print(json.dumps({"episode": episode, "frame": frame, "valid_fraction": check["valid_depth_fraction"],
                              "lr": check["left_right_consistency"]}), flush=True)
        result["episodes"].append(item)
    path = OUT / "local_results.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"meshes_checked": len(result['mesh_checks']), "frames_checked": 8, "report": str(path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="pinned public data, meta, meshes and episode-0/2 RGB")
    parser.add_argument("--out", type=Path, required=True, help="fresh diagnostic directory")
    args = parser.parse_args()
    run(args.data, args.out)


if __name__ == "__main__":
    main()
