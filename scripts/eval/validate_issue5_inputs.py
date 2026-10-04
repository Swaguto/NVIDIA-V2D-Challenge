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
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True, help="public data, meta, meshes and episode-0/2 RGB")
parser.add_argument("--out", type=Path, required=True, help="fresh diagnostic directory")
args = parser.parse_args()
DATA, OUT = args.data, args.out
if OUT.exists() and any(OUT.iterdir()):
    raise ValueError("diagnostic output must be a fresh directory")
sys.path[:0] = [str(REPO / "scripts/eval"), str(REPO / "scripts/track3"),
                str(REPO / "reconstruction/modules/v2d_stereo_depth/lib")]
from glb_mesh import load_vertices, sample_surface
from reference_loader import load_reference
from stereo_depth import load_rig, stereo_depth
from validate_stereo import lr_stats, plane_inliers, read_frame


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rates(result):
    values, count = result
    return {"reverse_valid_textured_pixels": count,
            "disagreement_fraction": {str(t): value if np.isfinite(value) else None
                                      for t, value in values.items()}}


OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((DATA / "asset_download_manifest.json").read_text())
for item in manifest["files"]:
    path = DATA / item["path"]
    assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], path
result = {"dataset": manifest["dataset"], "revision": manifest["revision"],
          "input_manifest_sha256": sha(DATA / "asset_download_manifest.json"),
          "all_14_downloaded_asset_hashes_verified": True,
          "camera_calibration_sha256": sha(DATA / "meta/camera_calibration.json"),
          "mesh_checks": [], "episodes": [],
          "evidence_limits": ["No segmentation or object poses estimated in this diagnostic.",
            "Stereo statistics cover whole images, not a validated object mask.",
            "Plane is measured in camera coordinates; no support-surface identification.",
            "No independent camera trajectory or real official pose accuracy measured."]}
for path in sorted((DATA / "mesh").glob("*/*.glb")):
    with path.open("rb") as file: vertices, indices = load_vertices(file)
    full = trimesh.load(path, process=False).to_mesh()
    bounds = np.stack([vertices.min(0), vertices.max(0)])
    np.testing.assert_allclose(bounds, full.bounds, atol=1e-8)
    assert len(vertices) == len(full.vertices) and len(indices) == full.faces.size
    points = sample_surface(vertices, indices, num_points=4096)
    assert np.isfinite(points).all()
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
    assert np.all(np.diff(ref.capture_time) > 0) and np.isfinite(ref.capture_time).all()
    paths = {camera: DATA / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"
             for camera in (rig.left, rig.right)}
    videos = {}
    for camera, path in paths.items():
        cap = cv2.VideoCapture(str(path)); count = 0
        size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        while cap.read()[0]: count += 1
        cap.release()
        assert count == ref.steps and size == list(rig.size)
        assert np.array_equal(ref.frame_index, np.arange(count))
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
