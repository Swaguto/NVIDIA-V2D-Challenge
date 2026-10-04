"""One real RGB/stereo/CAD frame with manual observation controls; no GT pose score."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time

import cv2
import numpy as np
import torch
import trimesh

REPO = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True, help="pinned public assets for episode 39")
parser.add_argument("--out", type=Path, required=True, help="fresh control output directory")
args = parser.parse_args()
HERE = args.out
if HERE.exists() and (not HERE.is_dir() or any(HERE.iterdir())):
    parser.error("--out must be a fresh directory")
evidence = REPO / "docs/team/issue5/evidence"
expected = json.loads((evidence / "episode39_input_manifest.json").read_text())
for item in expected["files"]:
    path = args.data / item["path"]
    if path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(f"input differs from the pinned control asset: {path}")
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts/track3"))
from register_poses import register, trim_cloud, voxel_downsample

torch.set_num_threads(1)
sys.path.insert(0, str(REPO / "reconstruction/modules/v2d_stereo_depth/lib"))
from stereo_depth import load_rig, stereo_depth
rig = load_rig(str(args.data / "meta/camera_calibration.json"))
images, counts = [], []
for camera in (rig.left, rig.right):
    path = args.data / f"videos/chunk-000/observation.images.{camera}/episode_000039.mp4"
    capture = cv2.VideoCapture(str(path))
    counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    ok, image = capture.read(); capture.release()
    if not ok or image.shape[:2] != (800, 1280):
        raise ValueError(f"missing or mismatched frame zero: {path}")
    images.append(image)
if counts != [319, 319]:
    raise ValueError(f"stereo source frame counts differ from the pinned control: {counts}")
image, _ = rig.rectify_pair(*images)
depth, valid = stereo_depth(rig, *images)
P1 = rig.P1
annotation = json.loads((evidence / "episode39_mask_annotations.json").read_text())
if (annotation["episode"], annotation["frame_index"], annotation["camera_frame"],
        annotation["coordinate_frame"], annotation["resolution"]) != (
            39, 0, rig.left, "rectified_camera", [1280, 800]):
    raise ValueError("mask annotations do not match the frozen frame")
pan_polygon = np.asarray(annotation["pan_polygon"], np.int32)
table_polygon = np.asarray(annotation["table_polygon"], np.int32)
hole = annotation["handle_hole"]
pan = np.zeros(depth.shape, np.uint8); table = pan.copy()
cv2.fillPoly(pan, [pan_polygon], 1)
cv2.ellipse(pan, tuple(hole["center"]), tuple(hole["radii"]), 0, 0, 360, 0, -1)
cv2.fillPoly(table, [table_polygon], 1)
cv2.imwrite(str(HERE / "manual_pan_mask.png"), pan * 255)
cv2.imwrite(str(HERE / "manual_table_mask.png"), table * 255)
mesh_path = args.data / "mesh/mini_dust_pan/mini_dust_pan.glb"
scene = trimesh.load(mesh_path, process=False)
mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
cad, _ = trimesh.sample.sample_surface(mesh, 1200, seed=3)
cad_tensor = torch.tensor(cad, dtype=torch.float64)
yy, xx = np.indices(depth.shape)
points = np.stack([(xx - P1[0,2]) * depth / P1[0,0],
    (yy - P1[1,2]) * depth / P1[1,1], depth], axis=-1)


def silhouette(R, t):
    vertices = np.asarray(mesh.vertices) @ R.T + t
    pixels = vertices @ P1[:3,:3].T
    pixels = pixels[:,:2] / pixels[:,2,None]
    out = np.zeros(depth.shape, np.uint8)
    front = (vertices[np.asarray(mesh.faces), 2] > .01).all(1)
    for face in np.asarray(mesh.faces)[front]:
        cv2.fillConvexPoly(out, np.round(pixels[face]).astype(np.int32), 1)
    return out.astype(bool)


def iou(a, b):
    return float(np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1))


results = {}
for name, mask in [("table_inclusive", table), ("pan_only", pan)]:
    start = time.perf_counter()
    cloud = trim_cloud(voxel_downsample(points[(mask > 0) & valid], .004))
    count = len(cloud)
    if count > 1600:
        cloud = cloud[np.random.default_rng(7).choice(count, 1600, replace=False)]
    s, R, t, residual = register(cad_tensor, torch.tensor(cloud, dtype=torch.float64),
        allow_scale=False, trim=.6, coarse=600, refine=1200, n_seed=3)
    rendered = silhouette(R.numpy(), t.numpy())
    row = {"mask_pixels": int(mask.sum()), "valid_depth_pixels": int(np.count_nonzero((mask > 0) & valid)),
        "filtered_cloud_points_before_cap": count, "registration_cloud_points": len(cloud),
        "cloud_extent_m": np.ptp(cloud, axis=0).tolist(), "scale": s,
        "rotation_matrix": R.tolist(), "translation_m": t.tolist(), "residual_m": residual,
        "passes_current_native_residual_gate_50mm": residual <= .05,
        "rendered_vs_pan_only_iou": iou(rendered, pan > 0),
        "rendered_vs_input_mask_iou": iou(rendered, mask > 0),
        "seconds": time.perf_counter() - start}
    overlay = image.copy()
    overlay[rendered] = np.uint8(.6*overlay[rendered] + .4*np.array([0,255,255]))
    cv2.polylines(overlay, [pan_polygon], True, (0,255,0), 2)
    cv2.putText(overlay, name + f" | target IoU {row['rendered_vs_pan_only_iou']:.3f}",
        (15,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
    cv2.imwrite(str(HERE / f"frame0_{name}_control.png"), overlay)
    results[name] = row
    print(json.dumps({name: row}), flush=True)

manifest = expected
report = {"episode":39,"frame_index":0,"coordinate_frame":"rectified_camera", "camera_frame":"ego_cam_c",
    "data_revision":manifest["revision"],"inputs":manifest["files"],
    "calibration_sha256":hashlib.sha256((args.data/'meta/camera_calibration.json').read_bytes()).hexdigest(),
    "mesh_bounds_m":mesh.bounds.tolist(),"manual_pan_polygon":pan_polygon.tolist(),
    "manual_handle_hole":hole,"manual_table_polygon":table_polygon.tolist(),
    "registration_parameters":{"cad_samples":1200,"cad_seed":3,"cloud_cap":1600,"cloud_seed":7,"voxel_m":.004,"trim":.6,"coarse":600,"refine":1200,"n_seed":3,"scale":1,"warm_start":False},
    "rectification":{"R1":rig.R1.tolist(),"R2":rig.R2.tolist(),"P1":rig.P1.tolist(),"P2":rig.P2.tolist(),"Q":rig.Q.tolist(),"baseline_m":rig.baseline_m},
    "depth_parameters":{"producer":"stereo_depth defaults, StereoSGBM","num_disparities":128,"block_size":5,"min_valid_px":400,"valid_fraction":float(valid.mean())},
    "environment":{"python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,"torch":torch.__version__,"opencv":cv2.__version__,"trimesh":trimesh.__version__,"device":"cpu","dtype":"float64","torch_threads":1},
    "video_container_frame_counts":counts,
    "control_script_sha256_lf":hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),
    "registration_source_sha256_lf":hashlib.sha256((REPO/'scripts/track3/register_poses.py').read_text().encode()).hexdigest(),
    "stereo_source_sha256_lf":hashlib.sha256((REPO/'reconstruction/modules/v2d_stereo_depth/lib/stereo_depth.py').read_text().encode()).hexdigest(),
    "annotation_sha256":hashlib.sha256((evidence/'episode39_mask_annotations.json').read_bytes()).hexdigest(),
    "input_manifest_sha256":hashlib.sha256((evidence/'episode39_input_manifest.json').read_bytes()).hexdigest(),
    "outputs":[{"path":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(HERE.glob('*.png'))],
    "results":results,"limits":["One real RGB frame; masks manually outlined for observation controls, not model-generated.",
        "SGBM depth has no independent depth truth.","Bounded CPU cloud/cad samples, not a native GPU or episode benchmark.",
        "No original team-run input masks/depth/poses or generator available.","No world camera trajectory or official pose accuracy measured."]}
(HERE / "mask_control_results.json").write_text(json.dumps(report,indent=2,allow_nan=False))
