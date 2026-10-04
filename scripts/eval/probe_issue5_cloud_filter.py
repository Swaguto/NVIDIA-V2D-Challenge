"""Refit frozen camera observations with one selected point-cloud filter, without GT."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--observations", type=Path, required=True)
parser.add_argument("--depth-root", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--filter", choices=["local_density", "radial"], required=True)
args = parser.parse_args()
REPO = Path(__file__).resolve().parents[2]
if args.out.exists():
    parser.error("output must be fresh")
os.environ["V2D_PUBLIC"] = str(args.data)
os.environ["V2D_CALIB"] = str(args.data / "meta/camera_calibration.json")
sys.path.insert(0, str(REPO / "scripts/track3"))
import cv2
import numpy as np
import torch
import register_poses as registration
from v2d.world_calib.vm.fit_ego_rig import load_mesh
from v2d.world_calib.vm.visualize_object_pose import render_depth


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


frozen = args.observations.read_bytes()
baseline = json.loads(frozen)
if (baseline["gt_use"] != "none" or baseline.get("coordinate_frame") != "rectified_camera"
        or baseline.get("camera_frame") != "ego_cam_c"):
    raise ValueError("expected frozen GT-free rectified-left observations")
if sha(args.data / "meta/camera_calibration.json") != baseline["calibration_sha256"]:
    raise ValueError("calibration changed")
for item in baseline["inputs"]:
    if sha(args.data / item["path"]) != item["sha256"]:
        raise ValueError("RGB/CAD input changed")
torch.set_num_threads(4)
rig = registration.load_rig()
names = baseline["objects"]
cads = {name: torch.tensor(registration.load_cad(name, 6000, 0), device="cuda", dtype=torch.float32)
        for name in names}
meshes = {name: load_mesh(args.data / f"mesh/{name}/{name}.glb") for name in names}
report = copy.deepcopy(baseline)
report.update(frames=[], producer="frozen mask/depth refit; selected cloud filter",
              observation_input_sha256=hashlib.sha256(frozen).hexdigest(),
              worker_sha256_lf=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),
              native_source_sha256_lf=hashlib.sha256((REPO / "scripts/track3/register_poses.py").read_text().encode()).hexdigest())
report["params"]["cloud_filter"] = {"method": args.filter,
    "mad_threshold": 3.5 if args.filter == "local_density" else 2.5,
    "iterations": 1 if args.filter == "local_density" else 3,
    "minimum_spacing_ratio": 2.0 if args.filter == "local_density" else None,
    "neighbors": 11 if args.filter == "local_density" else None}
args.out.mkdir(parents=True)
for original in baseline["frames"]:
    episode, frame = original["episode"], original["frame_index"]
    folder = args.out / f"e{episode:03d}/{frame:06d}"
    folder.mkdir(parents=True)
    depth_path = args.depth_root / f"e{episode:03d}/{frame:06d}/depth.npz"
    if sha(depth_path) != original["depth_sha256"]:
        raise ValueError("frozen depth changed")
    loaded = np.load(depth_path)
    depth, valid = loaded["depth"], loaded["valid"]
    points = registration.backproject_full(depth, valid, rig.P1)
    paths = [args.data / f"videos/chunk-000/observation.images.{cam}/episode_{episode:06d}.mp4"
             for cam in (rig.left, rig.right)]
    image = rig.rectify_pair(*[registration.read_frame(p, frame) for p in paths])[0]
    overlay = image.copy()
    row = copy.deepcopy(original)
    row["fits"] = {}
    for i, name in enumerate(names):
        mask_path = args.observations.parent / f"e{episode:03d}/{frame:06d}/{name}_mask.png"
        if sha(mask_path) != row["views"][rig.left]["objects"][name]["mask_sha256"]:
            raise ValueError("frozen mask changed")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
        selected = mask.ravel() & valid.ravel()
        fit = {"status": "rejected", "reason": "insufficient_masked_depth"}
        row["fits"][name] = fit
        if selected.sum() < 300:
            continue
        downsampled = registration.voxel_downsample(points[selected], .004)
        cloud = registration.trim_cloud(downsampled, method=args.filter)
        fit.update(unfiltered_cloud_points=len(downsampled), cloud_points=len(cloud))
        if len(cloud) < 120:
            fit["reason"] = "insufficient_filtered_cloud"
            continue
        scale, rotation, translation, residual = registration.register(
            cads[name], torch.tensor(cloud, device="cuda", dtype=torch.float32))
        R, t = rotation.cpu().numpy(), translation.cpu().numpy()
        fit.update(scale=float(scale), rotation_wxyz=registration._mat_to_wxyz(R).tolist(),
                   translation=t.tolist(), residual_m=float(residual),
                   status="observed" if np.isfinite(residual) and residual <= .05 else "rejected",
                   reason=None if np.isfinite(residual) and residual <= .05 else "residual_gate")
        mesh = meshes[name]
        rendered = render_depth(np.asarray(mesh.vertices) @ R.T + t,
                                mesh.faces, rig.P1[:3, :3], image.shape[:2])
        silhouette = np.isfinite(rendered) & (rendered > 0)
        fit["cad_model_mask_iou"] = float((mask & silhouette).sum() / max(1, (mask | silhouette).sum()))
        overlay[mask] = np.uint8(.65 * overlay[mask] + .35 * np.array([(0, 255, 0), (255, 0, 255)][i]))
        contours = cv2.findContours(silhouette.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    cv2.putText(overlay, f"{args.filter} ep{episode} {frame}: pot green / lid magenta / CAD yellow",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    cv2.imwrite(str(folder / "overlay.jpg"), overlay)
    report["frames"].append(row)
    print(json.dumps({"filter": args.filter, "episode": episode, "frame": frame, "fits": row["fits"]}), flush=True)
    (args.out / "probe_results.partial.json").write_text(json.dumps(report, indent=2))
(args.out / "probe_results.json").write_text(json.dumps(report, indent=2))
if args.observations.read_bytes() != frozen:
    raise ValueError("frozen observations changed during refit")
print("Frozen selected-filter predictions", flush=True)
