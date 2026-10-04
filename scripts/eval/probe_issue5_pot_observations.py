"""Frozen public ep2/ep0 pot/lid observation diagnostic, without pose GT."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--models', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
REPO = Path(__file__).resolve().parents[2]
DATA, OUT, MODELS = args.data, args.out, args.models
os.environ["V2D_PUBLIC"] = str(DATA)
os.environ["V2D_CALIB"] = str(DATA / "meta/camera_calibration.json")
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(REPO / "scripts/track3"))
sys.path.insert(0, str(REPO / "reconstruction/modules/v2d_world_calib"))
import cv2
import numpy as np
import torch
import transformers
import register_poses as registration
from v2d.world_calib.vm.fit_ego_rig import load_mesh
from v2d.world_calib.vm.visualize_object_pose import render_depth


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


require(not OUT.exists(), "output must be fresh")
expected = json.loads((REPO / "docs/team/issue5/evidence/asset_download_manifest.json").read_text())
selected = [x for x in expected["files"] if x["path"].startswith("videos/") or
            x["path"].startswith("mesh/white_pot/") or x["path"].startswith("mesh/white_pot_lid/")]
for row in selected:
    path = DATA / row["path"]
    require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "input changed: " + str(path))
calib = DATA / "meta/camera_calibration.json"
require(sha(calib) == sha(REPO / "tests/fixtures/track3_camera_calibration.json"), "calibration changed")
models = json.loads((REPO / "docs/team/issue5/evidence/episode39_models.json").read_text())
for row in models:
    path = MODELS / row["model"].replace("/", "--") / row["revision"] / row["file"]
    require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "model changed")
require(torch.cuda.is_available(), "CUDA unavailable")
OUT.mkdir(parents=True)
registration.GDINO = str(MODELS / "IDEA-Research--grounding-dino-base/12bdfa3120f3e7ec7b434d90674b3396eccf88eb")
registration.SAM2 = str(MODELS / "facebook--sam2.1-hiera-large/665f8e2ad61cf5f53d65644ff27c8ee525124610")
torch.set_num_threads(4)
names = ["white_pot", "white_pot_lid"]
seg = registration.Segmenter(names, device="cuda")
rig = registration.load_rig()
cads = {n: torch.tensor(registration.load_cad(n, 6000, 0), device="cuda", dtype=torch.float32) for n in names}
meshes = {n: load_mesh(DATA / f"mesh/{n}/{n}.glb") for n in names}


class CaptureDetections:
    def __init__(self, processor):
        self.processor = processor
        self.last = None

    def __call__(self, *args, **kwargs):
        return self.processor(*args, **kwargs)

    def post_process_grounded_object_detection(self, *args, **kwargs):
        result = self.processor.post_process_grounded_object_detection(*args, **kwargs)
        self.last = [{"label": label, "score": float(score), "box_xyxy": box}
                     for label, score, box in zip(result[0]["text_labels"], result[0]["scores"].tolist(), result[0]["boxes"].tolist())]
        return result


seg.gp = CaptureDetections(seg.gp)
rows = []
start = time.perf_counter()
report = {"episodes": [2, 0], "objects": names, "gt_use": "none", "prompt": seg.prompt,
          "frame_selection": "eight rounded linspace source indices per full video; no GT read",
          "inputs": selected, "calibration_sha256": sha(calib), "models": models,
          "source_sha256_lf": {str(p.relative_to(REPO)): hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
                                for p in [REPO / "scripts/track3/register_poses.py", REPO / "reconstruction/modules/v2d_world_calib/vm/visualize_object_pose.py"]},
          "worker_sha256_lf": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
          "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                          "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "opencv": cv2.__version__},
          "params": {"box_threshold": .30, "text_threshold": .25, "scale": 1, "cad_samples": 6000, "seed": 0,
                     "max_residual_m": .05, "warm_starts": False, "smoothing": False}, "frames": rows,
          "limits": ["Fixed public diagnostic indices; only eight frames per episode.",
                     "No GT pose, camera trajectory or corrected masks in inference.",
                     "Residual/overlap are fit diagnostics, not pose accuracy.",
                     "Right-view masks only; left SGBM depth/registration, no independent depth truth."]}
for episode in [2, 0]:
    paths = [DATA / f"videos/chunk-000/observation.images.{c}/episode_{episode:06d}.mp4" for c in (rig.left, rig.right)]
    counts = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    require(counts[0] == counts[1] and counts[0] > 0, "stereo frame mismatch")
    indices = np.rint(np.linspace(0, counts[0] - 1, 8)).astype(int)
    for frame in indices:
        frame = int(frame)
        raw = [registration.read_frame(p, frame) for p in paths]
        images = rig.rectify_pair(*raw)
        depth, valid = registration.stereo_depth(rig, *raw)
        points = registration.backproject_full(depth, valid, rig.P1)
        row = {"episode": episode, "frame_index": frame, "total_source_frames": counts[0], "views": {}, "fits": {}}
        folder = OUT / f"e{episode:03d}/{frame:06d}"
        folder.mkdir(parents=True)
        np.savez_compressed(folder / "depth.npz", depth=depth, valid=valid)
        row["depth_sha256"] = sha(folder / "depth.npz")
        for camera, image in zip((rig.left, rig.right), images):
            masks, assignments = seg(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), .30)
            view = {"detections": seg.gp.last, "assignments": assignments,
                    "candidates": seg.last_candidates, "objects": {}}
            row["views"][camera] = view
            overlay = image.copy()
            colors = [(0, 255, 0), (255, 0, 255)]
            for bi, name in enumerate(names):
                mask = masks[bi]
                path = folder / f"{camera}_{name}_mask.png"
                require(cv2.imwrite(str(path), mask.astype(np.uint8) * 255), "mask save failed")
                overlay[mask] = np.uint8(.65 * overlay[mask] + .35 * np.array(colors[bi]))
                detail = {"mask_pixels": int(mask.sum()), "mask_sha256": sha(path)}
                view["objects"][name] = detail
                if camera != rig.left:
                    continue
                masked = mask.ravel() & valid.ravel()
                detail["valid_depth_pixels"] = int(masked.sum())
                fit = {"status": "rejected", "reason": "insufficient_masked_depth"}
                row["fits"][name] = fit
                if masked.sum() < 300:
                    continue
                cloud = registration.trim_cloud(registration.voxel_downsample(points[masked], .004))
                fit["cloud_points"] = len(cloud)
                fit["cloud_extent_m"] = np.ptp(cloud, axis=0).tolist()
                if len(cloud) < 120:
                    fit["reason"] = "insufficient_filtered_cloud"
                    continue
                result = registration.register(cads[name], torch.tensor(cloud, device="cuda", dtype=torch.float32))
                require(result is not None, "registration returned no result")
                scale, rotation, translation, residual = result
                fit.update(scale=float(scale), rotation_wxyz=registration._mat_to_wxyz(rotation.cpu().numpy()).tolist(),
                           translation=translation.cpu().numpy().tolist(), residual_m=float(residual))
                pose = np.eye(4)
                pose[:3, :3] = rotation.cpu().numpy()
                pose[:3, 3] = translation.cpu().numpy()
                mesh = meshes[name]
                vertices = np.asarray(mesh.vertices) @ pose[:3, :3].T + pose[:3, 3]
                rendered = render_depth(vertices, mesh.faces, rig.P1[:3, :3], image.shape[:2])
                silhouette = np.isfinite(rendered) & (rendered > 0)
                fit["cad_model_mask_iou"] = float((mask & silhouette).sum() / max((mask | silhouette).sum(), 1))
                fit.update(status="observed" if np.isfinite(residual) and residual <= .05 else "rejected",
                           reason=None if np.isfinite(residual) and residual <= .05 else "residual_gate")
                contour = cv2.findContours(silhouette.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(overlay, contour, -1, (0, 255, 255), 2)
            cv2.putText(overlay, f"ep {episode} frame {frame} {camera}: green pot, magenta lid, yellow CAD", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
            require(cv2.imwrite(str(folder / f"{camera}_overlay.jpg"), overlay), "overlay save failed")
        rows.append(row)
        report["elapsed_s"] = time.perf_counter() - start
        (OUT / "probe_results.partial.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"episode": episode, "frame": frame, "fits": row["fits"],
                          "mask_pixels": {cam: {n: x["mask_pixels"] for n,x in v["objects"].items()} for cam,v in row["views"].items()}}), flush=True)
(OUT / "probe_results.json").write_text(json.dumps(report, indent=2))
print("Completed native pot/lid observation control", flush=True)
