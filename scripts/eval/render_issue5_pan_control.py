"""Inspect real sampled native camera poses against propagated masks and stereo."""
from pathlib import Path
import hashlib
import json
import sys
import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data",type=Path,required=True)
parser.add_argument("--masks",type=Path,required=True)
parser.add_argument("--depth-probe",type=Path,required=True)
parser.add_argument("--poses",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"reconstruction/modules/v2d_stereo_depth/lib"))
sys.path.insert(0,str(REPO/"reconstruction/modules/v2d_world_calib"))
from stereo_depth import load_rig
from v2d.world_calib.vm.visualize_object_pose import render_depth
OUT,DATA=args.out,args.data
if OUT.exists() and any(OUT.iterdir()):
    parser.error("--out must be fresh")
OUT.mkdir(parents=True,exist_ok=True)
rig = load_rig(str(DATA / "meta/camera_calibration.json"))
scene = trimesh.load(DATA / "mesh/mini_dust_pan/mini_dust_pan.glb", process=False)
mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
root = args.poses / "e039/ego_object_poses/e039/mini_dust_pan/object_to_cam/ego_cam_c"
results = []
for frame in [0,39,79,119,159,199,239,318]:
    path = root / f"{frame:06d}.json"
    pose = json.loads(path.read_text())
    mask_path = args.masks/f"{frame:06d}_mask.png"
    if (pose.get("frame_index") != frame or pose.get("coordinate_frame") != "rectified_camera"
            or pose.get("camera_frame") != rig.left
            or pose.get("mask_sha256") != hashlib.sha256(mask_path.read_bytes()).hexdigest()):
        raise ValueError("pose and frozen target mask do not match")
    q = np.asarray(pose["rotation"])[[1,2,3,0]]
    camera_vertices = vertices @ Rotation.from_quat(q).as_matrix().T + np.asarray(pose["translation"])
    render = render_depth(camera_vertices, faces, rig.P1[:3,:3], (800,1280))
    silhouette = np.isfinite(render)
    mask = cv2.imread(str(args.masks/f"{frame:06d}_mask.png"),cv2.IMREAD_GRAYSCALE)>0
    stereo = np.load(args.depth_probe/f"{frame:06d}_depth.npz")
    common = mask & silhouette & stereo["valid"]
    signed = stereo["depth"][common] - render[common]
    row = {"frame_index":frame,"silhouette_vs_sam2_mask_iou":float((mask&silhouette).sum()/max((mask|silhouette).sum(),1)),
        "icp_residual_m":pose["residual"],"overlap_valid_depth_pixels":int(common.sum()),
        "median_signed_stereo_minus_front_mesh_m":float(np.median(signed)) if len(signed) else None,
        "median_absolute_depth_residual_m":float(np.median(np.abs(signed))) if len(signed) else None,
        "p90_absolute_depth_residual_m":float(np.percentile(np.abs(signed),90)) if len(signed) else None,
        "pose_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "mask_sha256":hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        "depth_sha256":hashlib.sha256((args.depth_probe/f"{frame:06d}_depth.npz").read_bytes()).hexdigest()}
    video = DATA / "videos/chunk-000/observation.images.ego_cam_c/episode_000039.mp4"
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
    ok, raw = cap.read(); cap.release()
    if not ok:
        raise RuntimeError("missing display frame")
    image = cv2.remap(raw,*rig._maps_l,cv2.INTER_LINEAR)
    image[silhouette] = np.uint8(.6*image[silhouette]+.4*np.array([0,255,255]))
    contours,_ = cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image,contours,-1,(0,255,0),2)
    cv2.putText(image,f"frame {frame} | CAD / SAM2 IoU {row['silhouette_vs_sam2_mask_iou']:.3f}", (15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    cv2.imwrite(str(OUT/f"{frame:06d}.png"),image)
    results.append(row); print(json.dumps(row),flush=True)
report={"episode":39,"frames":results,"limits":["Sampled camera-space fit diagnostics, no official pose accuracy.",
    "Full CAD silhouette versus SAM2 masks; external occlusion is not independently labeled.",
    "Frontmost CAD versus SGBM on common pixels; no independent depth truth."],
    "renderer_source_sha256_lf":hashlib.sha256((REPO/"reconstruction/modules/v2d_world_calib/vm/visualize_object_pose.py").read_text().encode()).hexdigest(),
    "source_script_sha256_lf":hashlib.sha256(Path(__file__).read_text().encode()).hexdigest()}
(OUT/"overlay_results.json").write_text(json.dumps(report,indent=2))
images=[cv2.resize(cv2.imread(str(OUT/f"{r['frame_index']:06d}.png")),(640,400)) for r in results]
cv2.imwrite(str(OUT/"contact_sheet.jpg"),np.vstack([np.hstack(images[i:i+2]) for i in range(0,8,2)]))
