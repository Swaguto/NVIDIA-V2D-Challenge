"""Controlled video-mask tracking from one reviewed RGB box; no GT poses."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--models", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--frames", type=int, default=319, help="bounded control; at most 319")
args = parser.parse_args()
if not 1 <= args.frames <= 319:
    parser.error("--frames must be between 1 and 319")
REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "docs/team/issue5/evidence"
DATA, OUT = args.data, args.out
MODEL = args.models / "facebook--sam2.1-hiera-large/665f8e2ad61cf5f53d65644ff27c8ee525124610"
os.environ["V2D_PUBLIC"] = str(DATA)
os.environ["V2D_CALIB"] = str(DATA / "meta/camera_calibration.json")
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(REPO / "scripts/track3"))
import cv2
import numpy as np
import torch
import transformers
from transformers import Sam2VideoModel, Sam2VideoProcessor
from register_poses import load_rig

if OUT.exists() and any(OUT.iterdir()):
    raise ValueError("output must be fresh")
manifest = json.loads((HERE/"episode39_models.json").read_text())
for row in manifest:
    if row["model"] != "facebook/sam2.1-hiera-large":
        continue
    path = MODEL / row["file"]
    if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("pinned SAM2 model changed")
expected_inputs = json.loads((HERE/"episode39_input_manifest.json").read_text())
for row in expected_inputs["files"]:
    path = DATA / row["path"]
    if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("pinned control input changed")
OUT.mkdir(parents=True)
torch.set_num_threads(4)
model = Sam2VideoModel.from_pretrained(MODEL).to("cuda").eval()
processor = Sam2VideoProcessor.from_pretrained(MODEL)
session = processor.init_video_session(inference_device="cuda", inference_state_device="cpu")
rig = load_rig()
source = DATA / "videos/chunk-000/observation.images.ego_cam_c/episode_000039.mp4"
capture = cv2.VideoCapture(str(source))
annotations = json.loads((REPO / "docs/team/issue5/evidence/episode39_mask_annotations.json").read_text())
polygon = np.array(annotations["pan_polygon"])
box = [*polygon.min(0).tolist(), *polygon.max(0).tolist()]
rows = []
started = time.perf_counter()
with torch.inference_mode():
    for frame in range(args.frames):
        ok, raw = capture.read()
        if not ok:
            raise RuntimeError(f"missing source frame {frame}")
        rectified = cv2.remap(raw, *rig._maps_l, cv2.INTER_LINEAR)
        inputs = processor(images=cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB), return_tensors="pt").to("cuda")
        if frame == 0:
            processor.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=1,
                input_boxes=[[box]], original_size=inputs.original_sizes[0])
        output = model(inference_session=session, frame=inputs.pixel_values[0])
        mask = processor.post_process_masks([output.pred_masks], original_sizes=inputs.original_sizes,
                                            binarize=True)[0][0,0].cpu().numpy().astype(bool)
        prefix = OUT / f"{frame:06d}"
        mask_path = Path(str(prefix) + "_mask.png")
        cv2.imwrite(str(mask_path), mask.astype(np.uint8)*255)
        row = {"frame_index":frame,"mask_pixels":int(mask.sum()),"mask_sha256":hashlib.sha256(mask_path.read_bytes()).hexdigest()}
        if frame == 0:
            manual = np.zeros(mask.shape,np.uint8)
            cv2.fillPoly(manual,[polygon.astype(np.int32)],1)
            cv2.ellipse(manual,tuple(annotations["handle_hole"]["center"]),tuple(annotations["handle_hole"]["radii"]),0,0,360,0,-1)
            manual = manual > 0
            row["manual_pan_iou"] = float((manual&mask).sum()/max((manual|mask).sum(),1))
        if frame in [0,39,79,119,159,199,239,318]:
            overlay = rectified.copy()
            overlay[mask] = np.uint8(.65*overlay[mask]+.35*np.array([0,255,0]))
            cv2.putText(overlay, f"frame {frame}: SAM2 video, reviewed first box", (15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            cv2.imwrite(str(prefix)+"_overlay.png", overlay)
            print(json.dumps(row),flush=True)
        rows.append(row)
        if frame%20 == 0:
            (OUT/"tracking_results.partial.json").write_text(json.dumps({"frames":rows},indent=2))
            print(f"tracked {frame+1}/{args.frames} frames in {time.perf_counter()-started:.1f}s",flush=True)
capture.release()
report={"episode":39,"frames":rows,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
    "initial_box_xyxy":box,"initial_box_producer":"bounds of the manually reviewed frame-0 RGB polygon; no GT pose",
    "model":"facebook/sam2.1-hiera-large","model_revision":"665f8e2ad61cf5f53d65644ff27c8ee525124610",
    "model_weights_sha256":hashlib.sha256((MODEL/"model.safetensors").read_bytes()).hexdigest(),
    "source_script_sha256_lf":hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),
    "environment":{"torch":torch.__version__,"transformers":transformers.__version__,"cuda":torch.version.cuda,
                   "device":torch.cuda.get_device_name()},"seconds":time.perf_counter()-started,
    "limits":["One manually reviewed RGB box at frame zero; not fully automatic target acquisition.",
              "No later manual corrections or GT poses supplied.","Only mask propagation; pose/world accuracy not measured."]}
record={"schema_version":1,"episode":39,"objects":["mini_dust_pan"],"camera_frame":rig.left,"coordinate_frame":"rectified_camera","resolution":list(rig.size),"gt_use":"none","calibration_sha256":hashlib.sha256((DATA/"meta/camera_calibration.json").read_bytes()).hexdigest(),"rectification":{k:getattr(rig,k).tolist() for k in ["R1","R2","P1","P2","Q"]},"producer":{k:report[k] for k in ["model","model_revision","model_weights_sha256","initial_box_xyxy","initial_box_producer","source_script_sha256_lf","environment","limits"]},"frames":{str(r["frame_index"]):{"mini_dust_pan":{"path":str(r["frame_index"]).zfill(6)+"_mask.png","sha256":r["mask_sha256"]}} for r in rows}}
(OUT/"manifest.json").write_text(json.dumps(record,indent=2))
(OUT/"tracking_results.json").write_text(json.dumps(report,indent=2))
print("Completed reviewed-first-box video control.",flush=True)
