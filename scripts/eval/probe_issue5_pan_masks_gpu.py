"""Run the repository's actual pinned DINO/SAM2 masks on fixed public frames."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--models", type=Path, required=True, help="cache from fetch_issue5_model_inputs.py")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--prompt", help="single-object prompt control, e.g. dustpan")
args = parser.parse_args()
REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "docs/team/issue5/evidence"
DATA, OUT, MODELS = args.data, args.out, args.models
os.environ["V2D_PUBLIC"] = str(DATA)
os.environ["V2D_CALIB"] = str(DATA / "meta/camera_calibration.json")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(MODELS.parent / "hf")
sys.path.insert(0, str(REPO / "scripts/track3"))
import cv2
import numpy as np
import torch
import transformers
import register_poses as registration

if OUT.exists() and any(OUT.iterdir()):
    raise ValueError("probe output must be fresh")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
manifest = json.loads((HERE / "episode39_models.json").read_text())
for row in manifest:
    path = MODELS / row["model"].replace("/", "--") / row["revision"] / row["file"]
    if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"pinned model changed: {path}")
inputs = json.loads((REPO / "docs/team/issue5/evidence/episode39_input_manifest.json").read_text())
for row in inputs["files"]:
    path = DATA / row["path"]
    if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"pinned input changed: {path}")
OUT.mkdir(parents=True)
registration.GDINO = str(MODELS / "IDEA-Research--grounding-dino-base" / "12bdfa3120f3e7ec7b434d90674b3396eccf88eb")
registration.SAM2 = str(MODELS / "facebook--sam2.1-hiera-large" / "665f8e2ad61cf5f53d65644ff27c8ee525124610")
torch.set_num_threads(4)
seg = registration.Segmenter(["mini_dust_pan"], device="cuda")
if args.prompt:
    phrase = args.prompt.strip().rstrip(".")
    seg.prompt = phrase + "."
    seg.phrase2idx = {phrase.lower(): 0}
    seg._order = [phrase.lower()]
rig = registration.load_rig()


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
for frame in [0, 39, 79, 119, 159, 199, 239, 318]:
    start = time.perf_counter()
    paths = [DATA / f"videos/chunk-000/observation.images.{camera}/episode_000039.mp4"
             for camera in (rig.left, rig.right)]
    images = [registration.read_frame(path, frame) for path in paths]
    rectified, _ = rig.rectify_pair(*images)
    depth, valid = registration.stereo_depth(rig, *images)
    masks, assignments = seg(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB), .30)
    torch.cuda.synchronize()
    target = masks[0]
    row = {"frame_index": frame, "detections": seg.gp.last, "assignments": assignments,
           "mask_pixels": int(target.sum()), "valid_masked_depth_pixels": int((target & valid).sum()),
           "seconds": time.perf_counter() - start}
    if frame == 0:
        annotation = json.loads((HERE/"episode39_mask_annotations.json").read_text())
        pan = np.zeros(target.shape, np.uint8)
        cv2.fillPoly(pan, [np.asarray(annotation["pan_polygon"],np.int32)], 1)
        cv2.ellipse(pan, tuple(annotation["handle_hole"]["center"]),tuple(annotation["handle_hole"]["radii"]),0,0,360,0,-1)
        pan = pan > 0
        row["manual_pan_mask_iou"] = float((target & pan).sum() / max((target | pan).sum(), 1))
    prefix = OUT / f"{frame:06d}"
    cv2.imwrite(str(prefix) + "_mask.png", target.astype(np.uint8)*255)
    np.savez_compressed(str(prefix) + "_depth.npz", depth=depth, valid=valid)
    overlay = rectified.copy()
    overlay[target] = np.uint8(.65*overlay[target] + .35*np.array([0,255,0]))
    cv2.putText(overlay, f"frame {frame}: actual DINO/SAM mask", (20,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
    cv2.imwrite(str(prefix) + "_overlay.png", overlay)
    rows.append(row)
    (OUT / "probe_results.json").write_text(json.dumps({"frames": rows}, indent=2))
    print(json.dumps(row), flush=True)
report = {"episode":39, "frame_selection":"same eight fixed indices used to inspect the provided clip; no GT pose read",
          "camera_frame":rig.left,"coordinate_frame":"rectified_camera","prompt":seg.prompt,
          "box_threshold":.30,"text_threshold":.25,"models":manifest,"inputs":inputs,
          "environment":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,
                         "device":torch.cuda.get_device_name(),"transformers":transformers.__version__,"opencv":cv2.__version__},
          "registration_source_sha256_lf":hashlib.sha256((REPO/"scripts/track3/register_poses.py").read_text().encode()).hexdigest(),
          "probe_source_sha256_lf":hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),
          "frames":rows,"limits":["Masks are actual pretrained model outputs; no GT object poses used.",
              "One manual frame-0 comparison; no independent depth truth or world trajectory.",
              "Pose fitting and full episode coverage not measured by this mask probe."]}
(OUT / "probe_results.json").write_text(json.dumps(report, indent=2))
print("Completed fixed-frame native mask probe.", flush=True)
