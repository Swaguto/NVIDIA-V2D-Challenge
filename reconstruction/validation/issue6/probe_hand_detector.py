"""Compare real detector evidence without replacing frozen fits or selecting a winner.

This development diagnostic runs full source sequences with the existing pinned
two-class detector and unchanged stereo/temporal wearer association. It records
all candidates and rejections. New detections are not accepted hand geometry,
contact evidence, or permission to interpolate an invisible hand.
"""

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import sys
import time

DETECTOR_SHA256 = "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033"
ULTRALYTICS_VERSION = "8.1.34"
HOLDOUTS = {0, 11, 23, 31, 39}
VARIANTS = tuple(
    (f"{name}_conf{int(confidence * 100):02d}", views, confidence)
    for name, views in (
        ("full640", ("full640",)),
        ("full1280", ("full1280",)),
        ("full640_tiles", ("full640", "tile00", "tile01", "tile10", "tile11")),
    )
    for confidence in (0.5, 0.25, 0.1)
)


def source_views(width, height):
    """Fixed overlapping views; no episode labels or selected hand locations."""
    if width < 4 or height < 4:
        raise ValueError("Invalid source dimensions")
    views = {"full640": ((0, 0, width, height), 640),
             "full1280": ((0, 0, width, height), 1280)}
    for x in range(2):
        for y in range(2):
            left = 0 if x == 0 else math.floor(width * 0.4)
            top = 0 if y == 0 else math.floor(height * 0.3)
            right = math.ceil(width * 0.6) if x == 0 else width
            bottom = math.ceil(height * 0.7) if y == 0 else height
            views[f"tile{x}{y}"] = ((left, top, right, bottom), 640)
    return views


def source_box(box, crop):
    """Ultralytics returns input-image coordinates, before our source offset."""
    if len(box) != 4 or any(not math.isfinite(float(x)) for x in box):
        return None
    left, top, right, bottom = crop
    width, height = right - left, bottom - top
    x0, y0, x1, y1 = map(float, box)
    x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
    y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0 + left, y0 + top, x1 + left, y1 + top]


def overlap(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def variant_candidates(candidates, views, confidence):
    """Classwise merge for tiles; independent copies for mutable wearer audits."""
    selected = [copy.deepcopy(row) for row in candidates
                if row["source_view"] in views and row["confidence"] >= confidence]
    if len(views) == 1:
        return selected  # Preserve the detector's native ordering/NMS for controls.
    kept = []
    for row in sorted(selected, key=lambda r: (-r["confidence"], r["source_view"], r["candidate_id"])):
        if not any(row["class"] == old["class"] and overlap(row["box_xyxy"], old["box_xyxy"]) > 0.7 for old in kept):
            kept.append(row)
    return kept


def development_episode(episode):
    if episode in HOLDOUTS:
        raise ValueError("Do not use untuned holdouts for an adaptive detector sweep")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    development_episode(manifest["episode"])
    source = args.source_root.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Use a new output directory separate from the source images/depth")
    weights = args.detector_weights.resolve(strict=True)
    if sha(weights) != DETECTOR_SHA256:
        raise ValueError("Detector weights differ from the pinned actual-run checkpoint")
    frames = manifest["frames"]
    if len(frames) < 2 or [row["id"] for row in frames] != list(range(len(frames))):
        raise ValueError("Preserve the complete contiguous original source axis")
    output.mkdir(parents=True, exist_ok=False)
    camera = manifest["color_camera"]
    views = source_views(camera["width"], camera["height"])
    protocol = {
        "episode": manifest["episode"], "status": "prepared", "quality_acceptance": False,
        "source_manifest_sha256": sha(manifest_path), "detector_weights_sha256": DETECTOR_SHA256,
        "ultralytics_required_version": ULTRALYTICS_VERSION,
        "probe_source_sha256": sha(Path(__file__)), "views": views,
        "variants": [{"name": name, "source_views": names, "confidence": conf} for name, names, conf in VARIANTS],
        "inference": {"minimum_confidence": 0.1, "iou": 0.7, "max_det": 300, "augment": False, "device": args.device},
        "control_definition": "full640_conf50 filters the detector's native 0.1-confidence NMS output at 0.5; it is a new runtime control, not the saved original detections",
        "no_winner_selected": True, "manual_regions": False, "raw_fits_modified": False,
        "scope": "Detection-only development sweep. Association, depth/identity gates and quality thresholds are unchanged. No MANO assets or contact pipeline are used.",
    }
    write(output / "protocol.json", protocol)
    try:
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        from wearer_selection import SelectionSettings, WearerSelector, candidate_depth

        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise ValueError("Run this real-image diagnostic on the prepared cloud CUDA worker")
        if importlib.metadata.version("ultralytics") != ULTRALYTICS_VERSION:
            raise ValueError("Use the original actual-run Ultralytics 8.1.34 detector implementation")
        torch.set_num_threads(4)
        detector = YOLO(str(weights)).to(args.device)
        if set(detector.names) != {0, 1}:
            raise ValueError("The pinned detector must retain the original left=0/right=1 classes")
        settings = SelectionSettings()
        selectors = {name: WearerSelector(camera["width"], camera["height"], settings) for name, _, _ in VARIANTS}
        protocol.update(status="running", wearer_settings=asdict(settings),
                        wearer_source_sha256=sha(Path(__file__).with_name("wearer_selection.py")),
                        runtime={"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                                 "ultralytics": importlib.metadata.version("ultralytics"),
                                 "opencv": cv2.__version__, "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(args.device)}, detector_names=detector.names)
        write(output / "protocol.json", protocol)
        observations = {name: {"left": [], "right": []} for name, _, _ in VARIANTS}
        started = time.monotonic()
        with (output / "frame-candidates-and-selections.jsonl").open("x", encoding="utf-8") as stream:
            for row in frames:
                image_path, depth_path = (source / row["color"]).resolve(strict=True), (source / row["depth_path"]).resolve(strict=True)
                if not image_path.is_relative_to(source) or not depth_path.is_relative_to(source):
                    raise ValueError("Source references must remain inside the verified episode input root")
                image_bytes, depth_bytes = image_path.read_bytes(), depth_path.read_bytes()
                image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                depth = np.load(io.BytesIO(depth_bytes), allow_pickle=False)
                if image is None or image.shape[:2] != (camera["height"], camera["width"]) or depth.shape != image.shape[:2]:
                    raise ValueError("Source image and calibrated stereo depth dimensions differ")
                candidates, rejected = [], []
                for view, (crop, size) in views.items():
                    x0, y0, x1, y1 = crop
                    prediction = detector.predict(image[y0:y1, x0:x1], conf=0.1, iou=0.7,
                                                  imgsz=size, max_det=300, augment=False, verbose=False)[0]
                    boxes = prediction.boxes
                    for local_index, (box, confidence, label) in enumerate(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())):
                        mapped = source_box(box, crop)
                        candidate = {"candidate_id": f"{view}:{local_index}", "source_view": view,
                                     "detector_box_view_xyxy": box, "confidence": confidence, "class": int(label)}
                        if mapped is None:
                            rejected.append(dict(candidate, reason="invalid_or_empty_source_box"))
                            continue
                        if int(label) not in (0, 1) or not math.isfinite(confidence):
                            raise ValueError("Unexpected detector class/confidence")
                        distance, pixels, fraction = candidate_depth(mapped, depth, settings)
                        candidates.append(dict(candidate, box_xyxy=mapped, depth_m=distance,
                                               depth_pixels=pixels, depth_valid_fraction=fraction))
                variants = {}
                for name, names, confidence in VARIANTS:
                    chosen_candidates = variant_candidates(candidates, names, confidence)
                    selected = selectors[name].select(row["id"], chosen_candidates)
                    variants[name] = {"candidates": chosen_candidates, "selected": selected}
                    for hand in ("left", "right"):
                        if selected[hand]["candidate_index"] is not None:
                            observations[name][hand].append(row["id"])
                record = {"frame_id": row["id"], "timestamp_s": row["timestamp_s"],
                          "source_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                          "source_depth_sha256": hashlib.sha256(depth_bytes).hexdigest(),
                          "all_candidates": candidates, "rejected_detector_boxes": rejected, "variants": variants}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if row["id"] % 25 == 0:
                    print(json.dumps({"episode": manifest["episode"], "completed_frames": row["id"] + 1,
                                      "source_frames": len(frames)}), flush=True)
        receipt = {"episode": manifest["episode"], "status": "detector_sweep_complete_requires_image_and_identity_review",
                   "source_frames": len(frames), "seconds": time.monotonic() - started,
                   "protocol_sha256": sha(output / "protocol.json"),
                   "frame_evidence_sha256": sha(output / "frame-candidates-and-selections.jsonl"),
                   "direct_selected_frames": observations, "no_winner_selected": True, "quality_acceptance": False}
        write(output / "receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
    except Exception as error:
        write(output / "failure.json", {"status": "failed_preserved_partial_diagnostic", "error": repr(error),
                                         "quality_acceptance": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
