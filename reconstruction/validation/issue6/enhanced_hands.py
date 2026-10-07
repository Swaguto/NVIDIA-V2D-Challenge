"""Select the wearer from actual stereo detections, then run pinned HaMeR."""

import argparse
import copy
from dataclasses import asdict
import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import shutil
import sys

import cv2
import numpy as np
from wearer_selection import SelectionSettings, WearerSelector, candidate_depth


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2) + "\n")


def detect(manifest, source, out, paths):
    from ultralytics import YOLO

    detector = YOLO("/workspace/pr32-models/detector.pt").to("cuda")
    settings = SelectionSettings()
    selector = WearerSelector(
        manifest["color_camera"]["width"], manifest["color_camera"]["height"], settings
    )
    raw, audit = [], []
    for index, (row, path) in enumerate(zip(manifest["frames"], paths)):
        image = cv2.imread(str(path))
        depth = np.load(source / row["depth_path"], allow_pickle=False)
        result = detector(image, conf=0.5, verbose=False)[0]
        candidates = []
        for box, confidence, label in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            distance, pixels, fraction = candidate_depth(box, depth, settings)
            candidates.append(
                {
                    "box_xyxy": box.tolist(),
                    "confidence": float(confidence),
                    "class": int(label),
                    "depth_m": distance,
                    "depth_pixels": pixels,
                    "depth_valid_fraction": fraction,
                }
            )
        selected = selector.select(index, candidates)
        entry = {
            "frame_idx": index,
            "img_path": str(path),
            "left_bbox": None,
            "right_bbox": None,
            "left_keypoints": None,
            "right_keypoints": None,
            "left_conf": 0.0,
            "right_conf": 0.0,
        }
        for side in ("left", "right"):
            chosen = selected[side]["candidate_index"]
            if chosen is None:
                continue
            candidate = candidates[chosen]
            box = np.array(candidate["box_xyxy"])
            center = (box[:2] + box[2:]) / 2
            size = max(box[2:] - box[:2]) * 1.2
            entry[side + "_bbox"] = np.clip(
                np.r_[center - size / 2, center + size / 2],
                0,
                [image.shape[1] - 1, image.shape[0] - 1] * 2,
            )
            entry[side + "_conf"] = candidate["confidence"]
        raw.append(entry)
        audit.append(
            {"frame_index": row["id"], "candidates": candidates, "selected": selected}
        )
        if index % 50 == 0:
            print(
                f"Episode {manifest['episode']}: wearer detections {index}/{len(paths)}",
                flush=True,
            )
    write(out / "raw_detections.json", raw)
    write(
        out / "wearer_candidates.json",
        {
            "episode": manifest["episode"],
            "frames": audit,
            "settings": asdict(settings),
            "manual_annotations": False,
            "reference_poses_used": False,
            "algorithm": "nearby lower-image bootstrap followed by measured-depth, motion and size continuity; ambiguous frames unknown",
            "visual_acceptance": "pending",
        },
    )
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--detections-only", action="store_true")
    parser.add_argument("--use-saved-detections", action="store_true")
    args = parser.parse_args()
    source = Path(f"/workspace/issue6-full-inputs/e{args.episode:03d}")
    out = Path(f"/workspace/temporal-hands/e{args.episode:03d}")
    manifest = json.loads((source / "manifest.json").read_text())
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for row in manifest["frames"]:
        path = source / row["color"]
        destination = frames / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
    paths = [frames / Path(row["color"]).name for row in manifest["frames"]]
    if args.use_saved_detections:
        raw = json.loads((out / "raw_detections.json").read_text())
        for row in raw:
            for side in ("left", "right"):
                if row[f"{side}_bbox"] is not None:
                    row[f"{side}_bbox"] = np.asarray(row[f"{side}_bbox"], dtype=float)
        owner = json.loads((out / "wearer_candidates.json").read_text())
        assert owner["manual_annotations"] is False and len(raw) == len(paths)
        assert [row["frame_index"] for row in owner["frames"]] == [
            row["id"] for row in manifest["frames"]
        ]
    else:
        if (out / "raw_detections.json").exists():
            raise ValueError(
                "Existing detections must be explicitly reused or preserved in another run directory"
            )
        raw = detect(manifest, source, out, paths)
    if args.detections_only:
        return
    fork = Path("/workspace/Dyn-HaMR/third-party/hamer")
    sys.path.insert(0, str(fork))
    spec = importlib.util.spec_from_file_location("enhanced_hamer", fork / "run.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    cleaned = upstream.clean_bbox_sequences(copy.deepcopy(raw))
    write(out / "cleaned_detections.json", cleaned)
    model, cfg = upstream.load_hamer("/workspace/hamer-weights")
    model = model.to("cuda").eval()
    renderer = upstream.Renderer(cfg, faces=model.mano.faces)
    options = SimpleNamespace(
        rescale_factor=1.3, render=False, res_folder=str(out / "predictions.pkl")
    )
    predictions = upstream.run_hamer_on_cleaned_bboxes(
        cleaned, model, cfg, renderer, options
    )
    with (out / "predictions.pkl").open("wb") as stream:
        pickle.dump(predictions, stream)
    sys.path.insert(0, "/workspace/Dyn-HaMR/dyn-hamr")
    from preproc.export_hamer import export_sequence_results

    export_sequence_results(
        str(out / "predictions.pkl"), str(out / "tracks"), str(out / "shots.json")
    )
    print(
        json.dumps(
            {
                "episode": args.episode,
                "frames": len(paths),
                "left_raw": sum(row["left_bbox"] is not None for row in raw),
                "right_raw": sum(row["right_bbox"] is not None for row in raw),
                "manual_annotations": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
