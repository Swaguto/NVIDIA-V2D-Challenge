#!/usr/bin/env python3
"""Propagate Track 3 object masks with SAM2 video memory.

Grounding DINO finds an object box in an episode frame. SAM2 propagates it in
both directions. Output masks use rectified left camera coordinates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    GroundingDinoForObjectDetection,
    Sam2VideoModel,
    Sam2VideoProcessor,
)

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, os.environ.get("V2D_RUNTIME", str(ROOT / "reconstruction/modules/v2d_stereo_depth/lib")))
from stereo_depth import load_rig  # noqa: E402

from object_prompts import (  # noqa: E402
    SEMANTIC_PROMPTS,
    assign_label,
    detector_box_limit,
    numbered_groups,
    numbered_instance_seeds,
)
from instance_mask_quality import numbered_mask_conflicts  # noqa: E402


def seed_boxes(frames: list[np.ndarray], names: list[str], threshold: float,
               device: str, stride: int, max_box_area: float, semantic_frames=None,
               canonical_box_limits: bool = False):
    prompt = ". ".join(dict.fromkeys(SEMANTIC_PROMPTS.get(name, name.replace("_", " ")) for name in names)) + "."
    revision = os.environ.get("V2D_DINO_REVISION")
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base", revision=revision)
    model = GroundingDinoForObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base", revision=revision).to(device).eval()
    best: dict[int, tuple[float, list[float], str, int]] = {}
    groups = numbered_groups(names)
    grouped = {index for group in groups for index in group}
    locked = set()
    inspected = sorted(set(range(0, len(frames), stride)) | {len(frames) - 1})
    with torch.inference_mode():
        for frame in inspected:
            rgb = semantic_frames[frame] if semantic_frames is not None else frames[frame]
            inputs = processor(images=rgb, text=prompt, return_tensors="pt").to(device)
            result = processor.post_process_grounded_object_detection(
                model(**inputs), inputs.input_ids, threshold=threshold,
                text_threshold=0.25, target_sizes=[rgb.shape[:2]])[0]
            detections = []
            for label, score, box in zip(result["text_labels"], result["scores"].tolist(),
                                         result["boxes"].tolist()):
                index = assign_label(label, names)
                x0, y0, x1, y1 = box
                area_fraction = max(0, x1 - x0) * max(0, y1 - y0) / (rgb.shape[0] * rgb.shape[1])
                limit = detector_box_limit(names[index] if index is not None else None,
                                           max_box_area, canonical_box_limits)
                print(json.dumps(dict(detector_frame=frame, detector_label=label,
                                      score=float(score), assigned_index=index,
                                      area_fraction=area_fraction,
                                      box_xyxy=box, maximum_box_area_fraction=limit,
                                      accepted_box_size=0.0002 <= area_fraction <= limit)), flush=True)
                if not 0.0002 <= area_fraction <= limit:
                    continue
                detections.append(dict(label=label, score=float(score), box=[float(value) for value in box]))
                if index is None or index in grouped:
                    continue
                # Prefer early, high-confidence seeds. The size gate excludes
                # the round table that Grounding DINO calls a white pot.
                rank = score - 0.00015 * frame
                if index not in best or rank > best[index][0] - 0.00015 * best[index][3]:
                    best[index] = (float(score), [float(value) for value in box], label, frame)
            for group in groups:
                if set(group).issubset(locked):
                    continue
                joint = numbered_instance_seeds(detections, names, group)
                if not joint:
                    continue
                for index, detection in joint.items():
                    label = detection['label']
                    if detection['identity_policy'] == 'inferred_initial_left_to_right':
                        label = 'inferred_initial_left_to_right:' + names[index] + ':' + label
                    best[index] = (detection['score'], detection['box'], label, frame)
                locked.update(group)

    del model, processor
    torch.cuda.empty_cache()
    return best


def load_rectified_frames(path: Path, rig) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        left, _ = rig.rectify_pair(frame, frame)
        frames.append(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"empty video: {path}")
    return frames


def suppress_lid_overlap(out: Path, names: list[str], frames: int) -> dict[str, list[int]]:
    """Leave the pot unobserved if its mask has latched onto the lid."""
    suppressed: dict[str, list[int]] = {}
    for lid in (name for name in names if name.endswith("_lid")):
        body = lid.removesuffix("_lid")
        if body not in names:
            continue
        suppressed[body] = []
        for frame in range(frames):
            body_path = out / body / f"{frame:06d}.png"
            lid_path = out / lid / f"{frame:06d}.png"
            if not body_path.is_file() or not lid_path.is_file():
                continue
            body_mask = cv2.imread(str(body_path), cv2.IMREAD_GRAYSCALE) > 127
            lid_mask = cv2.imread(str(lid_path), cv2.IMREAD_GRAYSCALE) > 127
            overlap = int((body_mask & lid_mask).sum())
            smaller = min(int(body_mask.sum()), int(lid_mask.sum()))
            if smaller and overlap / smaller > 0.70:
                body_path.unlink()
                suppressed[body].append(frame)
    return suppressed


def suppress_numbered_overlap(out: Path, names: list[str], frames: int):
    """Retain ambiguous raw masks, but exclude them from measured pose fitting."""
    indices = {index for group in numbered_groups(names) for index in group}
    suppressed = {names[index]: [] for index in indices}
    for frame in range(frames) if indices else []:
        masks = {}
        for index in indices:
            path = out / names[index] / f"{frame:06d}.png"
            if path.is_file():
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Unreadable instance mask: {path}")
                masks[names[index]] = mask > 127
        for index in numbered_mask_conflicts(names, masks):
            name = names[index]
            source = out / name / f"{frame:06d}.png"
            target = out / "ambiguous-instance-masks" / name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            suppressed[name].append(frame)
    return suppressed


def run(episode: int, names: list[str], data: Path, out: Path, threshold: float,
        max_area: float, device: str, seed_stride: int, max_box_area: float,
        seed_only: bool = False, seed_json: Path | None = None,
        semantic_color: bool = False, canonical_box_limits: bool = False) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output must be a fresh directory: {out}")
    rig = load_rig()
    video = data / "videos/chunk-000" / f"observation.images.{rig.left}" / f"episode_{episode:06d}.mp4"
    frames = load_rectified_frames(video, rig)
    supplied = json.loads(seed_json.read_text()) if seed_json is not None else {}
    unknown = set(supplied) - set(names)
    if unknown:
        raise ValueError(f"seed JSON has unknown objects: {sorted(unknown)}")
    colour_frames = None
    if semantic_color:
        from rgb_stereo import seed_frames
        inspected = sorted(set(range(0, len(frames), seed_stride)) | {len(frames) - 1})
        colour_frames = seed_frames(data, episode, rig, inspected)
    seeds = {} if len(supplied) == len(names) else seed_boxes(
        frames, names, threshold, device, seed_stride, max_box_area, colour_frames,
        canonical_box_limits)
    for name, entry in supplied.items():
        frame = int(entry["frame"])
        box = [float(value) for value in entry["box_xyxy"]]
        if frame < 0 or frame >= len(frames) or len(box) != 4:
            raise ValueError(f"invalid seed for {name}")
        seeds[names.index(name)] = (1.0, box, f"manual:{name}", frame)
    if not seeds:
        raise RuntimeError(f"no seed boxes for episode {episode}")
    print(f"episode {episode}: {len(frames)} frames, seeds " +
          str({names[i]: entry for i, entry in seeds.items()}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    detector_rules = {
        "confidence_threshold": threshold,
        "canonical_box_limits": canonical_box_limits,
        "maximum_box_area_fraction": {
            name: detector_box_limit(name, max_box_area, canonical_box_limits) for name in names
        },
        "maximum_mask_area_fraction": max_area,
    }
    if seed_only:
        report = {"episode": episode, "objects": names, "frames": len(frames),
                  "seeds": {names[i]: {"score": seeds[i][0], "box_xyxy": seeds[i][1],
                             "label": seeds[i][2], "frame": seeds[i][3]} for i in seeds},
                  "gt_use": "none", "manual_annotations_used": bool(supplied),
                  "detector_rules": detector_rules}
        (out / "seed_report.json").write_text(json.dumps(report, indent=2))
        return report

    revision = os.environ.get("V2D_SAM2_REVISION")
    processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-large", revision=revision)
    model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-large", revision=revision).to(device).eval()
    counts = {name: 0 for name in names}
    rejected = {name: [] for name in names}
    height, width = frames[0].shape[:2]
    def write_results(results, index: int):
        for result in results:
            frame = int(result.frame_idx)
            logits = processor.post_process_masks(
                [result.pred_masks], original_sizes=[[height, width]], binarize=False)[0]
            mask = np.asarray(logits.float().cpu()).reshape(height, width) > 0
            name = names[index]
            area = int(mask.sum())
            if area < 100 or area > max_area * height * width:
                rejected[name].append(frame)
                continue
            path = out / name / f"{frame:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not path.exists()
            if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
                raise OSError(path)
            counts[name] += int(is_new)
            if frame % 50 == 0:
                print(f"  masks at {frame}/{len(frames)}: {counts}", flush=True)
    with torch.inference_mode():
        for index in sorted(seeds):
            frame = seeds[index][3]
            session = processor.init_video_session(video=frames, inference_device=device,
                                                   video_storage_device="cpu")
            processor.add_inputs_to_inference_session(
                session, frame_idx=frame, obj_ids=[index + 1],
                input_boxes=[[seeds[index][1]]])
            model(inference_session=session, frame_idx=frame)
            write_results(model.propagate_in_video_iterator(session, start_frame_idx=frame), index)
            if frame > 0:
                write_results(model.propagate_in_video_iterator(
                    session, start_frame_idx=frame, reverse=True), index)
            del session
    suppressed = suppress_lid_overlap(out, names, len(frames))
    numbered_suppressed = suppress_numbered_overlap(out, names, len(frames))
    counts = {name: len(list((out / name).glob("*.png"))) for name in names}
    report = {"episode": episode, "objects": names, "frames": len(frames),
              "seeds": {names[i]: {"score": seeds[i][0],
              "box_xyxy": seeds[i][1], "label": seeds[i][2], "frame": seeds[i][3]} for i in seeds},
              "mask_counts": counts, "rejected_frames": rejected,
              "suppressed_identity_switch_frames": suppressed,
              "ambiguous_numbered_overlap_frames": numbered_suppressed,
              "ambiguous_numbered_raw_masks": "ambiguous-instance-masks",
              "numbered_overlap_intersection_over_smaller_threshold": 0.70,
              "numbered_mask_quality_policy": "shared-foreground-overlap-unknown-v1",
              "coordinate_frame": "rectified_camera", "camera_frame": rig.left,
              "gt_use": "none", "model": "facebook/sam2.1-hiera-large"}
    report["model_revisions"] = {"grounding_dino": os.environ.get("V2D_DINO_REVISION"),
                                 "sam2": os.environ.get("V2D_SAM2_REVISION")}
    report["numbered_instance_assignment"] = {
        names[index]: {"policy": "inferred_initial_left_to_right", "canonical_identity_certified": False}
        for index in seeds if seeds[index][2].startswith("inferred_initial_left_to_right:")
    }
    report["unseeded_objects"] = [name for index, name in enumerate(names) if index not in seeds]
    report["detector_rules"] = detector_rules
    report["manual_annotations_used"] = bool(supplied)
    report["semantic_seed_camera"] = "calibrated ego_cam_a colour" if semantic_color else "rectified monochrome stereo"
    report["semantic_prompts"] = {name: SEMANTIC_PROMPTS.get(name, name.replace("_", " ")) for name in names}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--objects", required=True, help="comma-separated reference slot order")
    parser.add_argument("--data", type=Path, default=Path(os.environ.get(
        "V2D_PUBLIC", "/home/ubuntu/v2d_track3/data/hf/track_3/public")))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--max-area", type=float, default=0.15)
    parser.add_argument("--seed-stride", type=int, default=20)
    parser.add_argument("--max-box-area", type=float, default=0.10)
    parser.add_argument("--canonical-box-limits", action="store_true",
                        help="Allow long-spout watering-can boxes up to 20%%; retain other box and mask limits")
    parser.add_argument("--seed-only", action="store_true")
    parser.add_argument("--semantic-color", action="store_true",
                        help="Use calibrated ego RGB for semantic names and colours, then propagate in stereo")
    parser.add_argument("--seed-json", type=Path,
                        help="optional object to frame/box_xyxy overrides")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    names = [name.strip() for name in args.objects.split(",") if name.strip()]
    if not names or len(set(names)) != len(names):
        parser.error("--objects needs distinct object names")
    report = run(args.episode, names, args.data, args.out, args.box_threshold,
                 args.max_area, args.device, args.seed_stride, args.max_box_area,
                 args.seed_only, args.seed_json, args.semantic_color, args.canonical_box_limits)
    print(json.dumps(report.get("mask_counts", report["seeds"])), flush=True)


if __name__ == "__main__":
    main()
