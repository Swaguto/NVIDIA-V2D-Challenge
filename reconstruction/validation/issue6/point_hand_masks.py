"""Repair box-prompt hand masks with automatic frozen wrist/palm projections.

The temporal MANO fits and observation/identity gates are immutable inputs.
This generates image masks, not metric hand geometry or quality acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PALM_JOINTS = (0, 1, 5, 9, 13, 17)


def projected_prompt(hand, camera, bbox, direct, native_record):
    """Only a unique, directly observed same-side frozen fit can prompt SAM2."""
    if direct is not True or bbox is None:
        return None, "not_directly_observed"
    if hand is None or native_record is None:
        return None, "missing_or_ambiguous_frozen_fit"
    if (
        native_record.get("directly_observed") is not True
        or native_record["frame_idx"] != hand["frame_id"]
        or native_record["track_id"] != hand["track_id"]
        or native_record["is_right"] is not hand["is_right"]
        or native_record["mano"] != hand["mano"]
    ):
        return None, "frozen_fit_observation_or_identity_mismatch"
    box = np.asarray(bbox, float)
    joints = np.asarray(hand["joints_camera_m"], float)
    calibration = np.asarray(
        [camera[k] for k in ("fx", "fy", "cx", "cy", "width", "height")], float
    )
    if (
        box.shape != (4,)
        or not np.isfinite(box).all()
        or np.any(box[2:] <= box[:2])
        or joints.shape != (21, 3)
        or not np.isfinite(joints).all()
        or not np.isfinite(calibration).all()
        or min(calibration[:2]) <= 0
        or min(calibration[4:]) <= 0
    ):
        return None, "invalid_frozen_projection_input"
    points = []
    seen = set()
    for joint_id in PALM_JOINTS:
        x, y, z = joints[joint_id]
        if z <= 0:
            continue
        u, v = camera["fx"] * x / z + camera["cx"], camera["fy"] * y / z + camera["cy"]
        if (
            0 <= u < camera["width"]
            and 0 <= v < camera["height"]
            and box[0] <= u <= box[2]
            and box[1] <= v <= box[3]
            and (u, v) not in seen
        ):
            seen.add((u, v))
            points.append({"joint": joint_id, "x": float(u), "y": float(v)})
    if len(points) < 3:
        return None, "insufficient_in_view_same_hand_palm_points"
    return points, None


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if args.out.exists():
        raise ValueError(
            "Use a new output directory; raw masks and previous repairs must be retained"
        )
    manifest = json.loads(args.manifest_lock.read_text())["manifest"]
    hands = json.loads(args.frozen_hands.read_text())
    observed = json.loads(args.observation_gate.read_text())["direct_observation"]
    detections = json.loads(args.cleaned_detections.read_text())
    frames = manifest["frames"]
    c = manifest["color_camera"]
    if len(detections) != len(frames) or any(
        len(observed[s]) != len(frames) for s in ("left", "right")
    ):
        raise ValueError(
            "Observation/detection data must preserve the complete source frame axis"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested local CUDA device is unavailable")
    torch.set_num_threads(4)
    grouped = defaultdict(list)
    native = {}
    hashes = {
        str(p): sha256(p)
        for p in (
            args.manifest_lock,
            args.frozen_hands,
            args.observation_gate,
            args.cleaned_detections,
            args.weights,
        )
    }
    for hand in hands:
        grouped[(hand["frame_id"], hand["is_right"])].append(hand)
    for path in sorted(args.records_dir.glob("*/*.json")):
        record = json.loads(path.read_text())
        key = (record["frame_idx"], record["track_id"])
        if key in native:
            raise ValueError("Duplicate native frozen record")
        native[key] = record
        hashes[str(path)] = sha256(path)
    args.out.mkdir(parents=True)
    for side in ("left", "right"):
        (args.out / side).mkdir()
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(args.weights),
        device=args.device,
        apply_postprocessing=False,
    )
    predictor = SAM2ImagePredictor(model)
    started = time.monotonic()
    receipts = []
    for number, frame in enumerate(frames):
        if detections[number]["frame_idx"] != frame["id"]:
            raise ValueError("Corrected detection frame axis differs from source")
        prompts = {}
        for side, right in (("left", False), ("right", True)):
            candidates = grouped[(frame["id"], right)]
            hand = candidates[0] if len(candidates) == 1 else None
            original = native.get((frame["id"], hand["track_id"])) if hand else None
            prompts[side] = projected_prompt(
                hand,
                c,
                detections[number][f"{side}_bbox"],
                observed[side][number],
                original,
            )
        image_path = args.source_frames / f"{frame['id']:06d}.png"
        hashes[str(image_path)] = sha256(image_path)
        image = np.array(Image.open(image_path).convert("RGB"))
        if image.shape != (c["height"], c["width"], 3):
            raise ValueError("Source image differs from the calibrated pixel grid")
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=args.device,
                dtype=torch.bfloat16,
                enabled=args.device == "cuda",
            ),
        ):
            if any(points for points, _ in prompts.values()):
                predictor.set_image(image)
            for side, (points, reason) in prompts.items():
                score = None
                mask = np.zeros(image.shape[:2], bool)
                if points:
                    coordinates = np.array(
                        [[p["x"], p["y"]] for p in points], np.float32
                    )
                    masks, scores, _ = predictor.predict(
                        point_coords=coordinates,
                        point_labels=np.ones(len(points), np.int32),
                        multimask_output=False,
                    )
                    mask = masks[0] > 0
                    score = float(scores[0])
                path = args.out / side / f"{frame['id']:06d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                receipts.append(
                    {
                        "frame_id": frame["id"],
                        "timestamp_s": frame["timestamp_s"],
                        "hand": side,
                        "points": points,
                        "reason": reason,
                        "mask_pixels": int(mask.sum()),
                        "sam_score": score,
                        "mask_sha256": sha256(path),
                    }
                )
        if number % 16 == 0 or number == len(frames) - 1:
            print(
                json.dumps(
                    {
                        "episode": manifest["episode"],
                        "source_frames_processed": number + 1,
                        "total_source_frames": len(frames),
                        "elapsed_s": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    receipt = {
        "episode": manifest["episode"],
        "method": "independent_sam2_image_wrist_palm_points_from_frozen_native_fit",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "frames": len(frames),
        "elapsed_s": time.monotonic() - started,
        "raw_fits_unchanged": True,
        "manual_prompts": False,
        "observation_identity_gates_preserved": True,
        "quality_acceptance": False,
        "inputs_sha256": hashes,
        "masks": receipts,
    }
    (args.out / "point-mask-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest-lock",
        "frozen-hands",
        "records-dir",
        "source-frames",
        "observation-gate",
        "cleaned-detections",
        "weights",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
