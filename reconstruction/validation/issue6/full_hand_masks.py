"""SAM2 hand masks prompted by actual corrected tracker boxes, with observation gates."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from issue6.hand_masks import export_metric_mask
from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.sam2.lib.video_to_masks import video_to_masks

parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=int, required=True)
args = parser.parse_args()
out = Path(f"/workspace/temporal-hands/e{args.episode:03d}")
manifest = json.loads(
    Path(f"/workspace/issue6-full-inputs/e{args.episode:03d}/manifest.json").read_text()
)
raw = json.loads((out / "raw_detections.json").read_text())
cleaned = json.loads((out / "cleaned_detections.json").read_text())
rate = (len(manifest["frames"]) - 1) / (
    manifest["frames"][-1]["timestamp_s"] - manifest["frames"][0]["timestamp_s"]
)
video = out / "selected.mp4"
if not video.exists():
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-framerate",
            str(rate),
            "-i",
            str(out / "frames/%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    area = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    return float(area / max((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - area, 1))


observed = {}
prompts = []
for side, track in [("left", 2), ("right", 3)]:
    observed[side] = []
    last_prompt = -32
    for number, row in enumerate(cleaned):
        box = row[f"{side}_bbox"]
        # Only the same wearer hand supports this observation after cleaning.
        candidates = (
            [raw[number][f"{side}_bbox"]]
            if raw[number][f"{side}_bbox"] is not None
            else []
        )
        direct = box is not None and any(
            iou(box, candidate) >= 0.25 for candidate in candidates
        )
        observed[side].append(direct)
        if direct and number - last_prompt >= 32:
            prompts.append(
                Sam2Prompt(
                    frame_index=number,
                    object_id=track,
                    box=BoundingBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
                )
            )
            last_prompt = number
    if not any(observed[side]):
        raise RuntimeError(
            f"No directly observed {side} hand for episode {args.episode}"
        )
Sam2Prompts(prompts=prompts).save(str(out / "sam2_prompts.json"))
(out / "hand_tracks.json").write_text(
    json.dumps(
        {
            "tracks": [
                {"object_id": 2, "is_right": False, "role": "hand"},
                {"object_id": 3, "is_right": True, "role": "hand"},
            ]
        },
        indent=2,
    )
    + "\n"
)
video_to_masks(
    str(video),
    str(out / "sam2_prompts.json"),
    str(out / "visual_masks"),
    "/workspace/pr32-models",
)
for side, track in [("left", 2), ("right", 3)]:
    for number, direct in enumerate(observed[side]):
        export_metric_mask(
            out / "visual_masks" / str(track) / f"{number:06d}.png",
            out / "masks" / str(track) / f"{number:06d}.png",
            direct,
            (manifest["color_camera"]["width"], manifest["color_camera"]["height"]),
        )
(out / "observation_gate.json").write_text(
    json.dumps(
        {
            "direct_observation": observed,
            "rule": "Corrected bbox must overlap the same-side automatically selected wearer observation by IoU>=0.25; otherwise mask is empty and contact remains unknown",
            "frames": len(cleaned),
            "prompt_interval_frames": 32,
            "manual_annotations": False,
            "wearer_selection_receipt": str(out / "wearer_candidates.json"),
        },
        indent=2,
    )
    + "\n"
)
print(
    json.dumps(
        {
            "episode": args.episode,
            "left_observed": sum(observed["left"]),
            "right_observed": sum(observed["right"]),
            "prompts": len(prompts),
        },
        indent=2,
    ),
    flush=True,
)
