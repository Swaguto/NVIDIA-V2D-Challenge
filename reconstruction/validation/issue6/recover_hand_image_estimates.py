"""Recover estimated hand landmarks from two independent SAM2 propagations.

Windows contain a side and two saved, directly observed anchor frames. This
produces optional visual fitting evidence; canonical metric gates are untouched.
Requires the existing pinned HaMeR and SAM2 runtimes and licensed MANO assets.
"""

import argparse
import gc
import importlib.util
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def recover(args):
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    args.output.mkdir(parents=True, exist_ok=False)
    frames = {f["id"]: f for f in json.loads(args.manifest.read_text())["frames"]}
    gate = json.loads(args.gate.read_text())["direct_observation"]
    audit = json.loads(args.candidates.read_text())
    windows = json.loads(args.windows.read_text())
    predictor = SAM2ImagePredictor(
        build_sam2(
            args.sam_config,
            str(args.sam_weights),
            device="cuda",
            apply_postprocessing=False,
        )
    )
    for window in windows:
        side, a, b = window["side"], window["start"], window["end"]
        hand = 0 if side == "left" else 1
        for fid in (a, b):
            if not gate[side][fid]:
                raise ValueError("Mask anchor must be a canonical direct observation")
            kp = np.array(
                json.loads(
                    (
                        args.tracks / f"{hand:03d}" / f"{fid:06d}_keypoints.json"
                    ).read_text()
                )["people"][0]["pose_keypoints_2d"]
            )
            row = audit["frames"][fid]
            box = np.array(
                row["candidates"][row["selected"][side]["candidate_index"]]["box_xyxy"],
                np.float32,
            )
            image = np.array(Image.open(frames[fid]["color"]).convert("RGB"))
            h, w = image.shape[:2]
            points = np.clip(
                np.array(
                    [kp[0, :2], kp[1, :2], kp[0, :2] + 1.5 * (kp[0, :2] - kp[9, :2])],
                    np.float32,
                ),
                [0, 0],
                [w - 1, h - 1],
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(image)
                masks, _, _ = predictor.predict(
                    point_coords=points,
                    point_labels=np.array([1, 1, 0]),
                    box=box,
                    multimask_output=False,
                )
            folder = args.output / "anchors" / side
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(masks[0].astype(np.uint8) * 255).save(
                folder / f"{fid:06d}.png"
            )
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    predictor = build_sam2_video_predictor(
        args.sam_config,
        str(args.sam_weights),
        device="cuda",
        apply_postprocessing=False,
    )
    mask_receipts = []
    for window in windows:
        side, a, b = window["side"], window["start"], window["end"]
        folder = args.output / "segments" / f"{side}-{a}-{b}"
        video = folder / "frames"
        video.mkdir(parents=True)
        for i, fid in enumerate(range(a, b + 1)):
            Image.open(frames[fid]["color"]).convert("RGB").save(
                video / f"{i:06d}.jpg", quality=95
            )
        for direction, anchor, reverse in [
            ("forward", 0, False),
            ("backward", b - a, True),
        ]:
            mask = (
                np.asarray(
                    Image.open(args.output / "anchors" / side / f"{a + anchor:06d}.png")
                )
                > 0
            )
            dest = folder / direction
            dest.mkdir()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(
                    str(video), offload_video_to_cpu=True, offload_state_to_cpu=True
                )
                predictor.add_new_mask(state, frame_idx=anchor, obj_id=1, mask=mask)
                for i, _, logits in predictor.propagate_in_video(
                    state,
                    start_frame_idx=anchor,
                    max_frame_num_to_track=b - a,
                    reverse=reverse,
                ):
                    Image.fromarray(
                        (logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
                    ).save(dest / f"{a + i:06d}.png")
            del state
            gc.collect()
            torch.cuda.empty_cache()
        for fid in range(a, b + 1):
            forward = np.asarray(Image.open(folder / "forward" / f"{fid:06d}.png")) > 0
            backward = (
                np.asarray(Image.open(folder / "backward" / f"{fid:06d}.png")) > 0
            )
            mask = forward & backward
            h, w = mask.shape
            ys, xs = np.nonzero(mask)
            iou = float(mask.sum() / max((forward | backward).sum(), 1))
            border = bool(
                len(xs)
                and (
                    xs.min() < 2
                    or ys.min() < 2
                    or xs.max() >= w - 2
                    or ys.max() >= h - 2
                )
            )
            accepted = iou >= 0.25 and mask.sum() >= 64 and not border
            if accepted:
                dest = args.output / "masks" / side
                dest.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask.astype(np.uint8) * 255).save(
                    dest / f"{fid:06d}.png"
                )
            mask_receipts.append(
                {
                    "side": side,
                    "frame_id": fid,
                    "iou": iou,
                    "border": border,
                    "accepted": bool(accepted),
                }
            )
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    # The pinned HaMeR fork imports Chumpy, whose public release predates NumPy 1.24.
    for name, value in [
        ("bool", bool),
        ("int", int),
        ("float", float),
        ("complex", complex),
        ("object", object),
        ("str", str),
        ("unicode", str),
    ]:
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    from hamer.models import load_hamer

    spec = importlib.util.spec_from_file_location("crop_hamer", args.hamer_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    model, cfg = load_hamer(str(args.hamer_weights))
    model = model.cuda().eval()
    counts = {}
    for side in ["left", "right"]:
        counts[side] = 0
        for maskpath in sorted((args.output / "masks" / side).glob("*.png")):
            fid = int(maskpath.stem)
            mask = np.asarray(Image.open(maskpath)) > 0
            ys, xs = np.nonzero(mask)
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            size = max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.7
            image = np.asarray(Image.open(frames[fid]["color"]).convert("RGB"))
            h, w = image.shape[:2]
            pred = helper.run_hamer(
                model, cfg, image, float(cx), float(cy), float(size), side == "right"
            )
            joints = pred["pred_keypoints_3d"] + pred["pred_cam_t_full"]
            points = joints[:, :2] / joints[:, 2:] * pred["scaled_focal_length"] + [
                w / 2,
                h / 2,
            ]
            kp = np.c_[points, np.zeros(21)]
            semantic = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
            for k, (x, y) in enumerate(points):
                x, y = round(x), round(y)
                if 0 <= x < w and 0 <= y < h and semantic[y, x]:
                    kp[k, 2] = 0.5
            if np.count_nonzero(kp[:, 2]) < 4:
                continue
            dest = args.output / side
            dest.mkdir(exist_ok=True)
            data = {
                "keypoints": kp.tolist(),
                "mask": str(maskpath.resolve()),
                "estimated": True,
                "source": "bidirectional_sam2_consensus_and_fresh_hamer",
                "original_direct_observation": bool(gate[side][fid]),
            }
            (dest / f"{fid:06d}.json").write_text(json.dumps(data, indent=2))
            counts[side] += 1
    (args.output / "receipt.json").write_text(
        json.dumps(
            {
                "windows": windows,
                "mask_checks": mask_receipts,
                "accepted_estimates": counts,
                "original_contacts_unchanged": True,
                "limitation": "Model predictions; mask agreement is not independent accuracy evidence. Review source overlays.",
            },
            indent=2,
        )
    )
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "manifest",
        "gate",
        "candidates",
        "tracks",
        "windows",
        "output",
        "sam-weights",
        "hamer-weights",
        "hamer-helper",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    recover(parser.parse_args())
