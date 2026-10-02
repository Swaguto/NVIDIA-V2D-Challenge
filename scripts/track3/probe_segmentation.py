#!/usr/bin/env python3
"""Single-frame test: rectified stereo frame -> object masks -> metric point cloud.

Establishes the three things the pose stage needs before any registration:
  - Grounding DINO can find the episode's known objects from name-only prompts
  - SAM2 turns those boxes into clean masks
  - mask pixels carry usable metric depth (so registration has scale)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu")
sys.path.insert(0, "/home/ubuntu/repoeval")
from reference_loader import load_reference   # noqa: E402
from stereo_depth import backproject_full, load_rig, stereo_depth   # noqa: E402

PUB = Path("/home/ubuntu/v2d_track3/data/hf/track_3/public")
VID = PUB / "videos/chunk-000"
DEV = "cuda"
GDINO = "IDEA-Research/grounding-dino-base"
SAM2 = "facebook/sam2.1-hiera-large"


def read_frame(path, idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    assert ok
    return img


def main():
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fidx = int(sys.argv[2]) if len(sys.argv) > 2 else 120

    ref = load_reference(ep, PUB)
    names = list(ref.object_names)
    print(f"episode {ep} frame {fidx}: objects = {names}")

    rig = load_rig()
    il = read_frame(VID / f"observation.images.{rig.left}" / f"episode_{ep:06d}.mp4", fidx)
    ir = read_frame(VID / f"observation.images.{rig.right}" / f"episode_{ep:06d}.mp4", fidx)
    pl, _ = rig.rectify_pair(il, ir)
    depth, valid = stereo_depth(rig, il, ir)
    rgb = cv2.cvtColor(pl, cv2.COLOR_BGR2RGB)
    print(f"rectified {pl.shape}, depth valid {valid.mean():.1%}, "
          f"median {np.median(depth[valid]):.3f} m")

    # ---- Grounding DINO: one phrase per known object -------------------------
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    prompt = ". ".join(n.replace("_", " ") for n in names) + "."
    proc = AutoProcessor.from_pretrained(GDINO)
    model = GroundingDinoForObjectDetection.from_pretrained(GDINO).to(DEV).eval()
    with torch.inference_mode():
        inp = proc(images=rgb, text=prompt, return_tensors="pt").to(DEV)
        out = model(**inp)
    det = proc.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=0.30,
        text_threshold=0.25, target_sizes=[pl.shape[:2]])[0]
    print(f"\nGroundingDINO prompt: {prompt!r}")
    print(f"  {len(det['boxes'])} boxes (threshold 0.30)")
    phrase_to_name = {}
    for n in names:
        phrase_to_name[n.replace("_", " ")] = n
    for b, lab, sc in zip(det["boxes"].tolist(), det["text_labels"], det["scores"].tolist()):
        print(f"    {lab!r} score={sc:.3f} box={[round(v) for v in b]}")

    if len(det["boxes"]) == 0:
        print("no detections; cannot proceed")
        return

    # ---- SAM2: box prompts -> masks -----------------------------------------
    from transformers import Sam2Model, Sam2Processor
    sproc = Sam2Processor.from_pretrained(SAM2)
    smodel = Sam2Model.from_pretrained(SAM2).to(DEV).eval()
    boxes = [[[float(v) for v in b] for b in det["boxes"].tolist()]]
    with torch.inference_mode():
        sinp = sproc(images=rgb, input_boxes=boxes, return_tensors="pt").to(DEV)
        print(f"\nSAM2 input keys: {sorted(k for k in sinp if k != 'pixel_values')}")
        sout = smodel(**{k: v for k, v in sinp.items()}, multimask_output=False)
    raw_masks = sout.pred_masks if hasattr(sout, "pred_masks") else sout.output_masks
    # NB: post_process_masks' 3rd positional is mask_threshold, so pass sizes by keyword.
    kw = {}
    if sinp.get("reshaped_input_sizes") is not None:
        kw["reshaped_input_sizes"] = sinp["reshaped_input_sizes"]
    masks = sproc.post_process_masks(raw_masks, sinp["original_sizes"], binarize=True, **kw)[0]
    masks = np.asarray(masks.cpu() if torch.is_tensor(masks) else masks)[:, 0]
    print(f"  masks {masks.shape} dtype={masks.dtype} unique={np.unique(masks)[:5]}")

    # ---- mask -> metric point cloud ----------------------------------------
    pts_all = backproject_full(depth, valid, rig.P1)
    keep_all = valid.ravel()
    print(f"\n{'object':<22} {'mask px':>8} {'depth px':>9} {'median z':>9}")
    clouds = {}
    for i, n in enumerate(names):
        m = masks[i].astype(bool)
        sel = m.ravel() & keep_all
        if sel.sum() < 50:
            print(f"{n:<22} {m.sum():>8} {int(sel.sum()):>9} {'--':>9}")
            clouds[n] = None
            continue
        z = pts_all[sel, 2]
        print(f"{n:<22} {int(m.sum()):>8} {int(sel.sum()):>9} {np.median(z):>9.3f}")
        clouds[n] = pts_all[sel]

    # extent of each cloud, to compare against expected real-world size
    print("\ncloud extents (m) — compare against mesh dimensions:")
    for n, p in clouds.items():
        if p is None:
            continue
        ext = p.max(0) - p.min(0)
        print(f"  {n:<22} {ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f}")

    out = Path("/home/ubuntu/seg_probe"); out.mkdir(exist_ok=True)
    vis = pl.copy()
    colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
    for i, n in enumerate(names):
        m = masks[i].astype(bool)
        if m.sum() == 0:
            continue
        vis[m] = (0.55 * np.array(colours[i % len(colours)]) + 0.45 * vis[m]).astype(np.uint8)
        c = np.array(colours[i % len(colours)])
        ys, xs = np.nonzero(m)
        cv2.putText(vis, n, (int(xs.mean()), int(ys.min()) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(int(x) for x in c), 2)
    cv2.imwrite(str(out / f"ep{ep:03d}_f{fidx:04d}_masks.png"), vis)
    json.dump({n: (None if p is None else p[:2000].tolist()) for n, p in clouds.items()},
              open(out / f"ep{ep:03d}_f{fidx:04d}_clouds.json", "w"))
    print(f"\nwrote {out}/ep{ep:03d}_f{fidx:04d}_masks.png")


if __name__ == "__main__":
    main()