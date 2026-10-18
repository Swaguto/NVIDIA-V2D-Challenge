"""Choose a less occluded reconstruction view from image evidence only."""

import argparse, json
from pathlib import Path
import cv2, numpy as np
from reconstruct_stereo_object import sha


def mask_features(target, others, hands):
    target = np.asarray(target, bool)
    others = np.asarray(others, bool)
    hands = np.asarray(hands, bool)
    if target.shape != others.shape or target.shape != hands.shape or target.ndim != 2:
        raise ValueError("Visibility masks must share the calibrated pixel grid")
    y, x = np.nonzero(target)
    if not len(x):
        return dict(area=0, overlap_fraction=1.0, bbox_contamination=2.0)
    box = np.zeros_like(target)
    box[y.min() : y.max() + 1, x.min() : x.max() + 1] = True
    area = int(target.sum())
    box_area = int(box.sum())
    return dict(
        area=area,
        overlap_fraction=float((target & others).sum() / area),
        bbox_contamination=float(
            ((box & others).sum() + (box & hands).sum()) / box_area
        ),
    )


def rank(features):
    if not features:
        raise ValueError("No reconstruction image candidates")
    threshold = max(64, 0.4 * float(np.percentile([r["area"] for r in features], 95)))
    eligible = [r for r in features if r["area"] >= threshold]
    if not eligible:
        raise ValueError("No sufficiently complete target foreground")
    clean = [r for r in eligible if r["overlap_fraction"] <= 0.05]
    # Earlier views break visibility ties; maximum-depth-area frames can fuse
    # stacked instances. Depth sufficiency is checked independently below.
    return sorted(
        clean or eligible,
        key=lambda r: (r["bbox_contamination"], r["overlap_fraction"], r["frame"]),
    ), threshold


def select(data, masks, hand_masks, episode, name, out):
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((masks / "report.json").read_text())
    names = report["objects"]
    if report["episode"] != episode or name not in names:
        raise ValueError("Mask episode/identity differs")
    features = []
    hand_dirs = (
        sorted(p for p in hand_masks.iterdir() if p.is_dir()) if hand_masks else []
    )
    for path in sorted((masks / name).glob("*.png")):
        if not path.stem.isdigit():
            continue
        target = cv2.imread(str(path), 0)
        if target is None:
            raise ValueError("Unreadable target mask")
        other = np.zeros_like(target, bool)
        hands = other.copy()
        for obj in names:
            if obj == name:
                continue
            p = masks / obj / path.name
            if not p.is_file():
                raise ValueError("Other-object foreground coverage is incomplete")
            a = cv2.imread(str(p), 0)
            if a is None or a.shape != target.shape:
                raise ValueError("Object mask axes differ")
            other |= a > 127
        for folder in hand_dirs:
            p = folder / path.name
            if p.is_file():
                a = cv2.imread(str(p), 0)
                if a is None or a.shape != target.shape:
                    raise ValueError("Hand mask axes differ")
                hands |= a > 127
        features.append(
            dict(frame=int(path.stem), **mask_features(target > 127, other, hands))
        )
    candidates, threshold = rank(features)
    from stereo_depth import load_rig, stereo_depth

    rig = load_rig(str(data / "meta/camera_calibration.json"))
    videos = [
        data
        / "videos/chunk-000"
        / f"observation.images.{cam}"
        / f"episode_{episode:06d}.mp4"
        for cam in (rig.left, rig.right)
    ]
    caps = [cv2.VideoCapture(str(p)) for p in videos]
    checked = []
    chosen = None
    try:
        for row in candidates[:32]:
            images = []
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
                ok, image = cap.read()
                if not ok:
                    raise ValueError("Candidate is outside the source video axis")
                images.append(image)
            depth, valid = stereo_depth(rig, *images)
            target = cv2.imread(str(masks / name / f"{row['frame']:06d}.png"), 0) > 127
            measured = (
                valid & target & np.isfinite(depth) & (depth > 0.1) & (depth < 2.5)
            )
            checked.append(dict(**row, measured_foreground_pixels=int(measured.sum())))
            if measured.sum() >= 200:
                chosen = checked[-1]
                break
    finally:
        for cap in caps:
            cap.release()
    if chosen is None:
        raise ValueError("Low-occlusion candidates have insufficient measured depth")
    result = dict(
        schema="v2d.reconstruction-frame-selection.v1",
        episode=episode,
        object=name,
        selected=chosen,
        checked_candidates=checked,
        area_floor_pixels=threshold,
        source_mask_report_sha256=sha(masks / "report.json"),
        chosen_mask_sha256=sha(masks / name / f"{chosen['frame']:06d}.png"),
        provided_meshes_read=False,
        reference_pose_labels_read=False,
        selection_uses_simulator_results=False,
        physical_task_accepted=False,
        selector_sha256=sha(Path(__file__)),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["data", "masks", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--hand-masks", type=Path)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--object", required=True)
    a = p.parse_args()
    print(
        json.dumps(
            select(a.data, a.masks, a.hand_masks, a.episode, a.object, a.out), indent=2
        )
    )
