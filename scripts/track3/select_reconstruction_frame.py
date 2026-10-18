"""Choose a less occluded reconstruction view from image evidence only."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from reconstruct_stereo_object import sha


def mask_features(target, others, hands):
    target = np.asarray(target, bool)
    others = np.asarray(others, bool)
    hands = np.asarray(hands, bool)
    if target.shape != others.shape or target.shape != hands.shape or target.ndim != 2:
        raise ValueError("Visibility masks must share the calibrated pixel grid")
    y, x = np.nonzero(target)
    if not len(x):
        return {"area": 0, "overlap_fraction": 1.0, "bbox_contamination": 2.0}
    box = np.zeros_like(target)
    box[y.min() : y.max() + 1, x.min() : x.max() + 1] = True
    area = int(target.sum())
    box_area = int(box.sum())
    return {
        "area": area,
        "overlap_fraction": float((target & others).sum() / area),
        "hand_overlap_fraction": float((target & hands).sum() / area),
        "bbox_contamination": float(
            ((box & others).sum() + (box & hands).sum()) / box_area
        ),
    }


def metric_area(target, depth, valid, fx, fy):
    """Approximate projected metric area without favoring a closer camera view."""
    target, depth, valid = (
        np.asarray(target, bool),
        np.asarray(depth),
        np.asarray(valid, bool),
    )
    if target.ndim != 2 or target.shape != depth.shape or target.shape != valid.shape:
        raise ValueError(
            "Metric view area needs matching mask and measured depth grids"
        )
    if not np.isfinite([fx, fy]).all() or min(fx, fy) <= 0:
        raise ValueError("Metric view area needs positive calibrated focal lengths")
    good = target & valid & np.isfinite(depth) & (depth > 0.1) & (depth < 2.5)
    if good.sum() < 200:
        return None
    median = float(np.median(depth[good]))
    return {
        "metric_area_proxy_m2": float(target.sum() * median**2 / (fx * fy)),
        "measured_foreground_pixels": int(good.sum()),
        "median_depth_m": median,
    }


def rank(features, area_key="area"):
    if not features:
        raise ValueError("No reconstruction image candidates")
    if area_key not in {"area", "metric_area_proxy_m2"} or any(
        area_key not in row or not np.isfinite(row[area_key]) or row[area_key] < 0
        for row in features
    ):
        raise ValueError("Need one finite, consistent view-area measure")
    threshold = max(
        64 if area_key == "area" else 0,
        0.4 * float(np.percentile([r[area_key] for r in features], 95)),
    )
    eligible = [r for r in features if r["area"] >= 64 and r[area_key] >= threshold]
    if not eligible:
        raise ValueError("No sufficiently complete target foreground")
    clean = [
        r
        for r in eligible
        if r["overlap_fraction"] <= 0.05 and r.get("hand_overlap_fraction", 0) <= 0.05
    ]
    # Earlier views break visibility ties; maximum-depth-area frames can fuse
    # stacked instances. Depth sufficiency is checked independently below.
    return sorted(
        clean or eligible,
        key=lambda r: (
            r.get("hand_overlap_fraction", 0),
            r["bbox_contamination"],
            r["overlap_fraction"],
            r["frame"],
        ),
    ), threshold


def select(data, masks, hand_masks, episode, name, out, observations=None):
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((masks / "report.json").read_text())
    names = report["objects"]
    if report["episode"] != episode or name not in names:
        raise ValueError("Mask episode/identity differs")
    features = []
    excluded_unknown_other_foreground = []
    excluded_insufficient_measured_depth = []
    measured = None
    if observations is not None:
        measured = json.loads(observations.read_text())
        if (
            measured["episode"] != episode
            or measured["units"] != "metres"
            or measured["depth"]["encoding"] != "metres"
            or measured["input_contract"].get("reference_pose_tables_read") is not False
            or measured["color_camera"]["id"] != "ego_cam_c"
            or [r["id"] for r in measured["frames"]] != list(range(report["frames"]))
            or measured["input_contract"]["calibration_sha256"]
            != sha(data / "meta/camera_calibration.json")
        ):
            raise ValueError(
                "View selection needs matching label-free calibrated source observations"
            )
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
        other_coverage_known = True
        for obj in names:
            if obj == name:
                continue
            p = masks / obj / path.name
            if not p.is_file():
                other_coverage_known = False
                break
            a = cv2.imread(str(p), 0)
            if a is None or a.shape != target.shape:
                raise ValueError("Object mask axes differ")
            other |= a > 127
        if not other_coverage_known:
            # A missing detector result cannot establish a clean object view.
            # Exclude that candidate without discarding the complete episode.
            excluded_unknown_other_foreground.append(int(path.stem))
            continue
        for folder in hand_dirs:
            p = folder / path.name
            if p.is_file():
                a = cv2.imread(str(p), 0)
                if a is None or a.shape != target.shape:
                    raise ValueError("Hand mask axes differ")
                hands |= a > 127
        feature = dict(
            frame=int(path.stem), **mask_features(target > 127, other, hands)
        )
        if measured is not None:
            row = measured["frames"][feature["frame"]]
            dp, vp = (
                observations.parent / row[k] for k in ("depth_path", "depth_validity")
            )
            depth = np.load(dp, allow_pickle=False)
            valid = cv2.imread(str(vp), 0)
            if valid is None:
                raise ValueError("Missing measured source-depth validity")
            camera = measured["color_camera"]
            area = metric_area(
                target > 127, depth, valid > 127, camera["fx"], camera["fy"]
            )
            if area is None:
                excluded_insufficient_measured_depth.append(feature["frame"])
                continue
            feature.update(
                area, area_depth_sha256=sha(dp), area_depth_validity_sha256=sha(vp)
            )
        features.append(feature)
    area_key = "metric_area_proxy_m2" if measured is not None else "area"
    candidates, threshold = rank(features, area_key)
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
            checked.append(dict(row, measured_foreground_pixels=int(measured.sum())))
            if measured.sum() >= 200:
                chosen = checked[-1]
                break
    finally:
        for cap in caps:
            cap.release()
    if chosen is None:
        raise ValueError("Low-occlusion candidates have insufficient measured depth")
    result = {
        "schema": "v2d.reconstruction-frame-selection.v1",
        "episode": episode,
        "object": name,
        "selected": chosen,
        "checked_candidates": checked,
        "area_measure": area_key,
        "area_floor_pixels": threshold if area_key == "area" else None,
        "area_floor_m2": threshold if area_key == "metric_area_proxy_m2" else None,
        "source_observations_manifest_sha256": sha(observations)
        if observations
        else None,
        "excluded_insufficient_measured_depth_frames": excluded_insufficient_measured_depth,
        "ranking_candidates": features,
        "source_mask_report_sha256": sha(masks / "report.json"),
        "chosen_mask_sha256": sha(masks / name / f"{chosen['frame']:06d}.png"),
        "selected_foreground_sha256": {
            obj: sha(masks / obj / f"{chosen['frame']:06d}.png") for obj in names
        },
        "selected_hand_mask_sha256": {
            folder.name: sha(folder / f"{chosen['frame']:06d}.png")
            for folder in hand_dirs
            if (folder / f"{chosen['frame']:06d}.png").is_file()
        },
        "excluded_unknown_other_foreground_frames": excluded_unknown_other_foreground,
        "provided_meshes_read": False,
        "reference_pose_labels_read": False,
        "selection_uses_simulator_results": False,
        "physical_task_accepted": False,
        "selector_sha256": sha(Path(__file__)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["data", "masks", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--hand-masks", type=Path)
    p.add_argument(
        "--observations",
        type=Path,
        help="Use calibrated measured depth to remove camera-distance bias from area ranking",
    )
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--object", required=True)
    a = p.parse_args()
    print(
        json.dumps(
            select(
                a.data,
                a.masks,
                a.hand_masks,
                a.episode,
                a.object,
                a.out,
                a.observations,
            ),
            indent=2,
        )
    )
