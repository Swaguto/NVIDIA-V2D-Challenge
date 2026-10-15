"""Validate explicit image estimates without changing canonical observation gates."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_image_estimate(root, side, frame_id, camera, directly_observed):
    path = Path(root) / side / f"{frame_id:06d}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("estimated") is not True or data.get("metric_valid") or data.get("directly_observed"):
        raise ValueError("Additional image estimates cannot declare metric observations")
    if bool(data.get("original_direct_observation", directly_observed)) != directly_observed:
        raise ValueError("Additional estimate disagrees with the canonical observation gate")
    points = np.array(data["keypoints"], float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Image landmarks must be finite 21x3 predictions")
    if np.any((points[:, 2] < 0) | (points[:, 2] > 1)):
        raise ValueError("Invalid image landmark confidence")
    mask_path = Path(data["mask"])
    if not mask_path.is_absolute():
        mask_path = path.parent / mask_path
    mask = np.asarray(Image.open(mask_path)) > 0
    h, w = camera["height"], camera["width"]
    if mask.shape != (h, w) or mask.sum() < 64:
        raise ValueError("Image estimate needs an associated hand mask on the source grid")
    for x, y, score in points:
        if score <= 0:
            continue
        x, y = round(x), round(y)
        if not (0 <= x < w and 0 <= y < h) or not mask[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4].any():
            raise ValueError("Image landmark is outside its associated hand mask")
    if np.count_nonzero(points[:, 2]) < 4:
        raise ValueError("Image estimate needs at least four visible landmarks")
    return {"keypoints": points.tolist(), "mask": mask_path, "source": str(data["source"]), "estimated": True}
