"""Conservative wearer association for calibrated egocentric stereo footage.

Bootstrap on a nearby, sufficiently large hand entering from the lower image;
thereafter require motion, size and measured-depth continuity. This is a
recorded image prior, not a learned identity classifier. Ambiguous observations
remain missing. No episode ID, manual exclusion region or reference pose enters
selection. Review is still required before declaring an episode accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SelectionSettings:
    minimum_depth_m: float = 0.15
    maximum_depth_m: float = 0.95
    minimum_depth_pixels: int = 32
    minimum_depth_fraction: float = 0.10
    bootstrap_area_fraction: float = 0.0035
    bootstrap_center_y: float = 0.60
    maximum_gap_frames: int = 12
    ambiguity_margin: float = 0.18


def candidate_depth(box, depth, settings=SelectionSettings()):
    """Use the central box's median; insufficient stereo evidence is unknown."""
    x0, y0, x1, y1 = np.asarray(box, dtype=float)
    w, h = x1 - x0, y1 - y0
    xa, ya, xb, yb = np.rint(
        [x0 + 0.2 * w, y0 + 0.2 * h, x1 - 0.2 * w, y1 - 0.2 * h]
    ).astype(int)
    patch = depth[
        max(0, ya) : min(depth.shape[0], yb), max(0, xa) : min(depth.shape[1], xb)
    ]
    values = patch[np.isfinite(patch) & (patch > 0)]
    if (
        values.size < settings.minimum_depth_pixels
        or values.size < settings.minimum_depth_fraction * patch.size
    ):
        return None, int(values.size), float(values.size / max(patch.size, 1))
    return float(np.median(values)), int(values.size), float(values.size / patch.size)


class WearerSelector:
    def __init__(self, width, height, settings=SelectionSettings()):
        self.width, self.height, self.settings = width, height, settings
        self.previous = {}

    def select(self, frame, candidates):
        selected = {}
        for side, label in [("left", 0), ("right", 1)]:
            previous = self.previous.get(side)
            gap = frame - previous["frame"] if previous is not None else None
            continuing = (
                previous is not None and gap <= self.settings.maximum_gap_frames
            )
            ranked = []
            for index, candidate in enumerate(candidates):
                if candidate["class"] != label:
                    continue
                box = np.asarray(candidate["box_xyxy"], dtype=float)
                center = (box[:2] + box[2:]) / 2 / np.array([self.width, self.height])
                area = float(
                    np.maximum(box[2:] - box[:2], 0).prod() / (self.width * self.height)
                )
                distance = candidate.get("depth_m")
                candidate["owner_rejection"] = None
                if (
                    distance is None
                    or not self.settings.minimum_depth_m
                    <= distance
                    <= self.settings.maximum_depth_m
                ):
                    candidate["owner_rejection"] = "no reliable nearby stereo depth"
                    continue
                if continuing:
                    motion = float(np.linalg.norm(center - previous["center"]))
                    size_change = abs(math.log(max(area, 1e-10) / previous["area"]))
                    depth_change = abs(math.log(distance / previous["depth"]))
                    allowed_motion = 0.06 + 0.012 * gap
                    if (
                        motion > allowed_motion
                        or size_change > 0.85
                        or depth_change > 0.40
                    ):
                        candidate["owner_rejection"] = "failed wearer continuity gate"
                        continue
                    cost = (
                        motion / allowed_motion
                        + 0.4 * size_change
                        + 0.5 * depth_change
                        + 0.2 * (1 - candidate["confidence"])
                    )
                    basis = "stereo_depth_and_temporal_continuity"
                else:
                    if (
                        area < self.settings.bootstrap_area_fraction
                        or center[1] < self.settings.bootstrap_center_y
                    ):
                        candidate["owner_rejection"] = (
                            "no lower-image wearer bootstrap evidence"
                        )
                        continue
                    cost = (
                        distance
                        + 0.5 * (1 - center[1])
                        + 0.2 * (1 - candidate["confidence"])
                    )
                    basis = "nearby_lower_image_hand_bootstrap"
                candidate["owner_cost"] = float(cost)
                ranked.append((cost, index, center, area, distance, basis))
            ranked.sort(key=lambda item: item[0])
            if not ranked:
                selected[side] = {
                    "candidate_index": None,
                    "reason": "no supported wearer observation",
                }
                continue
            if (
                len(ranked) > 1
                and ranked[1][0] - ranked[0][0] < self.settings.ambiguity_margin
            ):
                selected[side] = {
                    "candidate_index": None,
                    "reason": "ambiguous wearer candidates",
                }
                continue
            cost, index, center, area, distance, basis = ranked[0]
            self.previous[side] = {
                "frame": frame,
                "center": center,
                "area": area,
                "depth": distance,
            }
            selected[side] = {
                "candidate_index": index,
                "reason": basis,
                "cost": float(cost),
            }
        return selected
