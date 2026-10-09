"""Find a transient detector label stealing the other wearer's hand track.

This only rejects uncertain imported observations. It never relabels a MANO fit,
adds a detection, changes raw predictions, or uses an episode/reference pose.
"""
from __future__ import annotations

import math


def identity_conflicts(audit, width, height):
    settings = audit["settings"]
    maximum_gap = settings["maximum_gap_frames"]
    history = {}
    quarantined = {}
    conflicts = []
    for row in audit["frames"]:
        frame = row["frame_index"]
        selected = row["selected"]
        for side in ("left", "right"):
            index = selected[side]["candidate_index"]
            if index is None:
                continue
            candidate = row["candidates"][index]
            box = candidate["box_xyxy"]
            center = ((box[0] + box[2]) / (2 * width),
                      (box[1] + box[3]) / (2 * height))
            area = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
            depth = candidate["depth_m"]
            other = "right" if side == "left" else "left"
            previous = history.get(other, [])
            missing_other = selected[other]["candidate_index"] is None
            bootstrap = selected[side]["reason"] == "nearby_lower_image_hand_bootstrap"
            conflicting = False
            if missing_other and (bootstrap or side in quarantined) and len(previous) == 2:
                before, last = previous
                gap = frame - last[0]
                if 0 < gap <= maximum_gap and depth is not None and depth > 0:
                    velocity = tuple((last[1][i] - before[1][i]) /
                                     (last[0] - before[0]) for i in range(2))
                    prediction = tuple(last[1][i] + velocity[i] * gap for i in range(2))
                    motion = math.dist(center, prediction)
                    size_change = abs(math.log(max(area, 1e-10) / last[2]))
                    depth_change = abs(math.log(depth / last[3]))
                    allowed_motion = 0.06 + 0.012 * gap
                    conflicting = (motion <= allowed_motion and size_change <= 0.85
                                   and depth_change <= 0.40)
                    if conflicting:
                        conflicts.append({"frame_id": frame, "hand": side,
                                          "candidate_index": index,
                                          "reason": "opposite_hand_identity_conflict",
                                          "opposite_previous_frame": last[0],
                                          "prediction_error_normalized": motion,
                                          "allowed_motion_normalized": allowed_motion})
            owner = other if conflicting else side
            if conflicting:
                quarantined[side] = frame
            else:
                quarantined.pop(side, None)
            history.setdefault(owner, []).append((frame, center, area, depth))
            history[owner] = history[owner][-2:]
    return conflicts
