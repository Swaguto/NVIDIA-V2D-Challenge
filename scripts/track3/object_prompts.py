"""Canonical prompt aliases and explicit automatic instance association.

Numbered same-category instances use the earliest joint detection. When the
detector cannot distinguish their printed identities, initial left-to-right
order is an inferred convention, never a certified canonical label.
"""

import itertools
import math
import re

SEMANTIC_PROMPTS = {
    "white_pot": "white saucepan with a long handle",
    "white_pot_lid": "detached circular lid",
    "mini_dust_pan": "small dustpan",
    "mini_sweeper": "small hand brush",
    "wooden_spoon": "wooden spoon",
    "blue_cup": "blue cup",
    "beige_cup": "beige cup",
    "plastic_dish_rack": "plastic dish rack",
    "water_pitcher": "white watering can with a long spout",
    "wooden_piece_1": "wooden piece 1",
    "wooden_piece_2": "wooden piece 2",
}
SEMANTIC_ALIASES = {
    "white_pot": ["white saucepan", "saucepan"],
    "white_pot_lid": ["circular lid", "pot lid", "lid"],
    "mini_dust_pan": ["dustpan"],
    "mini_sweeper": ["hand brush", "small brush"],
    "wooden_spoon": ["spoon"],
    "blue_cup": ["blue mug", "cup", "mug"],
    "beige_cup": ["beige mug", "white cup", "cream cup", "cup", "mug"],
    "plastic_dish_rack": ["dish rack", "dish drying rack", "rack"],
    "water_pitcher": [
        "watering",
        "watering can",
        "watering jug",
        "water pitcher",
        "pitcher",
        "jug",
    ],
    "wooden_piece_1": [
        "wooden board",
        "wooden piece",
        "wooden block",
        "wooden puzzle board",
    ],
    "wooden_piece_2": [
        "wooden board",
        "wooden piece",
        "wooden block",
        "wooden puzzle board",
    ],
}


def detector_box_limit(name, fallback, canonical_limits=False):
    """Long spouts need a larger box; retain the pot/table rejection prior.

    This optional class prior changes only the detector box area. Confidence,
    propagated mask area, measured depth and reconstruction gates are separate.
    """
    if canonical_limits and name == "water_pitcher":
        return max(fallback, 0.20)
    return fallback


def label_candidates(label, names):
    label = " ".join(label.lower().replace("_", " ").split())
    matches = []
    for index, name in enumerate(names):
        for phrase in {
            name.replace("_", " "),
            SEMANTIC_PROMPTS.get(name, ""),
            *SEMANTIC_ALIASES.get(name, []),
        }:
            if phrase and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", label):
                matches.append((phrase, index))
    # A longer specific phrase can subsume "cup" or "white pot"; two separate
    # canonical names in one detector label remain ambiguous.
    matches = [
        (p, i) for p, i in matches if not any(p != q and p in q for q, _ in matches)
    ]
    return sorted({index for _, index in matches})


def assign_label(label, names):
    candidates = label_candidates(label, names)
    return candidates[0] if len(candidates) == 1 else None


def numbered_groups(names):
    groups = {}
    for index, name in enumerate(names):
        match = re.fullmatch(r"(.+)_([0-9]+)", name)
        if match:
            groups.setdefault(match[1], []).append((int(match[2]), index))
    return [
        tuple(index for _, index in sorted(group))
        for group in groups.values()
        if len(group) > 1
    ]


def intersection_fraction(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return overlap / area if area > 0 else 1.0


def numbered_instance_seeds(detections, names, group):
    """Require a complete, distinct joint view rather than duplicating one box."""
    if not 2 <= len(group) <= 4:
        raise ValueError("Numbered association requires two to four distinct slots")
    candidates = []
    for detection in detections:
        ids = label_candidates(detection["label"], names)
        b = detection["box"]
        if (
            ids
            and set(ids).issubset(group)
            and len(b) == 4
            and all(math.isfinite(x) for x in [*b, detection["score"]])
            and b[2] > b[0]
            and b[3] > b[1]
        ):
            candidates.append(dict(detection, candidates=ids))
    candidates = sorted(candidates, key=lambda d: d["score"], reverse=True)[:12]
    joint = []
    for items in itertools.combinations(candidates, len(group)):
        if any(
            intersection_fraction(a["box"], b["box"]) > 0.2
            for a, b in itertools.combinations(items, 2)
        ):
            continue
        fixed = {d["candidates"][0]: d for d in items if len(d["candidates"]) == 1}
        if len(fixed) != sum(len(d["candidates"]) == 1 for d in items):
            continue
        remaining = sorted(
            (d for d in items if len(d["candidates"]) > 1),
            key=lambda d: (d["box"][0] + d["box"][2], d["box"][1]),
        )
        free = [index for index in group if index not in fixed]
        if any(index not in d["candidates"] for index, d in zip(free, remaining)):
            continue
        mapping = dict(fixed)
        mapping.update(zip(free, remaining))
        joint.append((sum(d["score"] for d in items), mapping, set(free)))
    if not joint:
        return {}
    _, selected, inferred = max(joint, key=lambda item: item[0])
    return {
        index: dict(
            d,
            identity_policy="inferred_initial_left_to_right"
            if index in inferred
            else "automatic_specific_label",
        )
        for index, d in selected.items()
    }
