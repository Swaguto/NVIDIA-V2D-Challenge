"""Identity regressions independent of learned hand detector confidences."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "validation/issue6"))
from wearer_selection import WearerSelector, candidate_depth


def hand(box, depth=0.45, confidence=0.8, label=0):
    return {"box_xyxy": box, "depth_m": depth, "confidence": confidence, "class": label}


def test_bystander_cannot_replace_occluded_wearer_by_detector_confidence():
    selector = WearerSelector(1280, 800)
    wearer = hand([250, 580, 400, 770])
    background = hand([80, 100, 150, 190], depth=1.4, confidence=0.99)
    assert selector.select(0, [background, wearer])["left"]["candidate_index"] == 1
    for frame in range(1, 30):
        assert selector.select(frame, [background])["left"]["candidate_index"] is None
    returning = hand([260, 590, 410, 780])
    assert selector.select(30, [background, returning])["left"]["candidate_index"] == 1


def test_nearby_competing_same_side_candidates_are_unknown():
    selector = WearerSelector(1280, 800)
    candidates = [hand([250, 580, 400, 770]), hand([600, 580, 750, 770])]
    assert selector.select(0, candidates)["left"]["reason"] == "ambiguous wearer candidates"


def test_depth_or_motion_switch_does_not_acquire_new_person():
    selector = WearerSelector(1280, 800)
    assert selector.select(0, [hand([250, 580, 400, 770])])["left"]["candidate_index"] == 0
    for candidate in [hand([900, 300, 1100, 500]), hand([250, 580, 400, 770], depth=0.9)]:
        assert selector.select(1, [candidate])["left"]["candidate_index"] is None


def test_sparse_or_absent_metric_depth_cannot_establish_ownership():
    depth = np.full((800, 1280), np.nan)
    assert candidate_depth([250, 580, 400, 770], depth)[0] is None
    depth[650:680, 300:330] = 0.45
    assert candidate_depth([250, 580, 400, 770], depth)[0] is None
    depth[580:770, 250:400] = 0.45
    assert candidate_depth([250, 580, 400, 770], depth)[0] == 0.45
