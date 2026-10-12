# ruff: noqa: E402
# Validation helpers are loaded after the repository path is added.
"""Controls prevent visually unsupported fits being accepted to hide flicker."""

import sys
from pathlib import Path

import numpy as np

VALIDATION = Path(__file__).resolve().parents[3] / "validation/issue6"
sys.path.insert(0, str(VALIDATION))
import json

import pytest
from automatic_hand_quality import classify
from automatic_hands import flow_crop, method_hash, reject_cross_side_conflict, sha, suppress, write
from freeze_automatic_hands import freeze

POLICY = json.loads((VALIDATION / "automatic_hands.json").read_text())


def test_visible_but_wrong_depth_is_rejected():
    mask = np.ones((20, 20), bool)
    render = np.full((20, 20), 0.5)
    depth = np.full((20, 20), 0.53)
    assert "depth_mismatch" in classify(mask, render, depth, 0.95, POLICY)["reasons"]


def test_good_depth_outside_hand_mask_is_rejected():
    mask = np.zeros((20, 20), bool)
    mask[:5] = True
    render = np.full((20, 20), 0.5)
    depth = render.copy()
    assert "mesh_outside_image_hand_mask" in classify(mask, render, depth, 0.95, POLICY)["reasons"]


def test_no_stereo_is_unknown_even_with_perfect_mask():
    mask = np.ones((20, 20), bool)
    render = np.full((20, 20), 0.5)
    depth = np.full((20, 20), np.nan)
    assert "insufficient_stereo_support" in classify(mask, render, depth, 0.95, POLICY)["reasons"]


def test_good_current_evidence_passes_consistency_only():
    mask = np.ones((20, 20), bool)
    render = np.full((20, 20), 0.5)
    assert not classify(mask, render, render, 0.95, POLICY)["reasons"]


def test_measured_foreground_occlusion_does_not_count_as_bad_silhouette():
    mask = np.zeros((20, 20), bool)
    mask[:10] = True
    render = np.full((20, 20), 0.5)
    depth = render.copy()
    depth[10:] = 0.3
    result = classify(mask, render, depth, 0.95, POLICY)
    assert not result["reasons"]
    assert result["depth_occluded_mesh_pixels"] == 200


def test_fully_occluded_mesh_cannot_pass_as_observed():
    mask = np.ones((20, 20), bool)
    render = np.full((20, 20), 0.5)
    depth = np.full((20, 20), 0.3)
    assert "insufficient_stereo_support" in classify(mask, render, depth, 0.95, POLICY)["reasons"]


def test_duplicate_detections_do_not_create_ambiguity():
    base = {"box_xyxy": [0, 0, 10, 10], "confidence": 0.9, "class": 0}
    duplicate = dict(base, confidence=0.8)
    other_side = dict(base, **{"class": 1})
    assert suppress([duplicate, base, other_side], 0.5) == [base, other_side]


def test_blank_frame_cannot_propose_recovery():
    image = np.zeros((64, 64), np.uint8)
    assert flow_crop(image, image, [4, 4, 50, 50], POLICY) is None


def test_one_image_region_cannot_be_both_hands():
    candidates = [{"box_xyxy": [0, 0, 100, 100]}, {"box_xyxy": [2, 2, 98, 98]}]
    selected = {"left": {"candidate_index": 0}, "right": {"candidate_index": 1}}
    assert reject_cross_side_conflict(selected, candidates, 0.5)
    assert all(s["candidate_index"] is None for s in selected.values())


def test_new_track_cannot_steal_stable_other_hand():
    candidates = [{"box_xyxy": [0, 0, 100, 100]}, {"box_xyxy": [2, 2, 98, 98]}]
    selected = {
        "left": {"candidate_index": 0, "reason": "stereo_depth_and_temporal_continuity"},
        "right": {"candidate_index": 1, "reason": "nearby_lower_image_hand_bootstrap"},
    }
    assert reject_cross_side_conflict(selected, candidates, 0.5) == ["right"]
    assert selected["left"]["candidate_index"] == 0


def freeze_fixture(tmp_path, episode):
    config = tmp_path / "automatic_hands.json"
    write(config, POLICY)
    hashes = method_hash(config)
    receipt = {"episode": episode, "method_files": hashes, "status": "completed_proposals_require_accuracy_checks"}
    for key in (
        "detector_sha256",
        "hamer_checkpoint_sha256",
        "sam_checkpoint_sha256",
        "mano_right_sha256",
        "hamer_config_sha256",
        "mano_mean_params_sha256",
        "hamer_source_commit",
    ):
        receipt[key] = "fixture"
    rp = tmp_path / "receipt.json"
    write(rp, receipt)
    qp = tmp_path / "quality.json"
    write(
        qp,
        {
            "episode": episode,
            "config_sha256": sha(config),
            "method_files": hashes,
            "method_unchanged_during_run": True,
            "source_proposal_receipt_sha256": sha(rp),
        },
    )
    return config, rp, qp


def test_development_evidence_can_be_frozen(tmp_path):
    config, rp, qp = freeze_fixture(tmp_path, 12)
    freeze(config, [rp], [qp], tmp_path / "freeze.json")
    assert json.loads((tmp_path / "freeze.json").read_text())["quality_accepted"] is False


def test_holdout_cannot_be_used_to_freeze_method(tmp_path):
    config, rp, qp = freeze_fixture(tmp_path, 0)
    with pytest.raises(ValueError, match="held-out"):
        freeze(config, [rp], [qp], tmp_path / "freeze.json")


def test_method_change_invalidates_development_evidence(tmp_path):
    config, rp, qp = freeze_fixture(tmp_path, 12)
    policy = POLICY.copy()
    policy["maximum_depth_residual_m"] = 0.03
    write(config, policy)
    with pytest.raises(ValueError, match="current method"):
        freeze(config, [rp], [qp], tmp_path / "freeze.json")
