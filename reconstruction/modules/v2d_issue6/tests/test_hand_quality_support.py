"""Refuse unsupported visible hands and prevent native-export geometry drift."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

VALIDATION = Path(__file__).resolve().parents[3] / "validation/issue6"
sys.path.insert(0, str(VALIDATION))
from automatic_hand_quality import classify  # noqa: E402
from hand_quality_support import DetectorBoundedPredictor, clip_mask, export_native_record  # noqa: E402

POLICY = json.loads((VALIDATION / "automatic_hands.json").read_text())


def test_forearm_mask_is_bounded_but_displaced_mesh_still_fails():
    mask = np.zeros((80, 80), bool)
    mask[10:70, 10:35] = True
    bounded = clip_mask(mask, [10, 10, 35, 35])
    scene = np.full(mask.shape, 0.5)
    correct = np.zeros(mask.shape)
    correct[10:35, 10:35] = 0.5
    assert not classify(bounded, correct, scene, 0.95, POLICY)["reasons"]
    wrong = np.zeros(mask.shape)
    wrong[10:35, 10:70] = 0.5
    assert "mesh_outside_image_hand_mask" in classify(bounded, wrong, scene, 0.95, POLICY)["reasons"]


def test_high_confidence_background_and_wrong_depth_remain_rejected():
    mask = np.ones((40, 40), bool)
    bounded = clip_mask(mask, [10, 10, 30, 30])
    wrong_region = np.zeros(mask.shape)
    wrong_region[:10, :10] = 0.5
    assert classify(bounded, wrong_region, np.full(mask.shape, 0.5), 0.99, POLICY)["reasons"]
    correct_region = bounded.astype(float) * 0.5
    assert "depth_mismatch" in classify(bounded, correct_region, np.full(mask.shape, 0.53), 0.99, POLICY)["reasons"]


@pytest.mark.parametrize("box", [[20, 10, 5, 15], [0, 0, 0, 10], [np.nan, 0, 3, 4], [50, 50, 60, 60]])
def test_unsupported_detector_box_is_refused(box):
    with pytest.raises(ValueError):
        clip_mask(np.ones((40, 40), bool), box)


def test_predictor_preserves_confidence_and_logits_and_records_mask_bounds():
    logits = np.array([0.1, 0.2])
    scores = np.array([0.91])

    class Fake:
        def set_image(self, image):
            self.image = image

        def predict(self, **kwargs):
            return np.ones((1, 40, 40), bool), scores, logits

    predictor = DetectorBoundedPredictor(Fake())
    predictor.set_image(np.zeros((40, 40, 3)))
    masks, actual_scores, actual_logits = predictor.predict(box=[10, 10, 30, 30])
    assert masks.sum() == 400
    assert actual_scores is scores and actual_logits is logits
    assert predictor.calls[0]["unconstrained_pixels"] == [1600]
    assert predictor.calls[0]["constrained_pixels"] == [400]


def native_fixture(scale=0.73):
    return (
        {
            "directly_observed": True,
            "hand_scale": scale,
            "camera": {"pred_cam_t_full": [0.1, 0.2, 0.6]},
            "mano": {"hand_pose": [0.0] * 45, "betas": [0.0] * 10},
            "direct_observation": {"interpolated": False},
        },
        {"stereo_pixels": 300, "median_depth_residual_m": 0.008},
        {"accepted_by_consistency_proxies": True, "quality_accepted": False},
    )


def test_nonunit_export_preserves_hand_dimensions_and_measured_translation():
    source, measurement, evidence = native_fixture()
    original = copy.deepcopy(source)
    offset = np.array([0.003, -0.002, 0.01])
    result = export_native_record(source, offset, {}, measurement, evidence, POLICY)
    # An independent local hand shape catches the former scale=1.0 export error.
    local = np.array([[0.0, 0.0, 0.0], [0.04, -0.01, 0.09], [-0.03, 0.02, 0.08]])
    measured = local * 0.73 + np.array([0.1, 0.2, 0.6]) + offset
    exported = local * result["hand_scale"] + result["cam_t"]
    np.testing.assert_allclose(exported, measured, atol=1e-15, rtol=0)
    assert source == original
    assert result["mano"] == source["mano"]
    assert result["directly_observed"] is True
    assert result["automatic_consistency"]["quality_accepted"] is False


def test_hamer_unit_scale_default_stays_supported():
    source, measurement, evidence = native_fixture()
    del source["hand_scale"]
    assert export_native_record(source, np.zeros(3), {}, measurement, evidence, POLICY)["hand_scale"] == 1.0


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_bad_hand_scale_is_refused(scale):
    source, measurement, evidence = native_fixture(scale)
    with pytest.raises(ValueError):
        export_native_record(source, np.zeros(3), {}, measurement, evidence, POLICY)


def test_display_hold_and_accuracy_claim_cannot_be_exported_as_observations():
    source, measurement, evidence = native_fixture()
    source["directly_observed"] = False
    with pytest.raises(ValueError, match="Held"):
        export_native_record(source, np.zeros(3), {}, measurement, evidence, POLICY)
    source["directly_observed"] = True
    evidence["quality_accepted"] = True
    with pytest.raises(ValueError, match="accuracy acceptance"):
        export_native_record(source, np.zeros(3), {}, measurement, evidence, POLICY)
