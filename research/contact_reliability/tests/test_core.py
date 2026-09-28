import copy
import numpy as np
import pytest
from v2d_reliability.core import SIGNALS, aligned_weights, fit_normalizer, score
from v2d_reliability.signals import mask_iou, reprojection_error, temporal_acceleration


def observation(sequence="dev", t=8):
    values = np.zeros((t, 2, 2, 4))
    values[..., 0:2] = 1
    values[..., 2] = .01
    values[..., 3] = .1
    return {"schema_version": "observations_v1", "sequence_id": sequence,
            "timestamps": (np.arange(t) / 30).tolist(), "hand_ids": ["left", "right"],
            "object_ids": ["base", "lid"], "signal_names": list(SIGNALS),
            "signals": values.tolist(), "valid": np.ones_like(values, dtype=bool).tolist()}


def test_reliability_declines_under_corruption_and_missing_falls_back(caplog):
    doc = observation()
    norm = fit_normalizer([doc], ["dev"])
    clean = np.array(score(doc, norm)["weights"])
    doc["signals"][3][0][1] = [0, 0, 100, 100]
    doc["valid"][4][1][0] = [False] * 4
    weights = np.array(score(doc, norm)["weights"])
    assert weights[3, 0, 1] < clean[3, 0, 1]
    assert weights[4, 1, 0] == 1
    assert weights.min() >= .25
    assert "all-missing" in caplog.text


def test_fit_rejects_holdout_and_nonfinite():
    with pytest.raises(ValueError, match="development"):
        fit_normalizer([observation("holdout")], ["dev"])
    doc = observation(); doc["signals"][0][0][0][0] = float("nan")
    with pytest.raises(ValueError):
        fit_normalizer([doc], ["dev"])


def test_alignment_no_implicit_reorder_or_shift():
    doc = observation(); sidecar = score(doc, fit_normalizer([doc], ["dev"]))
    assert aligned_weights(sidecar, "dev", doc["timestamps"], doc["hand_ids"], doc["object_ids"]).shape == (8, 2, 2)
    for field, changed in [("sequence_id", "wrong"), ("timestamps", np.array(doc["timestamps"]) + .001),
                           ("hand_ids", ["right", "left"]), ("object_ids", ["lid", "base"])]:
        args = {k: doc[k] for k in ("sequence_id", "timestamps", "hand_ids", "object_ids")}
        args[field] = changed
        with pytest.raises(ValueError):
            aligned_weights(sidecar, **args)


def test_variants_and_cache_identity():
    doc = observation(); norm = fit_normalizer([doc], ["dev"])
    doc["signals"][1][0][0] = [0, 0, 10, 10]
    ref = score(doc, norm)
    const = score(doc, norm, variant="constant")
    shuffled = score(doc, norm, variant="shuffled")
    assert np.allclose(const["weights"], np.mean(ref["weights"]))
    assert np.allclose(np.sort(np.array(shuffled["weights"]), axis=0), np.sort(np.array(ref["weights"]), axis=0))
    assert shuffled == score(doc, norm, variant="shuffled")
    assert np.all(np.array(score(doc, norm, variant="baseline")["weights"]) == 1)
    assert len({score(doc, norm, floor=f)["cache_key"] for f in [0, .25, .5]}) == 3
    changed = copy.deepcopy(doc); changed["signals"][0][0][0][0] = 0
    assert score(changed, norm)["cache_key"] != ref["cache_key"]


def test_signal_extractors():
    a = np.array([[True, False], [False, False]])
    b = np.array([[True, True], [False, False]])
    assert mask_iou(a, b) == .5
    assert np.isnan(mask_iou(a & False, a & False))
    error = reprojection_error([[0, 0], [0, 0]], [[3, 4], [999, 999]], [True, False], [3, 4])
    assert error == 1
    times = np.arange(5, dtype=float)
    positions = np.stack((times ** 2, times * 0, times * 0), axis=-1)
    assert np.allclose(temporal_acceleration(positions, times)[1:-1], 2)
    with pytest.raises(ValueError):
        temporal_acceleration(positions, [0, 1, 1, 3, 4])
