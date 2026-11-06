"""Versioned sidecars; fit normalizers only on declared development sequences."""
import logging
import numpy as np
from .io import identity

SIGNALS = ("visibility", "mask_iou", "reprojection_error", "temporal_acceleration")
SCHEMA = "reliability_v1"


def validate_observations(doc):
    if doc.get("schema_version") != "observations_v1":
        raise ValueError("unsupported observation schema")
    if not isinstance(doc.get("sequence_id"), str) or not doc["sequence_id"]:
        raise ValueError("sequence_id required")
    for key in ("hand_ids", "object_ids"):
        values = doc.get(key, [])
        if not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError(f"nonempty unique {key} required")
    if any(h not in ("left", "right") for h in doc["hand_ids"]):
        raise ValueError("hand_ids must be left/right")
    times = np.asarray(doc["timestamps"], float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    expected = (len(times), len(doc["hand_ids"]), len(doc["object_ids"]), len(SIGNALS))
    values = np.asarray(doc["signals"], float)
    mask_raw = np.asarray(doc["valid"])
    if values.shape != expected or mask_raw.shape != expected or mask_raw.dtype != bool:
        raise ValueError(f"signals/boolean validity must have shape {expected}")
    if not np.isfinite(values).all():
        raise ValueError("use zero plus valid=false for missing signals")
    if np.any((values[..., :2][mask_raw[..., :2]] < 0) | (values[..., :2][mask_raw[..., :2]] > 1)):
        raise ValueError("visibility and IoU must lie in [0,1]")
    if np.any(values[..., 2:][mask_raw[..., 2:]] < 0):
        raise ValueError("errors cannot be negative")
    if tuple(doc.get("signal_names", [])) != SIGNALS:
        raise ValueError("signal ordering mismatch")
    return values, mask_raw


def fit_normalizer(observations, development_ids):
    docs = list(observations)
    allowed = set(development_ids)
    if not docs or any(d["sequence_id"] not in allowed for d in docs):
        raise ValueError("normalization inputs must belong to the development split")
    if len({d["sequence_id"] for d in docs}) != len(docs):
        raise ValueError("duplicate normalization sequence")
    arrays = [validate_observations(d) for d in docs]
    scales = []
    for i in range(4):
        samples = np.concatenate([v[..., i][m[..., i]] for v, m in arrays])
        scales.append(float(max(np.quantile(samples, .95), 1e-8)) if len(samples) else None)
    return {"schema_version": "normalizer_v1", "signal_names": list(SIGNALS),
            "scales": scales, "development_ids": sorted(d["sequence_id"] for d in docs),
            "input_hashes": [identity(d) for d in docs]}


def score(doc, normalizer, floor=.25, variant="reliability", seed=17):
    values, valid = validate_observations(doc)
    if floor not in (0, .25, .5):
        raise ValueError("weight floor must be 0, 0.25 or 0.5")
    if normalizer.get("schema_version") != "normalizer_v1" or tuple(normalizer["signal_names"]) != SIGNALS:
        raise ValueError("normalizer schema mismatch")
    scales = normalizer["scales"]
    if len(scales) != 4 or any(s is not None and (not np.isfinite(s) or s <= 0) for s in scales):
        raise ValueError("invalid normalizer scales")
    valid = valid.copy()
    quality = values.copy()
    for i in range(4):
        if scales[i] is None:
            valid[..., i] = False
        elif i >= 2:
            quality[..., i] = np.exp(-values[..., i] / scales[i])
    count = valid.sum(-1)
    reliability = np.divide(np.where(valid, quality, 0).sum(-1), count,
                            out=np.ones(count.shape), where=count > 0)
    if np.any(count == 0):
        logging.warning("%s: %d all-missing cells use baseline weight", doc["sequence_id"], (count == 0).sum())
    weights = floor + (1 - floor) * reliability
    if variant == "baseline":
        weights = np.ones_like(weights)
    elif variant == "constant":
        weights = np.full_like(weights, weights.mean())
    elif variant == "shuffled":
        # One common permutation preserves bimanual/object correlation per frame.
        weights = weights[np.random.default_rng(seed).permutation(len(weights))]
    elif variant != "reliability":
        raise ValueError("unknown variant")
    config = {"floor": floor, "variant": variant, "seed": seed}
    return {**doc, "schema_version": SCHEMA, "effective_valid": valid.tolist(),
            "weights": weights.tolist(), "normalizer_hash": identity(normalizer),
            "configuration": config, "input_hash": identity(doc),
            "cache_key": identity({"implementation": "0.1.0", "input": doc,
                                   "normalizer": normalizer, "configuration": config})}


def aligned_weights(sidecar, sequence_id, timestamps, hand_ids, object_ids):
    if sidecar.get("schema_version") != SCHEMA:
        raise ValueError("unsupported sidecar schema")
    validate_observations({**sidecar, "schema_version": "observations_v1"})
    if sidecar["sequence_id"] != sequence_id:
        raise ValueError("sequence mismatch")
    times = np.asarray(timestamps, float)
    if times.shape != np.shape(sidecar["timestamps"]) or not np.allclose(times, sidecar["timestamps"], atol=1e-8, rtol=0):
        raise ValueError("timestamp mismatch: explicit resampling is required")
    if list(hand_ids) != sidecar["hand_ids"] or list(object_ids) != sidecar["object_ids"]:
        raise ValueError("hand/object order mismatch")
    weights = np.asarray(sidecar["weights"], float)
    if weights.shape != (len(times), len(hand_ids), len(object_ids)):
        raise ValueError("weight dimensions mismatch")
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("invalid weights")
    return weights
