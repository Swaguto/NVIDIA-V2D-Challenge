"""Synthetic wiring demonstration only. Produces no robotics performance claim."""
from pathlib import Path
import numpy as np
from v2d_reliability.core import SIGNALS, fit_normalizer, score, aligned_weights
from v2d_reliability.io import write_json
from v2d_reliability.signals import extract

out = Path("runs/synthetic_smoke")
t, h, b, k = 8, 2, 2, 5
times = np.arange(t) / 30
valid_keypoints = np.ones((t, h, b, k), dtype=bool)
masks = np.ones((t, h, b, 4, 4), dtype=bool)
keypoints = np.zeros((t, h, b, k, 2))
positions = np.zeros((t, h, b, 3))
values, valid = extract(valid_keypoints, masks, masks, keypoints, keypoints, positions, times, [640, 480])
doc = {"schema_version": "observations_v1", "sequence_id": "synthetic_dev",
       "timestamps": times.tolist(), "hand_ids": ["left", "right"], "object_ids": ["base", "lid"],
       "signal_names": list(SIGNALS), "signals": values.tolist(), "valid": valid.tolist()}
normalizer = fit_normalizer([doc], ["synthetic_dev"])
write_json(out / "normalizer.json", normalizer)
doc["signals"][3][0][1] = [0, 0, 10, 10]
write_json(out / "observations.json", doc)
for variant in ("baseline", "reliability", "constant", "shuffled"):
    sidecar = score(doc, normalizer, variant=variant)
    aligned_weights(sidecar, doc["sequence_id"], times, doc["hand_ids"], doc["object_ids"])
    write_json(out / f"{variant}.json", sidecar)
write_json(out / "summary.json", {"synthetic_only": True, "robot_training_executed": False,
           "corrupted_cell_weight": score(doc, normalizer)["weights"][3][0][1],
           "result": "Extraction, fitting, scoring, ablations and alignment completed"})
print(f"Synthetic smoke artifacts: {out.resolve()}")
