"""Load frozen native hand priors with an explicit frame-index convention."""

import json
from pathlib import Path


def load_native_priors(folder, fitted_axis, *, source_frame_indices=False):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError("Native hand-prior directory is unavailable: " + str(folder))
    raw = {"left": {}, "right": {}}
    for path in sorted(folder.rglob("*.json")):
        record = json.loads(path.read_text())
        if "mano" not in record or "frame_idx" not in record:
            continue
        side = "right" if record["is_right"] else "left"
        if not source_frame_indices:
            native_index = record["frame_idx"]
            if type(native_index) is not int or not 0 <= native_index < len(fitted_axis):
                raise ValueError("Native pose is outside the recorded fitted axis")
            record["inference_frame_idx"] = native_index
            record["frame_idx"] = fitted_axis[native_index]["id"]
        if record["frame_idx"] in raw[side]:
            raise ValueError("Ambiguous same-side native hand")
        raw[side][record["frame_idx"]] = record
    if not any(raw.values()):
        raise ValueError("No native hand poses were loaded; preserve and repair the conversion path")
    return raw
