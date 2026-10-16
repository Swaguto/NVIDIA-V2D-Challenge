"""Growing optimization windows; partial windows never accept a full grasp."""

import math


def window_end(phase, step_s, previous=None):
    if not math.isfinite(step_s) or not 0 <= step_s <= 5:
        raise ValueError(
            "Curriculum step must be finite, between zero and five seconds"
        )
    full = phase["release_start_s"] - 0.05
    start = phase["close_s"] + 0.05
    if full <= start:
        raise ValueError("Empty grasp window")
    if step_s == 0:
        return full
    return min(full, (start if previous is None else previous) + step_s)


def complete(phase, end):
    return end >= phase["release_start_s"] - 0.05 - 1e-8
