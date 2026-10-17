"""Growing optimization windows; partial windows never accept a full grasp."""

import math
import numpy as np
from .phase_prior import project


def project_pose_and_height(values, lower, upper, profile):
    """Thirteen raw motor parameters plus one independent height in metres."""
    values = np.asarray(values, float)
    if values.shape[-1] != 14 or not np.isfinite(values).all():
        raise ValueError("Need thirteen finite pose parameters and one carry height")
    pose = project(values[..., :13], lower, upper, profile=profile)
    height = np.clip(values[..., 13:14], 0, .025)
    return np.concatenate([pose, height], axis=-1)


def window_end(phase, step_s, previous=None, requested=None):
    if not math.isfinite(step_s) or not 0 <= step_s <= 5:
        raise ValueError(
            "Curriculum step must be finite, between zero and five seconds"
        )
    full = phase["release_start_s"] - 0.05
    start = phase["close_s"] + 0.05
    if full <= start:
        raise ValueError("Empty grasp window")
    if requested is not None:
        if not math.isfinite(requested) or not start + 0.1 <= requested <= full:
            raise ValueError(
                "Requested objective end must cover a nonempty grasp prefix"
            )
        return requested
    if step_s == 0:
        return full
    return min(full, (start if previous is None else previous) + step_s)


def complete(phase, end):
    return end >= phase["release_start_s"] - 0.05 - 1e-8


def initial_sigma(profile, scale=1.0):
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Search variance scale must be in (0, 1]")
    if profile == "local":
        value = [0.035] * 3 + [0.12] * 3 + [0.25] * 7
    elif profile == "grasp_refit":
        value = [0.12] * 3 + [0.4] * 3 + [0.65] * 7
    else:
        raise ValueError("Unknown physical parameter bounds")
    return np.array(value) * scale
