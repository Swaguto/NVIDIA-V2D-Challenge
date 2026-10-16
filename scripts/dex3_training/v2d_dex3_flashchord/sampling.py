"""Interpolate inferred commands without rewriting source evidence or time.

Only integer upsampling is allowed. Source knots remain exactly represented;
continuous poses use quaternion SLERP and discrete phase/capture flags are held
from the preceding knot. Intermediate confidence is the AND of its neighbours.
"""

import numpy as np


def interpolate_quaternions(a, b, weight):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(abs(dot), 0, 1))
    sine = np.sin(angle)
    denominator = np.where(sine > 1e-8, sine, 1)
    q = (np.sin((1 - weight) * angle) * a + np.sin(weight * angle) * b) / denominator
    q = np.where(sine > 1e-8, q, (1 - weight) * a + weight * b)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample_targets(data, count):
    source_count = len(data["frame_indices"])
    if count == source_count:
        return data
    ratio, remainder = divmod(count - 1, source_count - 1)
    if ratio < 1 or remainder:
        raise ValueError(
            "Only integer command upsampling with unchanged duration is allowed"
        )
    coordinate = np.arange(count, dtype=float) / ratio
    lo = np.floor(coordinate).astype(int)
    hi = np.minimum(lo + 1, source_count - 1)
    weight = coordinate - lo
    result = {}
    for key, value in data.items():
        if value.shape[0] != source_count:
            raise ValueError("Command time axes differ")
        w = weight.reshape((-1,) + (1,) * (value.ndim - 1))
        if key == "confidence_valid":
            result[key] = value[lo] & np.where(w == 0, value[lo], value[hi])
        elif key in {"phase_index", "capture_settled_registration"}:
            result[key] = value[lo].copy()
        else:
            result[key] = value[lo] * (1 - w) + value[hi] * w
        if key in {"robot_qpos", "object_poses", "grasp_pose_object_xyzw"}:
            result[key][..., 3:7] = interpolate_quaternions(
                value[lo, ..., 3:7], value[hi, ..., 3:7], w
            )
        # Preserve the original knots byte-for-byte, including quaternion signs.
        result[key][::ratio] = value
    return result
