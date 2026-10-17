"""World-frame angular velocity of a sampled orientation trajectory."""

import numpy as np
from scipy.spatial.transform import Rotation


def reference_angular_velocity(times, quaternions, maximum_speed=10.):
    times, q = np.asarray(times, float), np.asarray(quaternions, float)
    if (times.ndim != 1 or len(times) < 3 or q.shape != (len(times), 4)
            or not np.isfinite(times).all() or not np.isfinite(q).all()
            or np.any(np.diff(times) <= 0) or np.any(np.linalg.norm(q, axis=1) < 1e-8)
            or not np.isfinite(maximum_speed) or not 0 < maximum_speed <= 20):
        raise ValueError("Need a finite ordered quaternion trajectory and bounded speed")
    rotations = Rotation.from_quat(q)
    dt = np.diff(times)
    segments = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt[:, None]
    result = np.empty((len(times), 3))
    result[0] = segments[0]
    result[1:-1] = (dt[1:, None] * segments[:-1] + dt[:-1, None] * segments[1:]) / (dt[1:] + dt[:-1])[:, None]
    # The final source command is a hold, without extrapolation.
    result[-1] = 0
    if np.linalg.norm(result, axis=1).max() > maximum_speed:
        raise ValueError("Reference angular velocity exceeds the declared motor plan")
    return result
