"""Move MANO parameters between calibrated CV camera frames.

The root rotates about the model's shaped rest wrist, which is generally not
the origin. This operation changes coordinates only. It does not make a model
prediction directly observed or establish its metric accuracy.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def transform_mano_camera(pose, translation, rest_wrist, target_from_source, *, is_right):
    """Return axis-angle pose and translation in another camera frame.

    ``rest_wrist`` is the shaped MANO_RIGHT rest joint for the same fixed betas.
    Left records use the repository's HaMeR convention: mirror the right model
    in x after skinning, then add camera translation. Shape, finger pose and
    scale remain unchanged. Center-based mesh scaling also commutes with this
    conversion. Callers must retain the original observation/validity flags.

    Published factory matrices can have small float-rounding deviations from
    SO(3); reject larger deviations. Axis-angle storage projects only the root
    rotation to SO(3). Check resulting vertex/joint closure against the original
    factory transform when consuming real model parameters.
    """
    p = np.asarray(pose, dtype=float)
    t = np.asarray(translation, dtype=float)
    root = np.asarray(rest_wrist, dtype=float)
    frame = np.asarray(target_from_source, dtype=float)
    if p.shape != (48,) or t.shape != (3,) or root.shape != (3,) or frame.shape != (4, 4):
        raise ValueError("Camera conversion requires pose[48], translation[3], wrist[3], frame[4,4]")
    if type(is_right) is not bool:
        raise ValueError("Hand side must be an explicit boolean")
    if not all(np.isfinite(x).all() for x in (p, t, root, frame)):
        raise ValueError("Camera conversion inputs must be finite")
    if not np.allclose(frame[3], [0, 0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("Camera transform must be homogeneous")
    q = frame[:3, :3]
    gram = np.einsum("ji,jk->ik", q, q)
    if not np.allclose(gram, np.eye(3), atol=1e-5, rtol=0) or abs(np.linalg.det(q) - 1) > 1e-5:
        raise ValueError("Camera transform must be a proper rigid rotation")
    mirror = np.array([1., 1., 1.]) if is_right else np.array([-1., 1., 1.])
    wrist = root * mirror
    local_rotation = q * mirror[:, None] * mirror[None, :]
    new_pose = p.copy()
    new_pose[:3] = Rotation.from_matrix(
        np.einsum("ij,jk->ik", local_rotation, Rotation.from_rotvec(p[:3]).as_matrix())
    ).as_rotvec()
    new_translation = np.einsum("ij,j->i", q, t + wrist) + frame[:3, 3] - wrist
    return new_pose, new_translation
