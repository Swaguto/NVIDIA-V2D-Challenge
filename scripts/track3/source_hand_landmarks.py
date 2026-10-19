"""Explicit MANO fingertip conventions for source-model comparisons.

Both layouts share the same MANO mesh and 16 skeletal joints. Their five
representative surface vertices differ. A coordinate conversion is not a new
observation and must never upgrade source confidence or validity flags.
"""

import numpy as np


TIP_JOINT_IDS = (4, 8, 12, 16, 20)
TIP_VERTEX_IDS = {
    "manotorch21": (745, 317, 444, 556, 673),
    "hamer_smplx21": (744, 320, 443, 554, 671),
}


def convert_mano_landmarks(vertices, joints, *, source_layout, target_layout):
    """Copy 21 landmarks with explicitly selected fingertip surface vertices.

    Vertices and joints must be in the same camera/world frame, scale and hand
    convention. Mirrored left-hand meshes use the same vertex indices. Reject
    incorrectly declared input layouts rather than silently changing their
    meaning. The caller retains original meshes, landmarks and source flags.
    """
    if source_layout not in TIP_VERTEX_IDS or target_layout not in TIP_VERTEX_IDS:
        raise ValueError("Unknown MANO landmark layout")
    v, j = np.asarray(vertices), np.asarray(joints)
    if v.ndim < 2 or v.shape[-2:] != (778, 3) or j.shape != (*v.shape[:-2], 21, 3):
        raise ValueError("Expected matching batch axes for vertices[...,778,3] and joints[...,21,3]")
    if not np.isfinite(v).all() or not np.isfinite(j).all():
        raise ValueError("Landmark conversion requires finite geometry")
    if not np.allclose(j[..., TIP_JOINT_IDS, :], v[..., TIP_VERTEX_IDS[source_layout], :], atol=1e-6, rtol=0):
        raise ValueError("Declared source layout disagrees with its mesh fingertip vertices")
    output = j.copy()
    output[..., TIP_JOINT_IDS, :] = v[..., TIP_VERTEX_IDS[target_layout], :]
    return output
