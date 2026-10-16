"""Robot wrist roll about an existing pinch; source/object geometry is immutable."""

import numpy as np
from scipy.spatial.transform import Rotation


def roll_about_contacts(root, contacts, degrees):
    """Rotate a wrist around the two fixed opposing contact points."""
    root, contacts = np.asarray(root, float), np.asarray(contacts, float)
    if (
        root.shape != (7,)
        or contacts.shape != (2, 3)
        or not np.isfinite(np.r_[root, contacts.ravel(), degrees]).all()
        or abs(np.linalg.norm(root[3:]) - 1) > 1e-5
        or not -180 <= degrees <= 180
    ):
        raise ValueError("Invalid pinch-axis wrist roll")
    axis = contacts[1] - contacts[0]
    length = np.linalg.norm(axis)
    if length < 0.008:
        raise ValueError("Pinch contacts are coincident or too close")
    rotation = Rotation.from_rotvec(np.deg2rad(degrees) * axis / length)
    center = contacts.mean(axis=0)
    return np.r_[
        center + rotation.apply(root[:3] - center),
        (rotation * Rotation.from_quat(root[3:])).as_quat(),
    ]
