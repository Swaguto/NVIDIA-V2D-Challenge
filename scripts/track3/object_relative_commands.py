"""Propose robot wrist commands through an unknown object-holding interval.

This does not reconstruct observed human geometry or certify contact. Callers
must explicitly identify the object-hold assumption and retain the original
source validity. New commands still need source, motion, native collision,
swept-contact and grasp checks before training.
"""
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class InferredCommandGap(NamedTuple):
    commands: np.ndarray
    inferred_command_mask: np.ndarray
    original_source_valid: np.ndarray


def infer_object_relative_gap(commands, object_transforms, timestamps_s,
                              gap_start, gap_stop, *, source_valid,
                              contact_assumption):
    """Fill root channels in [gap_start,gap_stop); keep fingers and anchors exact.

    Commands have shape [T,14]: position, xyzw quaternion and seven robot finger
    joints. Object transforms use the same world frame. Both bracketing source
    frames must be valid and every interior source frame invalid. Validity is
    only the original caller-provided source flag; it need not imply direct or
    metric observation, and is never upgraded here.

    Translation and orientation interpolate between the two wrist transforms
    expressed in the object frame, then follow the supplied object trajectory.
    The returned mask identifies inferred robot commands, not measured contact.
    """
    if contact_assumption != "inferred_object_relative_hold":
        raise ValueError("Declare an inferred object-relative hold explicitly")
    q = np.asarray(commands, dtype=float)
    objects = np.asarray(object_transforms, dtype=float)
    times = np.asarray(timestamps_s, dtype=float)
    observed = np.asarray(source_valid)
    if q.ndim != 2 or q.shape[1:] != (14,) or len(q) < 3:
        raise ValueError("Need complete single-hand robot commands[T,14]")
    n = len(q)
    if objects.shape != (n, 4, 4) or times.shape != (n,) or observed.shape != (n,):
        raise ValueError("Object, command, source and clock axes must match")
    if observed.dtype != np.bool_:
        raise ValueError("Source validity must be explicit boolean flags")
    if not all(np.isfinite(a).all() for a in (q, objects, times)):
        raise ValueError("Commands, object transforms and source clock must be finite")
    if (np.diff(times) <= 0).any():
        raise ValueError("Source times must strictly increase")
    if not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Robot wrist quaternions must be normalized xyzw")
    if not np.allclose(objects[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("Objects must be rigid homogeneous transforms")
    r = objects[:, :3, :3]
    orthogonal = np.einsum("tki,tkj->tij", r, r, optimize=False)
    # Match the existing automatic gap selector's rigid-input tolerances.
    if (not np.allclose(orthogonal, np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(r), 1, atol=1e-5, rtol=0)):
        raise ValueError("Object orientations must be proper rotations")
    if (type(gap_start) is not int or type(gap_stop) is not int
            or not 1 <= gap_start < gap_stop < n):
        raise ValueError("Need a nonempty gap with both bracketing frames")
    left, right = gap_start - 1, gap_stop
    if not observed[[left, right]].all() or observed[gap_start:gap_stop].any():
        raise ValueError("Only fill an unknown interval bracketed by valid source frames")
    anchors = np.array([left, right])
    relative_position = np.einsum(
        "tji,tj->ti", r[anchors], q[anchors, :3] - objects[anchors, :3, 3],
        optimize=False,
    )
    relative_rotation = np.einsum(
        "tji,tjk->tik", r[anchors], Rotation.from_quat(q[anchors, 3:7]).as_matrix(),
        optimize=False,
    )
    gap = np.arange(gap_start, gap_stop)
    alpha = (times[gap] - times[left]) / (times[right] - times[left])
    positions = (1 - alpha[:, None]) * relative_position[0] + alpha[:, None] * relative_position[1]
    orientations = Slerp(times[anchors], Rotation.from_matrix(relative_rotation))(times[gap]).as_matrix()
    result = q.copy()
    result[gap, :3] = np.einsum("tij,tj->ti", r[gap], positions, optimize=False) + objects[gap, :3, 3]
    result[gap, 3:7] = Rotation.from_matrix(
        np.einsum("tij,tjk->tik", r[gap], orientations, optimize=False)
    ).as_quat()
    inferred = np.zeros(n, dtype=bool)
    inferred[gap] = True
    return InferredCommandGap(result, inferred, observed.copy())
