"""Bounded upward scene initialization and robot-reference clearance proposal.

Uses recorded cooked shapes, preserves original source arrays, and never writes
object state during a simulation. Face-axis SAT gives a conservative clearance
proposal; it is not a continuous collision certificate or a reconstruction score.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from cooked_grasp_audit import world_convexes, convex_escape


def upward_clearance(lower, upper, gap=0.0002, cap=0.06):
    """Smallest nonnegative upward shift outside all face-axis overlap intervals."""
    intervals = []
    for a in lower:
        for b in upper:
            # Vertical movement cannot close a separated horizontal AABB axis.
            if np.any(a.min(0)[:2] - b.max(0)[:2] > gap) or np.any(
                b.min(0)[:2] - a.max(0)[:2] > gap
            ):
                continue
            axes = np.r_[
                ConvexHull(a).equations[:, :3],
                ConvexHull(b).equations[:, :3],
                np.eye(3),
            ]
            axes = np.unique(np.round(axes, 9), axis=0)
            origin = a.mean(0)
            ap = np.einsum("ij,kj->ik", a - origin, axes)
            bp = np.einsum("ij,kj->ik", b - origin, axes)
            first = ap.min(0) - bp.max(0) - gap
            last = ap.max(0) - bp.min(0) + gap
            nz = axes[:, 2]
            horizontal = abs(nz) < 1e-9
            if np.any((first[horizontal] > 0) | (last[horizontal] < 0)):
                continue
            first, last, nz = first[~horizontal], last[~horizontal], nz[~horizontal]
            endpoints = np.c_[first / nz, last / nz]
            lo, hi = float(endpoints.min(1).max()), float(endpoints.max(1).min())
            if lo <= hi and hi >= 0:
                intervals.append((lo, hi))
    result = 0.0
    for lo, hi in sorted(intervals):
        if lo <= result <= hi:
            result = hi + 1e-7
    if result > cap:
        raise ValueError(f"Scene clearance requires {result:.6f} m, above {cap} m cap")
    return result


def rate_bounded_majorant(values, times, speed=0.05):
    """Raise neighbouring references so clearance remains and rate is bounded."""
    result = np.asarray(values, float).copy()
    for i in range(1, len(result)):
        result[i] = max(result[i], result[i - 1] - speed * (times[i] - times[i - 1]))
    for i in range(len(result) - 2, -1, -1):
        result[i] = max(result[i], result[i + 1] - speed * (times[i + 1] - times[i]))
    return result


def verify_initial_scene(cooked_file, projection, object_poses):
    """Compare fresh cooked pieces and independently check the initialized scene."""
    colliders = json.loads(Path(cooked_file).read_text())["colliders"]
    actual, errors = [], []
    for oi, pose in enumerate(object_poses):
        local = [
            v
            for c in colliders
            if c["asset"] == f"object-{oi}"
            for v in world_convexes(c, [0, 0, 0, 0, 0, 0, 1])
        ]
        expected = [
            np.asarray(v) for v in projection["expected_body_local_convexes"][oi]
        ]
        if len(local) != len(expected):
            raise ValueError("Fresh object cooked-piece count changed")
        costs = cdist([v.mean(0) for v in local], [v.mean(0) for v in expected])
        ii, jj = linear_sum_assignment(costs)
        for i, j in zip(ii, jj, strict=True):
            d = cdist(local[i], expected[j])
            errors.append(float(max(d.min(0).max(), d.min(1).max())))
        actual.append([Rotation.from_quat(pose[3:]).apply(v) + pose[:3] for v in local])
    if max(errors) > 1e-6:
        raise ValueError("Fresh cooked shapes differ from the scene-clearance proposal")
    plane_gap = (
        min(v[:, 2].min() for vs in actual for v in vs)
        - projection["inferred_support_height_stage_m"]
    )
    depth = max(convex_escape(a, b)[0] for a in actual[0] for b in actual[1])
    if plane_gap < -1e-6 or depth > 1e-7:
        raise ValueError(
            f"Initial scene remains intersecting: plane gap {plane_gap}, object depth {depth}"
        )
    return dict(
        maximum_cooked_shape_difference_m=max(errors),
        minimum_initial_table_gap_m=float(plane_gap),
        maximum_initial_object_piece_axis_depth_m=float(depth),
        independently_checked=True,
    )


def propose(case, inputs, cooked_file):
    manifest = json.loads((inputs / "manifest.json").read_text())
    stage = Rotation.from_matrix(
        manifest["stage_rotation_from_inferred_support_normal"]
    )
    with np.load(case / "bundle/episode.npz", allow_pickle=False) as data:
        source = {k: data[k] for k in data.files}
    with np.load(inputs / "inferred-support-sheet.npz", allow_pickle=False) as sheet:
        vertices = stage.apply(sheet["vertices_world_m"])
    if np.ptp(vertices[:, 2]) > 1e-6:
        raise ValueError(
            "Scene projection currently requires a horizontal inferred sheet"
        )
    plane = float(vertices[:, 2].mean())
    colliders = json.loads(cooked_file.read_text())["colliders"]
    local = [
        [
            v
            for c in colliders
            if c["asset"] == f"object-{oi}"
            for v in world_convexes(c, [0, 0, 0, 0, 0, 0, 1])
        ]
        for oi in [0, 1]
    ]
    times = source["timestamps_s"]
    shapes = [[], []]
    for oi in [0, 1]:
        for matrix in source[f"object_{oi}_world"]:
            rotation = stage * Rotation.from_matrix(matrix[:3, :3])
            position = stage.apply(matrix[:3, 3])
            shapes[oi].append([rotation.apply(v) + position for v in local[oi]])
    corrections = np.zeros((len(times), 2))
    for oi in [0, 1]:
        corrections[:, oi] = [
            max(0, plane + 0.0002 - min(v[:, 2].min() for v in vs)) for vs in shapes[oi]
        ]
    corrections[:, 0] = rate_bounded_majorant(corrections[:, 0], times)
    # This two-object adapter binds object 0 below object 1 at initialization.
    if (
        source["object_0_world"][0, :3, 3] @ stage.as_matrix()[2]
        >= source["object_1_world"][0, :3, 3] @ stage.as_matrix()[2]
    ):
        raise ValueError("Expected the automatically bound pot below its lid")
    for row in range(len(times)):
        lower = [v + [0, 0, corrections[row, 0]] for v in shapes[0][row]]
        upper = [v + [0, 0, corrections[row, 1]] for v in shapes[1][row]]
        corrections[row, 1] += upward_clearance(
            lower, upper, cap=0.06 - corrections[row, 1]
        )
    corrections[:, 1] = rate_bounded_majorant(corrections[:, 1], times)
    for row in range(len(times)):
        if (
            upward_clearance(
                [v + [0, 0, corrections[row, 0]] for v in shapes[0][row]],
                [v + [0, 0, corrections[row, 1]] for v in shapes[1][row]],
            )
            > 1e-8
        ):
            raise ValueError("Rate smoothing reintroduced object-reference overlap")
    if corrections.max() > 0.06 or not np.isfinite(corrections).all():
        raise ValueError("Scene correction exceeds the explicit 6 cm diagnostic bound")
    return dict(
        schema="v2d.cooked-scene-projection.v1",
        source_times_s=times.tolist(),
        upward_corrections_m=corrections.tolist(),
        maximum_correction_m=float(corrections.max()),
        maximum_vertical_rate_m_per_source_s=0.05,
        cap_m=0.06,
        axis_clearance_m=0.0002,
        inferred_support_height_stage_m=plane,
        source_cooked_sha256=hashlib.sha256(cooked_file.read_bytes()).hexdigest(),
        expected_body_local_convexes=[[v.tolist() for v in vs] for vs in local],
        original_source_arrays_changed=False,
        physical_task_accepted=False,
        scope="inferred initialization and robot command correction only; original source object trajectories remain the error reference; no object pose writes after initialization",
    )
