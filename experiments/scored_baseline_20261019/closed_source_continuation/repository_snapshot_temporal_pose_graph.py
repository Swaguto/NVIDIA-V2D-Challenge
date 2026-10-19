"""Finite joint-pose reachability with actual speed and added-motion bounds.

This is an exact graph calculation for the supplied finite banks. It does not
certify unsampled poses, source truth, root motion, collision or dynamics.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalPoseReachability:
    path_indices: tuple[int, ...] | None
    path_cost: float | None
    forward_pair_counts: tuple[int, ...]
    backward_pair_counts: tuple[int, ...]
    viable_node_indices: tuple[tuple[int, ...], ...]
    empty_bank_indices: tuple[int, ...]


def finite_temporal_reachability(
    banks,
    original_reference,
    timestamps_s,
    lower_limits,
    upper_limits,
    velocity_limits,
    *,
    component_cap,
    adjacent_added_cap,
    second_added_cap,
    node_costs=None,
    tolerance=1e-9,
):
    """Return minimum-cost paths and bidirectional viability for finite banks.

    Bounds on adjacent and second differences apply to corrections against the
    supplied ORIGINAL reference. Velocity limits apply to actual joint angles
    and timestamp differences. Invalid input poses fail, rather than being
    silently removed. Empty banks are valid evidence of finite-bank failure.
    A pair of consecutive node indices is the state needed for second-order
    constraints; greedy node selection is not equivalent to this recurrence.
    """
    original = np.asarray(original_reference, dtype=float)
    if original.ndim != 2 or len(original) < 3 or not np.isfinite(original).all():
        raise ValueError("original_reference must be finite (T,D), T>=3")
    count, dimensions = original.shape
    if dimensions == 0 or len(banks) != count:
        raise ValueError("A bank is required for every nonempty reference row")
    timestamps = np.asarray(timestamps_s, dtype=float)
    if timestamps.shape != (count,) or not np.isfinite(timestamps).all() or (np.diff(timestamps) <= 0).any():
        raise ValueError("Timestamps must be finite and strictly increasing")
    lower = np.asarray(lower_limits, dtype=float)
    upper = np.asarray(upper_limits, dtype=float)
    velocity = np.asarray(velocity_limits, dtype=float)
    if (
        any(a.shape != (dimensions,) or not np.isfinite(a).all() for a in [lower, upper, velocity])
        or (lower > upper).any()
        or (velocity <= 0).any()
    ):
        raise ValueError("Explicit finite position and positive velocity limits are required")
    if any(not np.isfinite(x) or x < 0 for x in [component_cap, adjacent_added_cap, second_added_cap, tolerance]):
        raise ValueError("Correction caps and tolerance must be finite and nonnegative")
    poses = [np.asarray(bank, dtype=float) for bank in banks]
    if any(p.ndim != 2 or p.shape[1] != dimensions or not np.isfinite(p).all() for p in poses):
        raise ValueError("Each finite bank must have shape (K,D), including K=0")
    deltas = [p - original[i] for i, p in enumerate(poses)]
    for p, d in zip(poses, deltas, strict=True):
        if ((p < lower - tolerance) | (p > upper + tolerance)).any() or (abs(d) > component_cap + tolerance).any():
            raise ValueError("A pose exceeds original component or actual position bounds")
    costs = [np.zeros(len(p)) for p in poses] if node_costs is None else [np.asarray(c, dtype=float) for c in node_costs]
    if len(costs) != count or any(c.shape != (len(p),) or not np.isfinite(c).all() or (c < 0).any() for p, c in zip(poses, costs, strict=True)):
        raise ValueError("Finite nonnegative costs must bind every node")
    edges = []
    for i in range(count - 1):
        speed = abs(poses[i + 1][None, :, :] - poses[i][:, None, :])
        correction = abs(deltas[i + 1][None, :, :] - deltas[i][:, None, :])
        edges.append(((speed <= velocity * (timestamps[i + 1] - timestamps[i]) + tolerance) & (correction <= adjacent_added_cap + tolerance)).all(axis=-1))
    triples = []
    for i in range(count - 2):
        second = deltas[i][:, None, None, :] - 2 * deltas[i + 1][None, :, None, :] + deltas[i + 2][None, None, :, :]
        triples.append((abs(second) <= second_added_cap + tolerance).all(axis=-1))
    forward = [edges[0].copy()]
    path_cost = np.where(edges[0], costs[0][:, None] + costs[1][None, :], np.inf)
    parents = [None]
    for i in range(1, count - 1):
        reachable = np.zeros(edges[i].shape, dtype=bool)
        next_cost = np.full(edges[i].shape, np.inf)
        parent = np.full(edges[i].shape, -1, dtype=int)
        for b, c in zip(*np.nonzero(edges[i]), strict=True):
            allowed = forward[-1][:, b] & triples[i - 1][:, b, c]
            if allowed.any():
                reachable[b, c] = True
                choices = np.where(allowed, path_cost[:, b], np.inf)
                a = int(np.argmin(choices))
                next_cost[b, c] = choices[a] + costs[i + 1][c]
                parent[b, c] = a
        forward.append(reachable);parents.append(parent);path_cost = next_cost
    backward = [None] * (count - 1)
    backward[-1] = edges[-1].copy()
    for i in range(count - 3, -1, -1):
        reachable = np.zeros(edges[i].shape, dtype=bool)
        for a, b in zip(*np.nonzero(edges[i]), strict=True):
            reachable[a, b] = (backward[i + 1][b] & triples[i][a, b]).any()
        backward[i] = reachable
    viable = [set() for _ in poses]
    for i, (left, right) in enumerate(zip(forward, backward, strict=True)):
        for a, b in zip(*np.nonzero(left & right), strict=True):
            viable[i].add(int(a));viable[i + 1].add(int(b))
    path = None;best = None
    if np.isfinite(path_cost).any():
        a, b = np.unravel_index(np.argmin(path_cost), path_cost.shape)
        best = float(path_cost[a, b]);indices = [0] * count
        indices[-2:] = [int(a), int(b)]
        for i in range(count - 2, 0, -1):
            indices[i - 1] = int(parents[i][indices[i], indices[i + 1]])
        path = tuple(indices)
    return TemporalPoseReachability(
        path, best, tuple(int(f.sum()) for f in forward),
        tuple(int(b.sum()) for b in backward),
        tuple(tuple(sorted(v)) for v in viable),
        tuple(i for i, p in enumerate(poses) if len(p) == 0),
    )
