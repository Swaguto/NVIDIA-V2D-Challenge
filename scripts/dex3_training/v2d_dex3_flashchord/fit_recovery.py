"""Discard invalid proposals; only a wholly clean replay can be scored."""

import numpy as np


def failed_worlds(arrays, cap_ok):
    valid = np.asarray(cap_ok, bool).copy()
    if valid.ndim != 1:
        raise ValueError("Need one native-cap flag per world")
    for values in arrays:
        values = np.asarray(values)
        if values.ndim < 2 or values.shape[1] != len(valid):
            raise ValueError("Readback must have frame and world axes")
        valid &= np.isfinite(values).all(axis=tuple(i for i in range(values.ndim) if i != 1))
    return np.flatnonzero(~valid)


def retry_candidates(candidates, invalid_worlds, incumbent):
    candidates = np.asarray(candidates)
    worlds = np.asarray(invalid_worlds, int)
    if candidates.ndim != 2 or not np.isfinite(candidates).all() or worlds.ndim != 1:
        raise ValueError("Invalid candidate recovery inputs")
    if not len(worlds) or worlds.min() < 0 or worlds.max() >= 2 * len(candidates):
        raise ValueError("Invalid paired-world indices")
    pairs = np.unique(worlds // 2)
    if len(pairs) > max(1, int(len(candidates) * .1)) or len(pairs) >= len(candidates):
        raise ValueError("Too many invalid proposals; search aborted")
    incumbent = np.asarray(incumbent)
    if incumbent.shape != candidates.shape[1:] or not np.isfinite(incumbent).all():
        raise ValueError("Need a finite incumbent for the clean replay")
    replay = candidates.copy()
    replay[pairs] = incumbent
    return replay, pairs
