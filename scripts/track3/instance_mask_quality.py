"""Detect ambiguous fused masks for distinct numbered object instances."""

import itertools

import numpy as np
from object_prompts import numbered_groups


def numbered_mask_conflicts(names, masks, threshold=0.70):
    """Return conflicting indices; overlap cannot establish which is occluded.

    Distinct visible objects cannot both explain the same foreground pixels.
    Mark both observations unknown when most of the smaller mask is shared.
    This does not infer an occlusion order or assign a canonical identity.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Overlap threshold must lie in (0, 1]")
    conflicts = set()
    for group in numbered_groups(names):
        for a, b in itertools.combinations(group, 2):
            x, y = masks.get(names[a]), masks.get(names[b])
            if x is None or y is None:
                continue
            x, y = np.asarray(x, bool), np.asarray(y, bool)
            if x.ndim != 2 or x.shape != y.shape:
                raise ValueError("Instance masks must share the calibrated pixel grid")
            smaller = min(int(x.sum()), int(y.sum()))
            if smaller and int((x & y).sum()) / smaller > threshold:
                conflicts.update((a, b))
    return conflicts
