"""Read-only references to contact-time native data, with explicit body mapping."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NativeContactReadout:
    # These are live solver buffers. Consumers must never write through them.
    data: object
    body_to_newton: object


def invert_body_mapping(body_to_newton, bodies_per_world):
    """Map global Newton body IDs to local native body IDs, rejecting ambiguity."""
    mapping = np.asarray(body_to_newton)
    if mapping.ndim != 2 or not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("Native body mapping must be a two-dimensional integer array")
    if bodies_per_world <= 0 or not mapping.shape[0]:
        raise ValueError("Native body mapping needs worlds and positive body count")
    inverse = np.full(mapping.shape[0] * bodies_per_world, -1, dtype=np.int32)
    for world, row in enumerate(mapping):
        for native_body, newton_body in enumerate(row):
            if newton_body == -1:
                continue
            if not world * bodies_per_world <= newton_body < (world + 1) * bodies_per_world:
                raise ValueError("Native body mapping crosses world boundaries")
            if inverse[newton_body] != -1:
                raise ValueError("Native body mapping contains duplicate Newton bodies")
            inverse[newton_body] = native_body
    return inverse
