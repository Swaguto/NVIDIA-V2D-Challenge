"""Fail with named diagnostics rather than silently accepting nonfinite training."""

import math
from numbers import Real


def nonfinite_fields(metrics):
    return {
        name: str(value)
        for name, value in metrics.items()
        if isinstance(value, Real) and not math.isfinite(value)
    }
