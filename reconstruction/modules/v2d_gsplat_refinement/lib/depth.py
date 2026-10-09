"""Load sparse inverse-depth observations without inventing depth at holes."""
from pathlib import Path

import numpy as np
from PIL import Image


def load_inverse_depth(path: str | Path, width: int, height: int) -> np.ndarray:
    """Decode ``65535 / (z + 1)``; zero and nonpositive depths are unknown.

    Nearest-neighbour resizing preserves individual stereo measurements and
    their validity. Interpolating encoded zero (unknown) with a valid sample
    would fabricate a finite, usually distant, surface at the hole boundary.
    """
    with Image.open(path) as image:
        if image.mode not in ("I;16", "I;16L", "I;16B", "I"):
            raise ValueError(f"Expected a single-channel inverse-u16 depth PNG: {path}")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.NEAREST)
        pixels = np.asarray(image, dtype=np.float32)
    depth = np.full(pixels.shape, np.inf, dtype=np.float32)
    valid = (pixels > 0) & (pixels < 65535)
    depth[valid] = 65535.0 / pixels[valid] - 1.0
    return depth
