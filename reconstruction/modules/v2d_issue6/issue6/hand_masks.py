"""Export metric masks without destroying the ungated visual segmentation."""

import os
from pathlib import Path

import numpy as np
from PIL import Image


def export_metric_mask(source, destination, directly_observed, size):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if directly_observed:
        mask = np.asarray(Image.open(source).convert("L")).copy()
        if mask.shape != (size[1], size[0]):
            raise ValueError("SAM2 mask is not on the source image grid")
    else:
        mask = np.zeros((size[1], size[0]), np.uint8)
    # Replacing a fresh file prevents a previous hardlink from clearing the
    # preserved visual mask when a metric observation is missing.
    temporary = destination.with_name(destination.name + ".tmp")
    Image.fromarray(mask).save(temporary, format="PNG")
    os.replace(temporary, destination)
