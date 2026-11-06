import os

import numpy as np
from PIL import Image

from issue6.hand_masks import export_metric_mask


def test_missing_metric_observation_cannot_erase_visual_mask(tmp_path):
    visual, metric = tmp_path / "visual.png", tmp_path / "metric.png"
    pixels = np.full((16, 20), 255, np.uint8)
    Image.fromarray(pixels).save(visual)
    os.link(visual, metric)
    before = visual.read_bytes()
    export_metric_mask(visual, metric, False, (20, 16))
    assert visual.read_bytes() == before
    assert not np.asarray(Image.open(metric)).any()
    export_metric_mask(visual, metric, True, (20, 16))
    assert np.array_equal(np.asarray(Image.open(metric)), pixels)
