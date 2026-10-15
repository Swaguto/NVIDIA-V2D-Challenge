import json

import numpy as np
import pytest
from PIL import Image

from issue6.image_observations import load_image_estimate


def fixture(tmp_path):
    side = tmp_path / "left"
    side.mkdir()
    Image.fromarray(np.full((32, 32), 255, np.uint8)).save(side / "mask.png")
    data = dict(
        estimated=True,
        original_direct_observation=False,
        keypoints=[[16, 16, 0.5]] * 21,
        mask="mask.png",
        source="fresh_hamer",
    )
    path = side / "000004.json"
    path.write_text(json.dumps(data))
    return path, data


def test_estimates_preserve_observation_authority(tmp_path):
    path, data = fixture(tmp_path)
    result = load_image_estimate(tmp_path, "left", 4, dict(width=32, height=32), False)
    assert result["estimated"] is True
    assert "metric_valid" not in result
    for change in [{"metric_valid": True}, {"directly_observed": True}, {"original_direct_observation": True}]:
        path.write_text(json.dumps(dict(data, **change)))
        with pytest.raises(ValueError):
            load_image_estimate(tmp_path, "left", 4, dict(width=32, height=32), False)


def test_reject_unassociated_landmarks_and_wrong_grid(tmp_path):
    path, data = fixture(tmp_path)
    with pytest.raises(ValueError):
        load_image_estimate(tmp_path, "left", 4, dict(width=64, height=32), False)
    data["keypoints"][0] = [50, 16, 0.5]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_image_estimate(tmp_path, "left", 4, dict(width=32, height=32), False)
