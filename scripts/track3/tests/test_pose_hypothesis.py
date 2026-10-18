"""Prefer measured fit over frame count and preserve the raw mesh body origin."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from select_pose_hypothesis import choose, observed_fit


def test_coverage_alone_does_not_select_an_inaccurate_track():
    selected, means = choose(
        {"fewer_better": [0.9, 0.9, 0], "all_poor": [0.4, 0.4, 0.4]}, 3
    )
    assert selected == "fewer_better"
    assert means["fewer_better"] == pytest.approx(0.6)
    with pytest.raises(ValueError, match="every source frame"):
        choose({"truncated": [0.9, 0.9]}, 3)
    with pytest.raises(ValueError, match="No measured evidence"):
        choose({"unknown": [0, 0, 0]}, 3)


def test_exact_ties_preserve_the_control_track_without_framewise_switching():
    assert choose({"control": [0.5, 0.5], "other": [0.5, 0.5]}, 2)[0] == "control"


def test_pixel_depth_fit_uses_raw_vertices_without_recentering_and_keeps_unknowns():
    K = np.array([[100.0, 0, 12], [0, 100.0, 12], [0, 0, 1.0]])
    vertices = np.array(
        [
            [(x - 12) * 0.5 / 100 + 0.3, (y - 12) * 0.5 / 100, 0.5]
            for y in range(6, 17)
            for x in range(6, 17)
        ]
    )
    faces = []
    for y in range(10):
        for x in range(10):
            i = y * 11 + x
            faces.extend([[i, i + 1, i + 11], [i + 1, i + 12, i + 11]])
    depth = np.ones((30, 30)) * 0.5
    valid = np.ones_like(depth, bool)
    mask = np.zeros_like(valid)
    mask[6:17, 6:17] = True
    pose = np.eye(4)
    pose[0, 3] = -0.3
    args = (vertices, np.array(faces), cKDTree(vertices))
    fit = observed_fit(*args, pose, depth, valid, mask, K)
    assert fit["quality"] > 0.99 and fit["silhouette_iou"] > 0.99
    pose[0, 3] += 0.2
    wrong = observed_fit(*args, pose, depth, valid, mask, K)
    assert wrong["quality"] == 0
    unknown = observed_fit(*args, pose, depth, valid & False, mask, K)
    assert unknown["quality"] == 0 and unknown["measured_pixels"] == 0
