"""Colour seeding must use calibrated depth, swapped stereo frames, and valid rays."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from rgb_stereo import rectified_to_rgb_map


def fixture():
    K = np.array([[100.0, 0, 5], [0, 100.0, 5], [0, 0, 1]])
    rig = SimpleNamespace(left="ego_cam_c", R1=np.eye(3), P1=np.c_[K, np.zeros(3)])
    rgb = dict(
        intrinsic_matrix=K.tolist(),
        distortion_coefficients=[0] * 5,
        resolution=[10, 10],
        extrinsics_to_stereo_left=np.eye(4).tolist(),
    )
    cal = dict(
        cameras={
            "ego_cam_a": rgb,
            "ego_cam_c": {"extrinsics_to_stereo_left": np.eye(4).tolist()},
        }
    )
    return rig, cal


def test_identity_mapping_uses_pixel_axes_and_rejects_unknown_depth():
    rig, cal = fixture()
    depth = np.ones((10, 10))
    valid = np.ones_like(depth, bool)
    valid[2, 3] = False
    mapping = rectified_to_rgb_map(cal, rig, depth, valid)
    np.testing.assert_array_equal(mapping[4, 6], [6, 4])
    np.testing.assert_array_equal(mapping[2, 3], [-1, -1])
    np.testing.assert_array_equal(
        rectified_to_rgb_map(cal, rig, depth, ~np.ones_like(valid)), -1
    )


def test_common_reference_chain_accounts_for_published_stereo_swap_and_depth():
    rig, cal = fixture()
    left = np.eye(4)
    left[0, 3] = -0.075
    rgb = np.eye(4)
    rgb[0, 3] = -0.0375
    cal["cameras"]["ego_cam_c"]["extrinsics_to_stereo_left"] = left.tolist()
    cal["cameras"]["ego_cam_a"]["extrinsics_to_stereo_left"] = rgb.tolist()
    depth = np.full((10, 10), 1.0)
    first = rectified_to_rgb_map(cal, rig, depth, np.ones_like(depth, bool))
    second = rectified_to_rgb_map(cal, rig, depth * 2, np.ones_like(depth, bool))
    np.testing.assert_allclose(first[5, 6], [2.25, 5])
    np.testing.assert_allclose(second[5, 6], [4.125, 5])
