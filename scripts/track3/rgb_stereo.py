"""Project the rig's colour camera into rectified stereo using measured depth."""

import json

import cv2
import numpy as np


def rectified_to_rgb_map(calibration, rig, depth, valid):
    cameras = calibration["cameras"]
    colour = cameras["ego_cam_a"]
    left = cameras[rig.left]
    # These factory extrinsics share the original stereo reference. It need not
    # be rig.left after the published 180-degree left/right swap.
    reference_from_rgb = np.asarray(colour["extrinsics_to_stereo_left"], float)
    reference_from_left = np.asarray(left["extrinsics_to_stereo_left"], float)
    rgb_from_left = np.linalg.inv(reference_from_rgb) @ reference_from_left
    depth = np.asarray(depth)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0)
    mapping = np.full((*depth.shape, 2), -1, np.float32)
    if not valid.any():
        return mapping
    y, x = np.nonzero(valid)
    z = depth[y, x]
    rectified = np.c_[
        (x - rig.P1[0, 2]) * z / rig.P1[0, 0], (y - rig.P1[1, 2]) * z / rig.P1[1, 1], z
    ]
    left_points = np.einsum("ni,ij->nj", rectified, rig.R1)
    rgb_points = (
        np.einsum("ni,ji->nj", left_points, rgb_from_left[:3, :3])
        + rgb_from_left[:3, 3]
    )
    projected, _ = cv2.projectPoints(
        rgb_points,
        np.zeros(3),
        np.zeros(3),
        np.asarray(colour["intrinsic_matrix"], float),
        np.asarray(colour["distortion_coefficients"], float),
    )
    uv = projected.reshape(-1, 2)
    width, height = colour["resolution"]
    good = np.isfinite(uv).all(axis=1) & (rgb_points[:, 2] > 0)
    good &= (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height - 1)
    )
    mapping[y[good], x[good]] = uv[good]
    return mapping


def seed_frames(data, episode, rig, indices):
    """Colour assists semantic seeding only; stereo geometry is unchanged."""
    from scipy.ndimage import distance_transform_edt
    from stereo_depth import stereo_depth

    calibration = json.loads((data / "meta/camera_calibration.json").read_text())
    base = data / "videos/chunk-000"
    names = [rig.left, rig.right, "ego_cam_a"]
    captures = [
        cv2.VideoCapture(
            str(base / f"observation.images.{name}" / f"episode_{episode:06d}.mp4")
        )
        for name in names
    ]
    if not all(c.isOpened() for c in captures):
        raise FileNotFoundError("Semantic colour seeding requires all three ego videos")
    if len(
        {int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in captures}
    ) != 1 or not np.allclose(
        [c.get(cv2.CAP_PROP_FPS) for c in captures], captures[0].get(cv2.CAP_PROP_FPS)
    ):
        raise ValueError("Colour/stereo video axes differ")
    requested = set(indices)
    output = {}
    try:
        for index in range(max(requested) + 1):
            samples = [c.read() for c in captures]
            if not all(ok for ok, _ in samples):
                raise ValueError("Colour/stereo video ended early")
            if index not in requested:
                continue
            if not np.allclose(
                [c.get(cv2.CAP_PROP_POS_MSEC) for c in captures],
                captures[0].get(cv2.CAP_PROP_POS_MSEC),
                atol=1e-2,
            ):
                raise ValueError("Colour/stereo presentation timestamps differ")
            left, right, rgb = [image for _, image in samples]
            grey, _ = rig.rectify_pair(left, right)
            depth, valid = stereo_depth(rig, left, right)
            # Fill only nearby holes for appearance. This never changes metric
            # depth/validity files or creates additional geometric observations.
            distance, nearest = distance_transform_edt(~valid, return_indices=True)
            appearance_depth = depth[tuple(nearest)]
            mapping = rectified_to_rgb_map(
                calibration, rig, appearance_depth, distance <= 16
            )
            colour = cv2.remap(
                rgb, mapping[:, :, 0], mapping[:, :, 1], cv2.INTER_LINEAR
            )
            available = mapping[:, :, 0] >= 0
            grey[available] = colour[available]
            output[index] = cv2.cvtColor(grey, cv2.COLOR_BGR2RGB)
    finally:
        for capture in captures:
            capture.release()
    return output
