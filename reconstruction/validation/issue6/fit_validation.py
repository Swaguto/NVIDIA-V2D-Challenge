"""Check fit evidence even when the worker enables optimized Python."""

import numpy as np


def verify_fit(initial, final, fixed_shape):
    with (
        np.load(initial, allow_pickle=False) as initial_data,
        np.load(final, allow_pickle=False) as final_data,
    ):
        for key in ("trans", "root_orient", "pose_body", "betas", "cam_R", "cam_t"):
            if not np.isfinite(final_data[key]).all():
                raise ValueError(f"Nonfinite final fit: {key}")
        if fixed_shape and not np.array_equal(
            initial_data["betas"], final_data["betas"]
        ):
            raise ValueError("Fixed shape changed during fit")


def verify_source_cameras(source, final, frame_indices):
    """Reject loader augmentation or camera optimization for calibrated inputs."""
    ids = np.asarray(frame_indices)
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or not len(ids)
        or ids[0] < 0
        or not (np.diff(ids) == 1).all()
    ):
        raise ValueError(
            "Source camera frame indices must be contiguous original integers"
        )
    with (
        np.load(source, allow_pickle=False) as original,
        np.load(final, allow_pickle=False) as fitted,
    ):
        expected = original["w2c"][ids]
        rotations, translations = fitted["cam_R"], fitted["cam_t"]
        if (
            expected.shape != (len(ids), 4, 4)
            or not len(rotations)
            or rotations.ndim != 4
            or rotations.shape[1:] != (len(ids), 3, 3)
            or translations.shape != rotations.shape[:2] + (3,)
        ):
            raise ValueError("Fitted camera and source frame axes differ")
        if not all(np.isfinite(x).all() for x in (expected, rotations, translations)):
            raise ValueError("Nonfinite source or fitted cameras")
        for rotation, translation in zip(rotations, translations, strict=True):
            np.testing.assert_allclose(
                rotation, expected[:, :3, :3], atol=1e-6, rtol=1e-6
            )
            np.testing.assert_allclose(
                translation, expected[:, :3, 3], atol=1e-6, rtol=1e-6
            )
    return {
        "source_frames_checked": len(ids),
        "hand_camera_tracks_checked": len(rotations),
        "loader_camera_augmentation_disabled": True,
        "camera_optimization_disabled": True,
    }
