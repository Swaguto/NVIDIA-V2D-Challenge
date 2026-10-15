import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="Requires the image tracking runtime")

from issue6.image_tracks import propagate_landmarks  # noqa: E402


def test_missing_detections_follow_source_pixels_not_a_held_pose(tmp_path):
    image = np.zeros((80, 80), np.uint8)
    image[10:55, 10:55] = np.random.default_rng(7).integers(20, 240, (45, 45), dtype=np.uint8)
    paths = []
    for i in range(3):
        shifted = cv2.warpAffine(image, np.float32([[1, 0, i * 2], [0, 1, i * 3]]), (80, 80))
        p = tmp_path / f"{i}.png"
        cv2.imwrite(str(p), shifted)
        paths.append(p)
    mask = tmp_path / "mask.png"
    cv2.imwrite(str(mask), (image > 0).astype(np.uint8) * 255)
    kp = np.array([[x, y, 1] for y in (20, 30, 40) for x in (15, 20, 25, 30, 35, 40, 45)], float)
    result = propagate_landmarks(paths, [mask, None, None], [kp, None, None], [0, 0.05, 0.1])
    assert np.count_nonzero(result[2, :, 2]) >= 15
    valid = result[2, :, 2] > 0
    np.testing.assert_allclose(result[2, valid, :2], kp[valid, :2] + [4, 6], atol=0.5)


def test_an_unobserved_track_is_never_initialized_from_background(tmp_path):
    p = tmp_path / "image.png"
    cv2.imwrite(str(p), np.zeros((80, 80), np.uint8))
    result = propagate_landmarks([p, p], [None, None], [None, None], [0, 0.05])
    assert not result.any()
