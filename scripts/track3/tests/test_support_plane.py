import sys
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from infer_support_plane import estimate, sha, verify_bundle


def test_plane_uses_repeated_background_and_points_toward_the_observer():
    rng = np.random.default_rng(9)
    xy = np.array(
        [
            [x, y]
            for x in np.arange(-0.25, 0.26, 0.01)
            for y in np.arange(-0.25, 0.26, 0.01)
        ]
    )
    one = np.c_[xy, np.ones(len(xy))]
    points = np.concatenate([one + rng.normal(0, 0.0005, one.shape) for _ in range(3)])
    rows = np.repeat([0, 10, 20], len(one))
    noise = rng.uniform(-2, 2, (3000, 3))
    points = np.r_[points, noise]
    rows = np.r_[rows, np.full(3000, 30)]
    report, data = estimate(points, rows, np.zeros((4, 3)))
    np.testing.assert_allclose(report["plane_world"], [0, 0, -1, 1], atol=0.001)
    assert report["persistent_1cm_cells"] > 100 and report["source_frames"] >= 3
    assert abs(report["footprint_area_m2"] - 0.25) < 0.01
    assert data["faces"].shape[1] == 3


def test_single_frame_or_collinear_points_do_not_create_a_support_plane():
    points = np.zeros((1200, 3))
    points[:, 0] = np.arange(1200) * 0.001
    with pytest.raises(ValueError):
        estimate(points, np.zeros(1200), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        estimate(points, np.arange(1200) % 3, np.zeros((3, 3)))


def test_support_bundle_rejects_a_different_world_gauge(tmp_path):
    camera = tmp_path / "camera.json"
    camera.write_text(json.dumps({"translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}))
    m = {
        "episode": 1,
        "input_contract": {"calibration_sha256": "calib"},
        "frames": [{"id": 0, "camera_pose_path": str(camera)}],
        "color_camera": dict(fx=100, fy=100, cx=50, cy=50, width=100, height=100),
    }
    payload = tmp_path / "episode.npz"
    values = dict(
        frame_indices=[0],
        intrinsics=[[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        image_size=[100, 100],
        camera_c2w=np.eye(4)[None],
    )
    bundle = dict(
        schema="v2d.track3.episode.v1",
        episode=1,
        frames=[0],
        input_contract=m["input_contract"],
        payload={"path": payload.name},
    )
    manifest = tmp_path / "manifest.json"
    for shifted in (False, True):
        values["camera_c2w"][0, 0, 3] = float(shifted)
        np.savez(payload, **values)
        bundle["payload"]["sha256"] = sha(payload)
        manifest.write_text(json.dumps(bundle))
        if shifted:
            with pytest.raises(AssertionError):
                verify_bundle(m, manifest)
        else:
            verify_bundle(m, manifest)
