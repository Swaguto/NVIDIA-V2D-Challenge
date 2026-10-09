import importlib.util
from pathlib import Path

import numpy as np
import pytest

from issue6.manifest import read_json

path = Path(__file__).resolve().parents[2] / "v2d_hand_alignment/lib/dynhamr_to_hamer_tracks.py"
spec = importlib.util.spec_from_file_location("dynhamr_conversion_test", path)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.7, np.pi - 1e-3, np.pi - 1e-7, np.pi])
def test_rotation_conversion_preserves_float32_half_turns(angle):
    axis = np.array([1.0, -2.0, 3.0])
    axis /= np.linalg.norm(axis)
    expected = converter._axis_angle_to_rotmat(axis * angle)
    recovered = converter._rotmat_to_axis_angle(expected.astype(np.float32))
    np.testing.assert_allclose(converter._axis_angle_to_rotmat(recovered), expected, atol=1e-7)


@pytest.mark.parametrize("right", [False, True])
def test_conversion_preserves_vertices_about_nonzero_mano_wrist(tmp_path, right):
    camera_rotation = converter._axis_angle_to_rotmat(np.array([0.3, -0.6, 0.2]))
    root_rotation = converter._axis_angle_to_rotmat(np.array([0.5, 0.1, -0.4]))
    pivot = np.array([0.03, -0.02, 0.01])
    translation = np.array([0.2, 0.1, 0.8])
    camera_translation = np.array([0.01, 0.02, -0.03])
    points = np.array([[0.08, 0.03, 0.01], [-0.02, 0.04, -0.01], [0.01, -0.08, 0.02]])
    source = tmp_path / "fit.npz"
    np.savez(
        source,
        trans=translation[None, None],
        root_orient=np.array([[[0.5, 0.1, -0.4]]]),
        pose_body=np.zeros((1, 1, 15, 3)),
        betas=np.zeros((1, 10)),
        cam_R=camera_rotation[None, None],
        cam_t=camera_translation[None, None],
        is_right=np.array([[right]]),
        intrins=np.array([500, 480, 312, 235]),
        mano_root_offsets=pivot[None],
    )
    out = tmp_path / "converted"
    converter.convert_dynhamr_to_hamer_tracks(str(source), str(out), image_size=(640, 480))
    record = read_json(out / str(3 if right else 2) / "000000.json")
    assert record["camera"]["principal_point"] == [312, 235]
    assert record["camera"]["scaled_focal_length_y"] == 480
    mirror = np.eye(3) if right else np.diag([-1, 1, 1])
    native_world = ((points - pivot) @ root_rotation.T + pivot + translation) @ mirror.T
    native_camera = native_world @ camera_rotation.T + camera_translation
    converted_rotation = converter._axis_angle_to_rotmat(record["mano"]["global_orient"])
    converted_camera = ((points - pivot) @ converted_rotation.T + pivot) @ mirror.T
    converted_camera += record["camera"]["pred_cam_t_full"]
    np.testing.assert_allclose(converted_camera, native_camera, atol=1e-12)
    naive_translation = camera_rotation @ mirror @ translation + camera_translation
    assert np.linalg.norm(naive_translation - record["camera"]["pred_cam_t_full"]) > 0.005


def test_moving_camera_conversion_requires_shape_pivot(tmp_path):
    source = tmp_path / "fit.npz"
    np.savez(
        source,
        trans=np.zeros((1, 1, 3)),
        root_orient=np.zeros((1, 1, 3)),
        pose_body=np.zeros((1, 1, 15, 3)),
        betas=np.zeros((1, 10)),
        cam_R=converter._axis_angle_to_rotmat(np.array([0.1, 0.2, 0.3]))[None, None],
        cam_t=np.zeros((1, 1, 3)),
        is_right=np.ones((1, 1)),
        intrins=[500, 500, 320, 240],
    )
    with pytest.raises(ValueError, match="MANO root offsets"):
        converter.convert_dynhamr_to_hamer_tracks(str(source), str(tmp_path / "converted"))
    assert not (tmp_path / "converted").exists()
