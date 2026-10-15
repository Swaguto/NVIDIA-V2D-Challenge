from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Requires the native hand runtime")
pytest.importorskip("scipy", reason="Requires the native hand runtime")

from issue6.image_fit import initialize_trailing_pose, refine_sequence  # noqa: E402


class RigidHand(torch.nn.Module):
    def __init__(self):
        super().__init__()
        points = torch.zeros((21, 3))
        points[:, 0] = torch.linspace(-0.03, 0.03, 21)
        points[:, 1] = 0.02 * torch.sin(torch.arange(21).float())
        self.register_buffer("points", points)

    def forward(self, pose, betas):
        points = self.points[None].expand(len(pose), -1, -1)
        return SimpleNamespace(verts=points, joints=points)


def test_source_landmarks_correct_a_biased_track_without_shape_changes():
    mano = RigidHand()
    n = 5
    record = {
        "mano": {"global_orient": [0] * 3, "hand_pose": [0] * 45, "betas": [0] * 10},
        "cam_t": [0, 0, 0.5],
        "hand_scale": 1.0,
    }
    target = mano.points.numpy() + [0.02, 0, 0.5]
    camera = {"fx": 500, "fy": 500, "cx": 320, "cy": 200}
    kp = np.ones((n, 21, 3))
    kp[:, :, 0] = 500 * target[:, 0] / target[:, 2] + 320
    kp[:, :, 1] = 500 * target[:, 1] / target[:, 2] + 200
    _, translation, receipt = refine_sequence(
        mano,
        [record] * n,
        kp,
        np.full((n, 21), 0.5),
        np.ones((n, 21)),
        np.tile(np.eye(3), (n, 1, 1)),
        np.zeros((n, 3)),
        camera,
        True,
        iterations=100,
    )
    assert receipt["median_keypoint_error_px_after"] < 1
    np.testing.assert_allclose(translation[:, 0], 0.02, atol=0.001)


def test_inferred_track_uses_measured_depth_to_avoid_a_near_camera_giant_hand():
    mano = RigidHand()
    n = 5
    record = {
        "mano": {"global_orient": [0] * 3, "hand_pose": [0] * 45, "betas": [0] * 10},
        "cam_t": [0, 0, 0.1],
        "hand_scale": 1.0,
        "flow_supported": True,
    }
    target = mano.points.numpy() + [0.02, 0, 0.5]
    camera = {"fx": 500, "fy": 500, "cx": 320, "cy": 200}
    kp = np.ones((n, 21, 3))
    kp[:, :, 0] = 500 * target[:, 0] / target[:, 2] + 320
    kp[:, :, 1] = 500 * target[:, 1] / target[:, 2] + 200
    _, translation, receipt = refine_sequence(
        mano,
        [record] * n,
        kp,
        np.full((n, 21), 0.5),
        np.ones((n, 21)),
        np.tile(np.eye(3), (n, 1, 1)),
        np.zeros((n, 3)),
        camera,
        True,
        iterations=30,
    )
    assert receipt["inferred_frames_initialized_from_stereo"] == list(range(n))
    np.testing.assert_allclose(translation[:, 2], 0.5, atol=0.005)


def test_trailing_pose_prior_follows_camera_rotation_without_claiming_observation():
    from scipy.spatial.transform import Rotation

    poses = np.ones((3, 48))
    poses[0] = 0
    rotations = Rotation.from_euler("z", np.array([0, 30, 60])[:, None], degrees=True).as_matrix()
    records = [
        {"directly_observed": True},
        {"directly_observed": False, "flow_depth_supported": True},
        {"directly_observed": False, "flow_depth_supported": False},
    ]
    fitted, ids = initialize_trailing_pose(poses, records, rotations, True)
    assert ids == [1]
    np.testing.assert_allclose(Rotation.from_rotvec(fitted[1, :3]).as_matrix(), rotations[1], atol=1e-7)
    np.testing.assert_array_equal(fitted[1, 3:], 0)
    np.testing.assert_array_equal(fitted[2], poses[2])
    assert records[1]["directly_observed"] is False
