import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from camera_prior import load_sequence, transport


def pose(angles, translation):
    p = np.eye(4)
    p[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    p[:3, 3] = translation
    return p


def test_camera_motion_changes_image_pose_while_preserving_world_body_pose():
    old = pose([0.1, 0.2, 0.3], [0.1, -0.2, 0.3])
    new = pose([-0.2, 0.1, 0.6], [-0.3, 0.2, 0.7])
    body = pose([0.2, -0.1, 0.4], [0.1, 0.3, 0.6])
    result = transport(body, old, new)
    np.testing.assert_allclose(
        np.linalg.inv(new) @ result, np.linalg.inv(old) @ body, atol=1e-12
    )
    gauge = pose([0.5, -0.2, 0.1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        transport(body, old @ gauge, new @ gauge), result, atol=1e-12
    )


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"gt_use": "gt_assisted"},
        {"coordinate_frame": "original_camera"},
        {"status": "inferred"},
    ],
)
def test_camera_provenance_and_coordinate_contract_is_verified(tmp_path, change):
    record = dict(
        episode=1,
        frame_index=0,
        camera_frame="ego_cam_c",
        coordinate_frame="rectified_camera",
        gt_use="independent",
        status="observed",
        rotation=[1, 0, 0, 0],
        translation=[0, 0, 0],
    )
    record.update(change)
    (tmp_path / "000000.json").write_text(json.dumps(record))
    if change:
        with pytest.raises(ValueError, match="independent"):
            load_sequence(tmp_path, 1, 1)
    else:
        matrices, digest = load_sequence(tmp_path, 1, 1)
        np.testing.assert_array_equal(matrices[0], np.eye(4))
        assert len(digest) == 64
