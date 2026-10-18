import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from sam3d_candidate import bake_metric_scale


def test_nonuniform_scale_is_baked_once_without_changing_pose_gauge():
    source = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    rotation = Rotation.from_rotvec([0.1, 0.3, -0.2])
    q = rotation.as_quat(scalar_first=True)
    transform = dict(scale=[0.1, 0.2, 0.3], rotation=q, translation=[0.0, 0.0, 0.6])
    mesh, pose = bake_metric_scale(source, transform)
    expected = (
        rotation.apply(source.vertices * np.array(transform["scale"]))
        + transform["translation"]
    )
    actual = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(source.extents, [1, 1, 1])
    np.testing.assert_allclose(mesh.extents, [0.1, 0.2, 0.3])
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scale", [-1, 0.1, 0.1]),
        ("rotation", [2, 0, 0, 0]),
        ("translation", [0, np.nan, 0]),
    ],
)
def test_invalid_model_transforms_are_not_used_as_metric_poses(field, value):
    transform = dict(
        scale=[0.1, 0.1, 0.1], rotation=[1, 0, 0, 0], translation=[0, 0, 0.5]
    )
    transform[field] = value
    with pytest.raises(ValueError):
        bake_metric_scale(trimesh.creation.box(), transform)
