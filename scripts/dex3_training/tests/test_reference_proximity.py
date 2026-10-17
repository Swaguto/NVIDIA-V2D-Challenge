import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_reference_proximity import audit


def bundle(tmp_path, world_rotation=None, translation=None, invalid=False, scale=1.0):
    rotation = np.eye(3) if world_rotation is None else world_rotation
    offset = np.zeros(3) if translation is None else translation
    vertices = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=float)
    hand = np.array([[0.001, 0, 0], [0.1, 0.002, 0]], dtype=float) * scale
    hands = np.broadcast_to(hand @ rotation.T + offset, (2, 2, 2, 3)).copy()
    valid = np.ones((2, 2), bool)
    if invalid:
        valid[1, 0] = False
        hands[1, 0] = np.nan
    poses = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    poses[:, :3, :3] = rotation
    poses[:, :3, 3] = offset
    path = tmp_path / "episode.npz"
    np.savez(path, hand_vertices_world=hands, hand_valid=valid,
             hand_directly_observed=valid, frame_indices=[10, 11],
             object_0_vertices=vertices, object_0_world=poses,
             object_0_scale=np.full(2, scale), contact_known=np.zeros((2, 2, 5), bool))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(units="metres", hand_sides=["left", "right"])))
    return path


def test_world_rigid_transform_does_not_change_proximity(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first = audit(bundle(a))
    second = audit(bundle(b, Rotation.from_rotvec([0.3, -0.6, 1.2]).as_matrix(), np.array([3, -1, 0.7])))
    for x, y in zip(first["frames"], second["frames"], strict=True):
        for side in ["left", "right"]:
            left = x["hands"][side]["proximity"][0]
            right = y["hands"][side]["proximity"][0]
            assert left["vertices_within_screen_radius"] == right["vertices_within_screen_radius"] == 2
            assert left["minimum_mesh_vertex_distance_mm"] == pytest.approx(right["minimum_mesh_vertex_distance_mm"], abs=1e-10)


def test_object_scale_preserves_metric_distance_units(tmp_path):
    result = audit(bundle(tmp_path, scale=2.0))
    assert result["frames"][0]["hands"]["left"]["proximity"][0]["minimum_mesh_vertex_distance_mm"] == pytest.approx(2)


def test_invalid_geometry_remains_unknown_and_source_is_unchanged(tmp_path):
    path = bundle(tmp_path, invalid=True)
    before = path.read_bytes()
    result = audit(path)
    assert result["frames"][1]["hands"]["left"]["proximity"] == []
    assert result["summary"]["left"]["valid_geometry_frames"] == 1
    assert result["summary"]["right"]["valid_geometry_frames"] == 2
    assert result["contact_labels_generated"] is False
    assert result["source_measured_contact_entries"] == 0
    assert path.read_bytes() == before


@pytest.mark.parametrize("radius", [0, -1, 0.02, float("nan")])
def test_screen_radius_must_be_explicit_and_bounded(tmp_path, radius):
    with pytest.raises(ValueError):
        audit(bundle(tmp_path), radius)
