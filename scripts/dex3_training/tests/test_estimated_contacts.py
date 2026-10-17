import json
from pathlib import Path
import sys

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("rtree")
sys.path.insert(0, str(Path(__file__).parents[1]))
from prepare_estimated_contacts import near_surface, prepare


def source(tmp_path, invert=False, rotate=False):
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    if invert:
        mesh.invert()
    hand = np.array([[0.101, 0, 0.01], [0.099, 0, 0.01], [0.095, 0, 0.01], [0.2, 0, 0.01]])
    pose = np.eye(4)
    if rotate:
        pose[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        pose[:3, 3] = [3, -2, 0.7]
    hands = np.broadcast_to(hand @ pose[:3, :3].T + pose[:3, 3], (2, 1, 4, 3)).copy()
    hands[1] = np.nan
    path = tmp_path / "episode.npz"
    np.savez(path, hand_vertices_world=hands, hand_valid=[[True], [False]],
             hand_directly_observed=[[True], [False]], frame_indices=[0, 1], timestamps_s=[0, 0.05],
             object_0_vertices=mesh.vertices, object_0_faces=mesh.faces,
             object_0_world=np.broadcast_to(pose, (2, 4, 4)), object_0_scale=[1, 1],
             contact_known=np.zeros((2, 1, 5), bool))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(units="metres", hand_sides=["left"], objects=[dict(name="box")])))
    return path


def test_bounded_triangle_query_matches_independent_brute_force_near_surface():
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    points = np.random.default_rng(42).uniform(-0.15, 0.15, (128, 3))
    expected, distance, _ = trimesh.proximity.closest_point_naive(mesh, points)
    actual, measured, _ = near_surface(mesh, points, 0.01)
    near = distance <= 0.01
    assert near.any()
    np.testing.assert_allclose(measured[near], distance[near], atol=1e-12)
    np.testing.assert_allclose(actual[near], expected[near], atol=1e-12)
    assert not np.any((measured <= 0.01) & ~near)


def test_estimated_normals_push_into_object_and_deep_or_unknown_points_stay_inactive(tmp_path):
    path = source(tmp_path)
    original = path.read_bytes()
    receipt = prepare(path, tmp_path / "out")
    with np.load(tmp_path / "out/estimated-contact-candidates.npz") as data:
        assert data["part_ids"][0, 0].tolist() == [1, 1, 0, 0]
        np.testing.assert_allclose(data["normal_force_directions_object"][0, 0, :2], [[-1, 0, 0]] * 2)
        np.testing.assert_allclose(data["points_object_m"][0, 0, :2, 0], 0.1)
        assert not data["part_ids"][1].any()
        assert np.isfinite(data["points_object_m"]).all()
    assert path.read_bytes() == original
    assert receipt["measured_contact_labels_created"] is False
    assert receipt["training_guidance_activated"] is False


def test_world_frame_changes_leave_object_local_candidates_invariant(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    prepare(source(a), a / "out")
    prepare(source(b, rotate=True), b / "out")
    with np.load(a / "out/estimated-contact-candidates.npz") as first, np.load(b / "out/estimated-contact-candidates.npz") as second:
        for key in ("points_object_m", "normal_force_directions_object", "part_ids"):
            np.testing.assert_allclose(first[key], second[key], atol=1e-7)


def test_inward_winding_cannot_silently_flip_contact_force_direction(tmp_path):
    with pytest.raises(ValueError, match="orient"):
        prepare(source(tmp_path, invert=True), tmp_path / "out")
