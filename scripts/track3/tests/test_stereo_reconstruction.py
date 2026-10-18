import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from reconstruct_stereo_object import fusion_depth, points_from_depth, seed_from_points
from reconstructed_view import prepare, verify_geometry


def cloud():
    rng = np.random.default_rng(42)
    points = rng.uniform(-0.05, 0.05, (3000, 3))
    axes = rng.integers(0, 3, len(points))
    points[np.arange(len(points)), axes] = rng.choice([-0.05, 0.05], len(points))
    return points


def test_metric_seed_keeps_one_body_frame_under_translation():
    points = cloud()
    mesh, pose = seed_from_points(points)
    shifted, second = seed_from_points(points + [1.0, 2.0, 3.0])
    np.testing.assert_allclose(mesh.extents, [0.1, 0.1, 0.1], atol=0.003)
    np.testing.assert_allclose(shifted.extents, mesh.extents, atol=0.003)
    np.testing.assert_allclose(second[:3, 3] - pose[:3, 3], [1, 2, 3], atol=0.003)
    assert mesh.is_watertight


def test_thin_handle_is_not_discarded_beside_dense_body():
    body = cloud() * 0.3
    handle = np.array(
        [
            [x, y, z]
            for x in np.arange(0.01, 0.2, 0.002)
            for y in [-0.002, 0, 0.002]
            for z in [-0.002, 0.002]
        ]
    )
    mesh, _ = seed_from_points(np.r_[body, handle])
    assert mesh.extents[0] > 0.20


@pytest.mark.parametrize(
    "bad", [np.ones((50, 3)), np.full((50, 3), np.nan), np.ones((5, 3))]
)
def test_seed_never_fabricates_volume_from_invalid_or_insufficient_data(bad):
    with pytest.raises(ValueError):
        seed_from_points(bad)


def test_depth_mask_rejects_unknown_pixels_and_retains_metric_scale():
    depth = np.full((20, 20), 0.5)
    valid = np.ones((20, 20), bool)
    mask = np.full((20, 20), 255, np.uint8)
    valid[:2] = False
    depth[-1] = np.nan
    points = points_from_depth(
        depth, valid, mask, np.array([[100, 0, 10], [0, 100, 10], [0, 0, 1]])
    )
    assert len(points) == 340
    assert np.allclose(points[:, 2], 0.5)


def test_fusion_excludes_far_stereo_outliers_without_moving_measured_depth():
    depth = np.full((20, 20), 0.5, dtype=np.float32)
    depth[5:7, 5:7] = 2.0
    valid = np.ones_like(depth, bool)
    mask = np.full(depth.shape, 255, np.uint8)
    pose = np.eye(4)
    pose[2, 3] = 0.5
    K = np.array([[100, 0, 10], [0, 100, 10], [0, 0, 1]])
    filtered, support = fusion_depth(
        depth, valid, mask, K, pose, [[-0.1] * 3, [0.1] * 3]
    )
    assert support["accepted"] and support["inside_seed_envelope_pixels"] == 396
    np.testing.assert_array_equal(filtered[5:7, 5:7], 0)
    assert np.all(filtered[filtered > 0] == 0.5)
    pose[0, 3] = 2.0
    rejected, support = fusion_depth(
        depth, valid, mask, K, pose, [[-0.1] * 3, [0.1] * 3]
    )
    assert not support["accepted"] and not rejected.any()


def fixture(tmp_path):
    data = tmp_path / "data"
    (data / "videos").mkdir(parents=True)
    (data / "meta").mkdir()
    folder = tmp_path / "geometry" / "pot"
    folder.mkdir(parents=True)
    payload = b"actual mesh bytes are checked independently"
    (folder / "mesh.glb").write_bytes(payload)
    report = {
        "schema": "v2d.video-object-reconstruction.v1",
        "stage": "depth_seed",
        "episode": 1,
        "object": "pot",
        "provided_meshes_read": False,
        "reference_pose_labels_read": False,
        "units": "metres",
        "body_origin_frozen": True,
        "mesh_sha256": hashlib.sha256(payload).hexdigest(),
    }
    (folder / "manifest.json").write_text(json.dumps(report))
    return data, folder.parent


def test_released_cad_view_cannot_enter_scored_pose_stage(tmp_path):
    with pytest.raises(ValueError, match="video-reconstructed"):
        verify_geometry(tmp_path, 1, ["pot"])


def test_seed_needs_explicit_bootstrap_mode_and_binds_payload(tmp_path):
    data, geometry = fixture(tmp_path)
    with pytest.raises(ValueError, match="allow-seed"):
        prepare(data, geometry, 1, ["pot"], tmp_path / "denied")
    out = tmp_path / "view"
    prepare(data, geometry, 1, ["pot"], out, allow_seed=True)
    assert not verify_geometry(out, 1, ["pot"])["provided_mesh_fallback"]
    (geometry / "pot/mesh.glb").write_bytes(b"changed geometry")
    with pytest.raises(ValueError, match="bound"):
        verify_geometry(out, 1, ["pot"])


def test_geometry_view_rejects_wrong_episode_and_path_names(tmp_path):
    data, geometry = fixture(tmp_path)
    with pytest.raises(ValueError):
        prepare(data, geometry, 2, ["pot"], tmp_path / "bad")
    with pytest.raises(ValueError):
        prepare(data, geometry, 1, ["../outside"], tmp_path / "bad")


def test_sam3d_view_binds_metric_image_sources_without_calling_them_observed_geometry(
    tmp_path,
):
    data, geometry = fixture(tmp_path)
    folder = geometry / "pot"
    p = folder / "manifest.json"
    m = json.loads(p.read_text())
    m.update(
        stage="sam3d_metric_depth",
        model_revision="2e73555018d2741ccd486e56c24fac41155a1dc6",
        unseen_surfaces_are_inferred=True,
        anchor_pose_is_inferred=True,
        measured_pointmap_pixels=1000,
        coordinate_frame="rectified_camera",
        source_files_sha256={},
    )
    for name in [
        "input.png",
        "mask.png",
        "pointmap.npy",
        "input-intrinsics.json",
        "raw.glb",
        "raw-transform.json",
    ]:
        (folder / name).write_bytes(("verified model source " + name).encode())
        m["source_files_sha256"][name] = hashlib.sha256(
            (folder / name).read_bytes()
        ).hexdigest()
    p.write_text(json.dumps(m))
    out = tmp_path / "sam-view"
    prepare(data, geometry, 1, ["pot"], out)
    assert (
        verify_geometry(out, 1, ["pot"])["objects"]["pot"]["stage"]
        == "sam3d_metric_depth"
    )
    (folder / "pointmap.npy").write_bytes(b"changed measured depth")
    with pytest.raises(ValueError, match="source reconstruction input changed"):
        verify_geometry(out, 1, ["pot"])
