"""CPU tests for the Track 3 -> gsplat refinement input adapter."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gsplat_inputs as gi


def random_pose(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    M = np.eye(4)
    M[:3, :3] = gi.quat_wxyz_to_matrix(q)
    M[:3, 3] = rng.normal(size=3)
    return M


def record(M, **extra):
    return {"rotation": gi.matrix_to_quat_wxyz(M[:3, :3]), "translation": M[:3, 3].tolist(), "scale": 1.0, **extra}


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        M = random_pose(rng)
        q = gi.matrix_to_quat_wxyz(M[:3, :3])
        assert q[0] >= 0
        assert np.allclose(gi.quat_wxyz_to_matrix(q), M[:3, :3], atol=1e-9)
    # 180 degree rotations exercise the non-trace branch.
    for R in (np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])):
        assert np.allclose(gi.quat_wxyz_to_matrix(gi.matrix_to_quat_wxyz(R)), R)


def test_depth_encoding_matches_refinement_decoder():
    z = np.array([[0.3, 1.0, 2.5], [np.nan, -1.0, 0.0]])
    valid = np.array([[True, True, False], [True, True, True]])
    px = gi.depth_to_inverse_u16(z, valid)
    assert px.dtype == np.uint16
    assert (px[1] == 0).all() and px[0, 2] == 0
    decoded = 1.0 / (px[0, :2] / 65535.0) - 1.0  # refine_simple._load_depth_resized
    assert np.allclose(decoded, z[0, :2], atol=2e-4)


def test_pose_record_rejects_non_unit_scale():
    with pytest.raises(ValueError):
        gi.pose_record_to_matrix({"rotation": [1, 0, 0, 0], "translation": [0, 0, 0], "scale": [1, 2, 1]})
    M = gi.pose_record_to_matrix({"rotation": [1, 0, 0, 0], "translation": [1, 2, 3], "scale": 1.0})
    assert np.allclose(M[:3, 3], [1, 2, 3])


def build_tree(tmp: Path, episode=2, frames=(0, 1, 2, 3)):
    rng = np.random.default_rng(1)
    data, results, bundle, hands = tmp / "data", tmp / "results", tmp / "bundle", tmp / "run"
    ep = f"e{episode:03d}"
    (data / "mesh" / "white_pot").mkdir(parents=True)
    (data / "mesh" / "white_pot" / "white_pot.glb").write_bytes(b"glb")
    (data / "meta").mkdir()
    (data / "meta" / "info.json").write_text(json.dumps({"fps": 20}))
    cams, objs_world = {}, {}
    manifest_frames = []
    for sub in ("color", "depth", "valid"):
        (bundle / sub).mkdir(parents=True, exist_ok=True)
    for f in frames:
        cams[f] = random_pose(rng)
        objs_world[f] = random_pose(rng)
        cam_dir = results / ep / "camera" / "ego_cam_c"
        cam_dir.mkdir(parents=True, exist_ok=True)
        (cam_dir / f"{f:06d}.json").write_text(json.dumps(record(cams[f], status="observed")))
        inf = results / ep / "inferred" / ep / "ego_object_poses" / ep / "white_pot" / "object_to_world"
        inf.mkdir(parents=True, exist_ok=True)
        (inf / f"{f:06d}.json").write_text(json.dumps(record(objs_world[f], status="inferred",
                                                             inference_method="world_hold")))
        if f % 2 == 0:  # observed on even frames only
            obs = results / ep / "poses" / ep / "ego_object_poses" / ep / "white_pot" / "object_to_cam" / "ego_cam_c"
            obs.mkdir(parents=True, exist_ok=True)
            (obs / f"{f:06d}.json").write_text(json.dumps(record(cams[f] @ objs_world[f], status="observed",
                                                                 gt_use="none")))
            mask = results / ep / "masks" / "white_pot"
            mask.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((4, 6), 255, np.uint8)).save(mask / f"{f:06d}.png")
        Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(bundle / "color" / f"{f:06d}.png")
        np.save(bundle / "depth" / f"{f:06d}.npy", np.full((4, 6), 0.8))
        Image.fromarray(np.full((4, 6), 255, np.uint8)).save(bundle / "valid" / f"{f:06d}.png")
        manifest_frames.append({"id": f, "color": f"color/{f:06d}.png", "depth_path": f"depth/{f:06d}.npy",
                                "depth_validity": f"valid/{f:06d}.png"})
    (bundle / "manifest.json").write_text(json.dumps({
        "episode": episode,
        "color_camera": {"id": "ego_cam_c", "fx": 500, "fy": 500, "cx": 3, "cy": 2, "width": 6, "height": 4},
        "objects": [{"id": "white_pot"}],
        "frames": manifest_frames,
    }))
    # Two right-hand tracks: a short one (frames 0-1) and a long one (1-3).
    for tid, fs, valid in (("3", (0, 1), True), ("7", (1, 2, 3), True)):
        d = hands / "align" / "records" / tid
        d.mkdir(parents=True)
        for f in fs:
            (d / f"{f:06d}.json").write_text(json.dumps({
                "frame_idx": f, "track_id": int(tid), "is_right": True,
                "mano": {"global_orient": [0, 0, 0], "hand_pose": [0] * 45, "betas": [0] * 10},
                "cam_t": [0, 0, 0.5], "diagnostics": {"metric_valid": valid}}))
    (hands / "hands" / "masks" / "right").mkdir(parents=True)
    Image.fromarray(np.full((4, 6), 255, np.uint8)).save(hands / "hands" / "masks" / "right" / "000001.png")
    return data, results, bundle, hands, cams, objs_world


def test_assemble(tmp_path):
    data, results, bundle, hands, cams, objs_world = build_tree(tmp_path)
    out = tmp_path / "out"
    summary = gi.assemble(data, results, bundle, 2, out, hands_run=hands)

    assert summary["counts"]["objects"]["white_pot"] == {"observed": 2, "inferred": 2, "missing": 0}
    assert summary["fps"] == 20 and summary["gt_use"] == "none"
    assert json.loads((out / "intrinsics.json").read_text())["fx"] == 500
    assert (out / "objects/white_pot/mesh.glb").read_bytes() == b"glb"
    assert sorted(p.name for p in (out / "objects/white_pot/masks").iterdir()) == ["000000.png", "000002.png"]

    for f in range(4):
        # Camera files are camera->world (inverse of the odometry pose).
        c2w = gi.pose_record_to_matrix(json.loads((out / f"camera_c2w/{f:06d}.json").read_text()))
        assert np.allclose(c2w @ cams[f], np.eye(4), atol=1e-9)
        # Every frame has camera_from_object, observed or composed from world.
        o2c = gi.pose_record_to_matrix(json.loads((out / f"objects/white_pot/poses/{f:06d}.json").read_text()))
        assert np.allclose(o2c, cams[f] @ objs_world[f], atol=1e-9)
        depth = np.asarray(Image.open(out / f"depth/{f:06d}.png"))
        assert np.allclose(1.0 / (depth / 65535.0) - 1.0, 0.8, atol=1e-4)

    # Hand tracks of one side are merged; the longer track wins frame 1.
    right = out / "hands/right/poses"
    assert sorted(p.name for p in right.iterdir()) == [f"{f:06d}.json" for f in range(4)]
    assert json.loads((right / "000001.json").read_text())["track_id"] == 7
    assert json.loads((right / "000000.json").read_text())["track_id"] == 3
    assert (out / "hands/right/masks/000001.png").exists()


@pytest.mark.parametrize("field,value", [
    ("status", "absent"), ("status", None), ("status", "unknown"),
    ("gt_use", "absent"), ("gt_use", None), ("gt_use", "ground_truth_assisted"),
])
def test_object_initialization_requires_explicit_observation_provenance(tmp_path, field, value):
    _, results, _, _, cams, objs_world = build_tree(tmp_path)
    observed, _ = gi.object_pose_dirs(results, 2, "white_pot")
    path = observed / "000000.json"
    row = json.loads(path.read_text())
    if value == "absent":
        row.pop(field)
    else:
        row[field] = value
    path.write_text(json.dumps(row))
    poses, provenance = gi.object_camera_poses(results, 2, "white_pot", [0, 2], cams)
    assert provenance == {0: "inferred:world_hold", 2: "observed"}
    assert np.allclose(poses[0], cams[0] @ objs_world[0])
    # Without independent inferred geometry, the frame remains absent.
    _, inferred = gi.object_pose_dirs(results, 2, "white_pot")
    (inferred / "000000.json").unlink()
    poses, provenance = gi.object_camera_poses(results, 2, "white_pot", [0, 2], cams)
    assert 0 not in poses and 0 not in provenance


def test_assemble_refuses_non_empty_output(tmp_path):
    data, results, bundle, _, _, _ = build_tree(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale").write_text("x")
    with pytest.raises(ValueError):
        gi.assemble(data, results, bundle, 2, out)


def test_frame_window(tmp_path):
    data, results, bundle, _, _, _ = build_tree(tmp_path)
    summary = gi.assemble(data, results, bundle, 2, tmp_path / "out", start=1, frames_limit=2)
    assert summary["frames"] == [1, 2]


def test_invalid_hands_do_not_replace_supported_track_by_default(tmp_path):
    data, results, bundle, hands, _, _ = build_tree(tmp_path)
    d = hands / "align/records/99"
    d.mkdir()
    source = json.loads((hands / "align/records/7/000001.json").read_text())
    source["diagnostics"]["metric_valid"] = False
    for f in range(4):
        (d / f"{f:06d}.json").write_text(json.dumps({**source, "frame_idx": f, "track_id": 99}))
    summary = gi.assemble(data, results, bundle, 2, tmp_path / "out", hands_run=hands)
    assert summary["metric_valid_hands_only"] is True
    assert json.loads((tmp_path / "out/hands/right/poses/000001.json").read_text())["track_id"] == 7


def test_missing_camera_is_not_silently_initialized_to_identity(tmp_path):
    data, results, bundle, _, _, _ = build_tree(tmp_path)
    (results / "e002/camera/ego_cam_c/000002.json").unlink()
    with pytest.raises(ValueError, match="Missing independent camera"):
        gi.assemble(data, results, bundle, 2, tmp_path / "out")


def test_vo_provenance_is_preserved_and_cannot_pass_slam_requirement(tmp_path):
    data, results, bundle, _, _, _ = build_tree(tmp_path)
    for file in (results / "e002/camera/ego_cam_c").glob('*.json'):
        value = json.loads(file.read_text())
        value.update(method='stereo_pnp_ransac', gt_use='independent')
        file.write_text(json.dumps(value))
    out = tmp_path / 'strict'
    with pytest.raises(ValueError, match='SLAM'):
        gi.assemble(data, results, bundle, 2, out, require_slam=True)
    assert not out.exists()
    summary = gi.assemble(data, results, bundle, 2, tmp_path / 'diagnostic')
    assert summary['camera_is_slam'] is False
    assert summary['camera_provenance']['0']['kind'] == 'odometry'
    assert len(summary['camera_provenance']['0']['source_sha256']) == 64
    output = json.loads((tmp_path / 'diagnostic/camera_c2w/000000.json').read_text())
    assert output['method'] == 'stereo_pnp_ransac' and output['direction'] == 'camera_to_world'


def test_documented_slam_input_passes_strict_adapter(tmp_path):
    data, results, bundle, _, _, _ = build_tree(tmp_path)
    for file in (results / "e002/camera/ego_cam_c").glob('*.json'):
        value = json.loads(file.read_text())
        value.update(method='stereo_droid_slam', gt_use='independent')
        file.write_text(json.dumps(value))
    summary = gi.assemble(data, results, bundle, 2, tmp_path / 'out', require_slam=True)
    assert summary['camera_is_slam'] is True


def test_slam_camera_gauge_does_not_change_inferred_camera_object_pose(tmp_path):
    data, results, bundle, _, cams, objs_world = build_tree(tmp_path)
    slam = tmp_path / 'slam'
    slam.mkdir()
    expected = []
    for frame in range(4):
        M = np.eye(4)
        M[:3, 3] = [5, frame * .1, -2]
        expected.append(M)
        value = {**record(M), 'method': 'stereo_depth_droid_slam',
                 'gt_use': 'none', 'direction': 'camera_to_world', 'camera_frame': 'ego_cam_c'}
        (slam / f'{frame:06d}.json').write_text(json.dumps(value))
    out = tmp_path / 'out'
    summary = gi.assemble(data, results, bundle, 2, out, slam_camera_dir=slam, require_slam=True)
    assert summary['camera_is_slam'] and summary['full_episode']
    for frame in range(4):
        c2w = gi.pose_record_to_matrix(json.loads((out / f'camera_c2w/{frame:06d}.json').read_text()))
        object_pose = gi.pose_record_to_matrix(json.loads((out / f'objects/white_pot/poses/{frame:06d}.json').read_text()))
        assert np.allclose(c2w, expected[frame])
        assert np.allclose(object_pose, cams[frame] @ objs_world[frame])


def test_slam_override_rejects_ambiguous_direction_before_writing(tmp_path):
    data, results, bundle, _, _, _ = build_tree(tmp_path)
    slam = tmp_path / 'slam'
    slam.mkdir()
    (slam / '000000.json').write_text(json.dumps({**record(np.eye(4)), 'method': 'droid_slam', 'gt_use': 'none'}))
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='camera_to_world'):
        gi.assemble(data, results, bundle, 2, out, slam_camera_dir=slam)
    assert not out.exists()


@pytest.mark.parametrize("source", ["camera", "inferred"])
def test_gt_assisted_initialization_is_rejected(tmp_path, source):
    data, results, bundle, _, _, _ = build_tree(tmp_path)
    if source == "camera":
        p = results / "e002/camera/ego_cam_c/000001.json"
    else:
        p = results / "e002/inferred/e002/ego_object_poses/e002/white_pot/object_to_world/000001.json"
    r = json.loads(p.read_text())
    r["gt_use"] = "assisted"
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="GT-assisted"):
        gi.assemble(data, results, bundle, 2, tmp_path / "out")


@pytest.mark.parametrize("field,value", [("rotation", [0, 0, 0, 0]), ("translation", [0, float('nan'), 0]), ("scale", float('nan'))])
def test_nonfinite_or_degenerate_pose_is_rejected(field, value):
    r = record(np.eye(4))
    r[field] = value
    with pytest.raises(ValueError):
        gi.pose_record_to_matrix(r)
