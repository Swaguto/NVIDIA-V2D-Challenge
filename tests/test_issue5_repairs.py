"""CPU regressions for issue 5 pose boundaries, provenance, rendering and scoring."""
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from v2d.world_calib.vm import fit_ego_rig as rig
from v2d.world_calib.vm import fit_object_pose as fit
from v2d.world_calib.vm import visualize_object_pose as vis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))
sys.path.insert(0, str(ROOT / "scripts/track3"))
import official_metric as metric
import score_track3_official as scorer
import convert_camera_poses as convert
import register_poses as registration
import smooth_rotation

torch.set_num_threads(1)


def transform(angle=0, translation=(0, 0, 0)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
    value[:3, 3] = translation
    return value


def test_quaternion_boundaries_random_signs_and_half_turns():
    rotations = np.concatenate([Rotation.random(100, random_state=4).as_quat(),
        Rotation.from_euler("xyz", [[0, 0, 0], [90, 0, 0], [180, 0, 0], [0, 179.999999, 0]], degrees=True).as_quat()])
    for q in rotations:
        for sign in (1, -1):
            W = rig._w_obj(np.r_[[.1, -.2, .7], sign * 3 * q])
            np.testing.assert_allclose(W[:3, :3], Rotation.from_quat(q).as_matrix(), atol=1e-10)
            np.testing.assert_allclose(rig.js_to_mat(rig.mat_to_js(W)), W, atol=1e-10)
            assert np.linalg.det(W[:3, :3]) == pytest.approx(1, abs=1e-10)


@pytest.mark.parametrize("q", [[0, 0, 0, 0], [np.nan, 0, 0, 1], [np.inf, 0, 0, 1]])
def test_invalid_visible_quaternion_rejected(q):
    with pytest.raises(ValueError):
        rig._w_obj(np.r_[[0, 0, 0], q])


@pytest.mark.parametrize("scale", [1., 1.2])
def test_registration_restores_centers_and_camera_space_warmstart(scale):
    generator = torch.Generator().manual_seed(7)
    cad = torch.randn((80, 3), generator=generator, dtype=torch.float64) * torch.tensor([.02, .03, .06])
    cad += torch.tensor([.04, -.03, .02])
    R = torch.tensor(Rotation.from_euler("xyz", [25, -15, 40], degrees=True).as_matrix())
    translation = torch.tensor([.1, .2, .7], dtype=torch.float64)
    for offset in (0., .17):
        truth = translation + torch.tensor([offset, -.1 * offset, 0])
        dst = scale * (cad @ R.T) + truth
        s, rotation, t, residual = registration.register(cad, dst, R0=R, t0=truth,
            s0=scale, allow_scale=scale != 1)
        assert s == pytest.approx(scale, abs=1e-10)
        torch.testing.assert_close(rotation, R, atol=1e-10, rtol=0)
        torch.testing.assert_close(t, truth, atol=1e-10, rtol=0)
        assert residual < 1e-8


def test_registration_coarse_path_preserves_nonzero_scan_origin():
    cad = torch.randn(100, 3, dtype=torch.float64) * .03 + torch.tensor([.08, -.03, .02])
    truth = torch.tensor([.1, .2, .7], dtype=torch.float64)
    _, R, t, residual = registration.register(cad, cad + truth)
    torch.testing.assert_close(R, torch.eye(3, dtype=torch.float64), atol=1e-7, rtol=0)
    torch.testing.assert_close(t, truth, atol=1e-7, rtol=0)
    assert residual < 1e-7


def test_small_trimmed_cloud_and_invalid_registration():
    x = torch.randn(16, 3, dtype=torch.float64)
    assert registration._trimmed_cost(x, x, .6)[0] < 1e-7
    with pytest.raises(ValueError):
        registration.register(x[:2], x)


def test_snapshot_smoothing_is_order_independent_and_does_not_fill():
    raw = [transform(2 * i, (i * i * .01, 0, 0)) for i in range(7)]
    raw[3] = None
    copies = [None if x is None else x.copy() for x in raw]
    forward = rig.smooth_poses(raw, frame_ids=np.arange(7))
    backward = rig.smooth_poses(raw[::-1], frame_ids=np.arange(7)[::-1])[::-1]
    assert forward[3] is None
    for original, before, a, b in zip(raw, copies, forward, backward):
        if original is not None:
            np.testing.assert_allclose(original, before)
            np.testing.assert_allclose(a, b, atol=1e-10)


def test_temporal_gates_use_elapsed_time_and_reject_jumps():
    assert not fit.temporal_check(transform(), transform(80), jump_deg=40)
    assert fit.temporal_check(transform(), transform(80), frame_delta=3, jump_deg=40)
    assert not fit.temporal_check(transform(), transform(80), frame_delta=3, time_delta=.02, fps=30)
    with pytest.raises(ValueError):
        fit.temporal_check(transform(), transform(), time_delta=-.1)


def test_final_pose_metrics_detect_smoothed_pose_moving_off_observations():
    generator = np.random.default_rng(0)
    mesh = generator.normal(size=(100, 3)) * .03
    normals = mesh / np.linalg.norm(mesh, axis=1, keepdims=True)
    true = transform(20, (.1, .2, .7))
    cloud = (true[:3, :3] @ mesh.T).T + true[:3, 3]
    good = rig.pose_metrics(true, mesh, normals, np.eye(4), cloud, np.empty((0, 3)), .01)
    bad = true.copy(); bad[:3, 3] += [.2, .2, 0]
    bad_metrics = rig.pose_metrics(bad, mesh, normals, np.eye(4), cloud, np.empty((0, 3)), .01)
    assert rig.inlier_pass(good, .85)
    assert not rig.inlier_pass(bad_metrics, .85)
    assert good["cam_right"]["rmse"] is None


def test_triangle_zbuffer_and_full_mesh_scene_transforms(tmp_path):
    import trimesh
    vertices = np.array([[-.2, -.2, 1], [.2, -.2, 1], [0, .2, 1]])
    vertices = np.concatenate([vertices * 2, vertices])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    K = np.array([[30, 0, 16], [0, 30, 16], [0, 0, 1.]])
    depth = vis.render_depth(vertices, faces, K, (32, 32))
    assert depth[16, 16] == pytest.approx(1)
    img = np.zeros((32, 32, 3), np.uint8)
    vis._render_mesh(img, vertices, faces, K, (255, 255, 0))
    assert img[16, 16, 0] > 0
    large = trimesh.creation.icosphere(subdivisions=5)
    assert len(large.vertices) > 4000
    scene = trimesh.Scene()
    scene.add_geometry(large, node_name="a", geom_name="a", transform=transform(0, (2, 0, 0)))
    scene.add_geometry(trimesh.creation.box(), node_name="b", geom_name="b", transform=transform(0, (-2, 0, 0)))
    path = tmp_path / "scene.glb"; scene.export(path)
    loaded = rig.load_mesh(path)
    assert len(loaded.vertices) == len(large.vertices) + 8
    assert loaded.bounds[0, 0] == pytest.approx(-2.5)
    assert loaded.bounds[1, 0] == pytest.approx(3)
    assert loaded.faces.max() < len(loaded.vertices)


def test_shared_camera_overlay_uses_hash_and_derived_right_view(tmp_path):
    import hashlib
    path = tmp_path / "camera.json"
    C = transform(20, (.1, -.3, .5))
    path.write_text(json.dumps(rig.mat_to_js(C)))
    source = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    M = transform(0, (.075, 0, 0))
    np.testing.assert_allclose(vis.camera_from_source(source, "right", "left", "right", M), M @ C)
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        vis.camera_from_source(source, "left", "left", "right", M)


def test_moving_camera_and_rectified_conversion(tmp_path):
    source, cameras, out = tmp_path / "native", tmp_path / "camera", tmp_path / "fixed"
    X = transform(35, (.1, -.2, .8))
    H = transform(11)
    for frame in (10, 12, 15):
        C = transform(frame * 2, (.01 * frame, -.1, .2))
        P = H @ C @ X
        for obj in ("a", "b"):
            p = source / "e002/ego_object_poses/e002" / obj / "object_to_cam/ego_cam_c" / f"{frame:06d}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({**rig.mat_to_js(P), "frame_index": frame,
                "coordinate_frame": "rectified_camera", "timestamp": frame / 30, "gt_use": "none"}))
        p = cameras / "ego_cam_c" / f"{frame:06d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**rig.mat_to_js(C), "frame_index": frame,
            "gt_use": "independent", "timestamp": frame / 30}))
    with pytest.raises(ValueError, match="R1"):
        convert.convert_episode(source, cameras, out, 2)
    assert not out.exists()
    assert convert.convert_episode(source, cameras, out, 2, rectification=H[:3, :3], require_independent=True) == 6
    data = scorer.load_candidate_json(out, 2, "object_to_world", ["a", "b"], 3, [10, 12, 15])
    for row in data:
        np.testing.assert_allclose(row[:, :3], np.broadcast_to(X[:3, 3], (2, 3)), atol=1e-10)
        np.testing.assert_allclose(Rotation.from_quat(row[:, 3:]).as_matrix(), np.broadcast_to(X[:3, :3], (2, 3, 3)), atol=1e-10)


def test_json_loader_rejects_duplicate_rows_and_invalid_quaternions(tmp_path):
    d = tmp_path / "ego_object_poses/e002/a/object_to_world"
    d.mkdir(parents=True)
    content = json.dumps(rig.mat_to_js(np.eye(4)))
    (d / "000010.json").write_text(content)
    (d / "10.json").write_text(content)
    with pytest.raises(ValueError, match="duplicate"):
        scorer.load_candidate_json(tmp_path, 2, "object_to_world", ["a"], 1, [10])
    (d / "10.json").unlink()
    (d / "000010.json").write_text(json.dumps({"translation": [0, 0, 0], "rotation": [0, 0, 0, 0]}))
    with pytest.raises(ValueError, match="invalid"):
        scorer.load_candidate_json(tmp_path, 2, "object_to_world", ["a"], 1, [10])


def test_npy_loader_and_gap_fill_flags(tmp_path):
    data = np.zeros((3, 2, 7)); data[..., 6] = 1
    data[0, 0] = np.nan; data[1, 1] = np.nan
    np.save(tmp_path / "e002.npy", data)
    loaded = scorer.load_candidate_json(tmp_path, 2, "object_to_world", ["a", "b"], 3)
    filled, coverage = scorer.gap_fill(loaded)
    assert coverage == pytest.approx(4 / 6)
    assert np.isfinite(filled).all()
    loaded[2, 0, 0] = np.inf
    with pytest.raises(ValueError, match="malformed"):
        scorer.gap_fill(loaded)


def test_metric_cli_uses_time_then_world_axis(tmp_path, monkeypatch, capsys):
    reference = np.zeros((6, 2, 7)); reference[..., 6] = 1
    reference[:, 1, 0] = .1
    candidate = tmp_path / "candidate.json"; gt = tmp_path / "reference.json"
    candidate.write_text(json.dumps({"poses": reference.tolist(), "frame_index": list(range(6))}))
    gt.write_text(json.dumps({"poses": reference.tolist(), "frame_index": list(range(6)),
        "vertices": np.random.default_rng(0).normal(size=(2, 32, 3)).tolist()}))
    monkeypatch.setattr(sys, "argv", ["metric", "--candidate", str(candidate), "--reference", str(gt)])
    assert metric.main() == 0
    assert "add_auc: 1.000000" in capsys.readouterr().out
    with pytest.raises(ValueError, match="reference"):
        metric.score_episode(reference[None], reference, np.zeros((2, 32, 3)))


def official_kernels(path):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) or isinstance(n, ast.Assign)
        and all(isinstance(t, ast.Name) and t.id != "_VERTICES" for t in n.targets)]
    scope = {"np": np, "Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


@pytest.mark.parametrize("filename", ["AUC.py", "SP-SR.py", "MP-SR.py", "RPE.py", "MPPE.py"])
def test_official_kernel_parity_multiworld_scale_and_moving_camera(filename):
    kit = Path(os.environ.get("V2D_SUBMISSION_KIT",
        ROOT.parent / "runs/issue5_diagnostic_20261003/kit/v2d_submission_kit")) / "metric_code/track_3"
    if not kit.exists():
        pytest.skip("set up the official participant kit for the parity check")
    official = official_kernels(kit / filename)
    generator = np.random.default_rng(2)
    ref = np.zeros((6, 2, 7)); ref[..., 6] = 1
    ref[:, 0, :3] = [.1, .2, .7]; ref[:, 1, :3] = [.2, .2, .7]
    vertices = generator.normal(scale=.03, size=(2, 32, 3))
    for mode in ("oracle", "offset", "moving", "body_offset", "noise", "scale", "multiworld"):
        candidate = ref[:, None].copy()
        if mode == "multiworld":
            candidate = np.repeat(candidate, 3, axis=1)
        for t in range(6):
            for w in range(candidate.shape[1]):
                C = transform(5 * t if mode == "moving" else 25 * w if mode == "multiworld" else 20 if mode == "offset" else 0,
                              (.03 * t, 0, 0) if mode == "moving" else (.3 * w, 0, 0) if mode == "multiworld" else (0, 0, 0))
                candidate[t, w, :, :3] = (C[:3, :3] @ ref[t, :, :3].T).T + C[:3, 3]
                q = Rotation.from_matrix(C[:3, :3]).as_quat()
                candidate[t, w, :, 3:] = metric.quaternion_multiply(q, candidate[t, w, :, 3:])
        if mode == "body_offset": candidate[:, :, 1, 0] += .1
        if mode == "noise": candidate[..., :3] += generator.normal(scale=.02, size=candidate[..., :3].shape)
        if mode == "scale": candidate[..., :3] *= 1.2
        reference = ref[:, None]
        aligned = official["_warp_to_reference_start"](candidate, reference)
        expected = {
            "add_auc": official["_add_auc"](official["_object_add"](aligned, reference, vertices)),
            "spider_sr": float(np.mean(official["_spider_success"](aligned, reference))),
            "maniptrans_sr": float(np.mean(official["_maniptrans_success"](aligned, reference, np.arange(2)))),
            "rpe_cm": official["_relative_position_error_cm"](aligned, reference),
            "mppe_cm": official["_mean_per_point_error_cm"](aligned, reference),
        }
        assert metric.score_episode(candidate, ref, vertices) == pytest.approx(expected, abs=1e-9)


def test_smoothing_refuses_existing_or_nested_output(tmp_path, monkeypatch):
    source = tmp_path / "source"; source.mkdir()
    (source / "sentinel").write_text("retain")
    for out in (source, source / "nested", tmp_path):
        monkeypatch.setattr(sys, "argv", ["smooth", "--predictions", str(source), "--out", str(out)])
        with pytest.raises(SystemExit): smooth_rotation.main()
        assert (source / "sentinel").read_text() == "retain"


def test_weighted_rotation_mean_uses_linear_weights():
    q = Rotation.from_euler("z", [[0], [90]], degrees=True).as_quat()
    mean = Rotation.from_quat(smooth_rotation.geodesic_mean(q, [1, 3]))
    assert mean.as_euler("xyz", degrees=True)[2] == pytest.approx(np.degrees(np.arctan(3)))


def test_rectified_baseline_from_real_calibration():
    import stereo_depth
    calibration = Path(os.environ.get("V2D_CALIB", ROOT / "tests/fixtures/track3_camera_calibration.json"))
    stereo = stereo_depth.load_rig(str(calibration))
    assert stereo.baseline_m == pytest.approx(abs(stereo.P2[0, 3] / stereo.P2[0, 0]), abs=1e-12)
    assert stereo.baseline_m == pytest.approx(np.linalg.norm(stereo.T_lr), abs=1e-10)


def test_camera_fitter_selected_gt_index_export_and_saved_residual(tmp_path, monkeypatch):
    data, work, out = tmp_path / 'data', tmp_path / 'work', tmp_path / 'camera_out'
    (data / 'meta').mkdir(parents=True)
    (data / 'meta/camera_calibration.json').write_text('{}')
    objects = ('unused', 'target')
    worlds = [transform(25, (.1, .2, .7)), transform(75, (-.2, .1, .6))]
    cameras = [transform(35, (.2, -.1, .3)), transform(-15, (-.1, .3, .2))]
    frames = np.array([10, 12])
    poses = np.zeros((2, 2, 7)); poses[..., 6] = 1
    for row, W in enumerate(worlds):
        poses[row, 1] = np.r_[W[:3, 3], Rotation.from_matrix(W[:3, :3]).as_quat()]
    reference = SimpleNamespace(object_names=objects, steps=2, pose_xyzw=poses,
        visible=np.array([[False, True], [False, True]]), frame_index=frames,
        capture_time=np.array([1., 1.1]))
    loader = SimpleNamespace(load_reference=lambda *args: reference,
        object_mesh_dir=lambda names, root: {name: name for name in names})
    mesh = np.random.default_rng(0).normal(size=(50, 3)) * .03
    normals = mesh / np.linalg.norm(mesh, axis=1, keepdims=True)
    monkeypatch.setattr(rig, 'load_reference_loader', lambda *args: loader)
    monkeypatch.setattr(rig, 'ego_rig_matrix', lambda *args: ('left', 'right', transform(0, (.075, 0, 0))))
    monkeypatch.setattr(rig, 'mesh_points_and_normals', lambda *args: (mesh, normals))
    monkeypatch.setattr(rig, 'load_intrinsics', lambda *args: (np.eye(3), 32, 32))
    estimated = []
    def estimate(work, obj, steps, b, *args):
        estimated.append((obj, b)); return .75, [.01]
    monkeypatch.setattr(rig, 'estimate_right_depth_scale', estimate)
    def cloud(*args, **kwargs):
        row = np.flatnonzero(frames == args[4])[0]
        P = cameras[row] @ worlds[row]
        return (P[:3, :3] @ mesh.T).T + P[:3, 3]
    monkeypatch.setattr(rig, 'frame_cloud', cloud)
    def solve(seed, mesh_w, normal_w, *args):
        errors = [np.linalg.norm(mesh_w.mean(0) - ((W[:3, :3] @ mesh.T).T + W[:3, 3]).mean(0)) for W in worlds]
        row = int(np.argmin(errors))
        return cameras[row], {'cam_left': {'count': 50, 'inlier': 1., 'rmse': 0.},
                              'cam_right': {'count': 0, 'inlier': 0., 'rmse': None}}
    monkeypatch.setattr(rig, 'solve_frame', solve)
    monkeypatch.setattr(sys, 'argv', ['rig', '--data-root', str(data), '--work', str(work),
        '--out', str(out), '--objects', 'target', '--no-smooth', '--est-right-scale'])
    assert rig.main() == 0
    assert estimated == [('target', 1)]
    report = json.loads((out / 'report.json').read_text())
    assert report['objects']['target']['passed'] == 2
    for row, frame in enumerate(frames):
        p = out / f'target/object_to_cam/left/{frame:06d}.json'
        saved = json.loads(p.read_text())
        np.testing.assert_allclose(rig.js_to_mat(saved), cameras[row] @ worlds[row], atol=1e-10)
        assert saved['gt_use'] == 'gt_assisted'
        assert report['objects']['target']['framedetails'][str(frame)]['rmse_left'] < 1e-8
    assert not (out / 'target/object_to_cam/left/000000.json').exists()


def test_overlay_repeated_invocation_excludes_vis_and_uses_shared_source(tmp_path, monkeypatch):
    import cv2
    import hashlib
    import trimesh
    data, work = tmp_path / 'data', tmp_path / 'work'
    (data / 'meta').mkdir(parents=True)
    (data / 'meta/camera_calibration.json').write_text('{}')
    output = work / 'ego_object_poses/e002/target/object_to_world'; output.mkdir(parents=True)
    camera = tmp_path / 'shared.json'; camera.write_text(json.dumps(rig.mat_to_js(np.eye(4))))
    source = {'path': str(camera), 'sha256': hashlib.sha256(camera.read_bytes()).hexdigest()}
    (output / '000010.json').write_text(json.dumps({**rig.mat_to_js(transform(0, (0, 0, 1))), 'camera_source': source}))
    mesh = trimesh.creation.icosphere(subdivisions=5, radius=.1)
    loader = SimpleNamespace(object_mesh_dir=lambda names, root: {n: n for n in names})
    monkeypatch.setattr(vis, 'load_reference_loader', lambda *args: loader)
    monkeypatch.setattr(vis, 'ego_rig_matrix', lambda *args: ('left', 'right', np.eye(4)))
    monkeypatch.setattr(vis, 'load_mesh', lambda *args: mesh)
    monkeypatch.setattr(vis, 'load_intrinsics', lambda *args: (np.array([[100, 0, 16], [0, 100, 16], [0, 0, 1.]]), 32, 32))
    monkeypatch.setattr(vis, '_load_rgb_first', lambda *args: np.zeros((32, 32, 3), np.uint8))
    sizes = []
    def render(img, vertices, faces, *args):
        sizes.append((len(vertices), int(faces.max())))
        img[16, 16] = [255, 255, 0]
    monkeypatch.setattr(vis, '_render_mesh', render)
    monkeypatch.setattr(sys, 'argv', ['vis', '--data-root', str(data), '--work', str(work), '--episode', '2'])
    assert vis.main() == vis.main() == 0
    assert len(sizes) == 2
    assert all(n > 4000 and maximum < n for n, maximum in sizes)
    img = cv2.imread(str(work / 'ego_object_poses/e002/vis/target/frames/000010.png'))
    assert img[16, 16, 0] > 0


def test_segmentation_label_matching_longest_name_and_detector_order():
    matcher = registration.Segmenter.__new__(registration.Segmenter)
    matcher.names = ['white_pot', 'white_pot_lid']
    matcher.phrase2idx = {'white pot': 0, 'white pot lid': 1}
    matcher._order = sorted(matcher.phrase2idx, key=len, reverse=True)
    assert matcher._assign(['white pot lid', 'white pot', 'unrelated']) == [1, 0, None]


def test_episode_scorer_structured_report_keeps_gaps_and_inferred_status(tmp_path, monkeypatch):
    predictions = tmp_path / 'predictions'
    ids = np.array([10, 12, 15, 16, 17, 20])
    poses = np.zeros((6, 2, 7)); poses[..., 6] = 1; poses[:, 1, 0] = .1
    reference = SimpleNamespace(object_names=('a', 'b'), steps=6, pose_xyzw=poses,
        frame_index=ids, capture_time=ids / 30., visible=np.ones((6, 2), bool))
    for row, frame in enumerate(ids):
        for body, name in enumerate(reference.object_names):
            if row == 0 and body == 0:
                continue
            path = predictions / 'ego_object_poses/e002' / name / 'object_to_world' / f'{frame:06d}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**rig.mat_to_js(transform(0, poses[row, body, :3])),
                'frame_index': int(frame), 'gt_use': 'independent',
                'status': 'inferred' if row == 3 and body == 1 else 'observed'}))
    vertices = np.random.default_rng(4).normal(scale=.03, size=(2, 32, 3))
    monkeypatch.setattr(scorer, 'load_reference', lambda *args: reference)
    monkeypatch.setattr(scorer, 'load_vertex_clouds', lambda *args, **kwargs: vertices)
    monkeypatch.setattr(scorer, 'batch_vertices_for', lambda *args: vertices)
    report_path = tmp_path / 'diagnostic.json'
    monkeypatch.setattr(sys, 'argv', ['score', '--predictions', str(predictions), '--data', str(tmp_path),
        '--episodes', '2', '--report', str(report_path), '--oracle'])
    assert scorer.main() == 0
    report = json.loads(report_path.read_text())
    row = report['episodes'][0]
    assert row['frame_index'] == ids.tolist()
    assert row['gap_fill_mask'][0][0]
    assert row['provenance']['inferred'][3][1]
    assert row['raw_observed_coverage_by_object'] == pytest.approx({'a': 5/6, 'b': 5/6})
    assert row['metrics'] == pytest.approx({'add_auc': 1, 'spider_sr': 1, 'maniptrans_sr': 1,
                                          'rpe_cm': 0, 'mppe_cm': 0}, abs=1e-9)


def test_episode_scorer_rejects_per_object_camera_provenance(tmp_path):
    for body, name in enumerate(('a', 'b')):
        path = tmp_path / 'ego_object_poses/e002' / name / 'object_to_world/000010.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({**rig.mat_to_js(np.eye(4)),
            'camera_source': {'sha256': str(body), 'path': f'camera_{body}.json'}}))
    with pytest.raises(ValueError, match='different camera'):
        scorer.load_candidate_json(tmp_path, 2, 'object_to_world', ['a', 'b'], 1, [10])


def test_world_scorer_refuses_raw_camera_coordinates(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['score', '--predictions', str(tmp_path), '--data', str(tmp_path),
                                    '--kind', 'object_to_cam'])
    with pytest.raises(SystemExit) as error: scorer.main()
    assert error.value.code == 2
