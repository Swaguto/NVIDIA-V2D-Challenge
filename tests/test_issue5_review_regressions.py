"""CLI regressions for failures found after the initial issue-5 draft.

All generated trajectories/assets here are synthetic controls, not real tracking.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from v2d.world_calib.vm import fit_ego_rig as rig

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/eval"), str(ROOT / "scripts/track3")]
import convert_camera_poses as convert
import score_track3_official as scorer
import smooth_rotation
import register_poses as registration


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def command(script, *args, env=None, optimized=False):
    return subprocess.run([sys.executable, *(["-O"] if optimized else []),
        str(ROOT / script), *map(str, args)], env=env, text=True,
        capture_output=True, timeout=45)


@pytest.mark.parametrize("existing_out", [False, True])
def test_right_scale_output_guard_runs_before_loading_or_estimation(tmp_path, monkeypatch, existing_out):
    argv = ["fit", "--data-root", str(tmp_path / "missing-data"),
            "--work", str(tmp_path / "work"), "--est-right-scale"]
    if existing_out:
        out = tmp_path / "existing"
        out.mkdir(); (out / "sentinel").write_text("keep")
        argv += ["--out", str(out)]
    def unexpected(*args):
        pytest.fail("must reject before reading inputs or estimating scale")
    monkeypatch.setattr(rig, "estimate_right_depth_scale", unexpected)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        rig.main()
    assert error.value.code == 2
    if existing_out:
        assert (out / "sentinel").read_text() == "keep"


def test_camera_only_smoothing_rejected_without_output(tmp_path, monkeypatch):
    source, out = tmp_path / "camera", tmp_path / "filtered"
    write_json(source / "e000/ego_object_poses/e000/obj/object_to_cam/ego_cam_c/000000.json",
               {**rig.mat_to_js(np.eye(4)), "coordinate_frame": "original_camera"})
    monkeypatch.setattr(sys, "argv", ["smooth", "--predictions", str(source), "--out", str(out)])
    with pytest.raises(SystemExit) as error:
        smooth_rotation.main()
    assert error.value.code == 2
    assert not out.exists()


@pytest.mark.parametrize("rectified", [False, True])
def test_moving_camera_convert_then_world_smooth_remains_scoreable(tmp_path, monkeypatch, rectified):
    raw, cameras, fixed, filtered = [tmp_path / n for n in ("raw", "cameras", "world", "filtered")]
    X = np.eye(4)
    X[:3, :3] = Rotation.from_euler("xyz", [25, -15, 70], degrees=True).as_matrix()
    X[:3, 3] = [.2, -.1, .8]
    H = np.eye(4)
    H[:3, :3] = Rotation.from_euler("x", 11, degrees=True).as_matrix()
    frames = [100, 104, 109, 110]
    originals = []
    for i, frame in enumerate(frames):
        C = np.eye(4)
        C[:3, :3] = Rotation.from_euler("xyz", [5*i, -9*i, 23*i], degrees=True).as_matrix()
        C[:3, 3] = [.05*i, .01*i, -.03*i]
        metadata = {"episode": 0, "frame_index": frame, "timestamp": i / 30,
                    "status": "observed", "camera_frame": "ego_cam_c"}
        write_json(cameras / f"ego_cam_c/{frame:06d}.json",
            {**rig.mat_to_js(C), **metadata, "gt_use": "independent", "coordinate_frame": "original_camera"})
        path = raw / f"e000/ego_object_poses/e000/obj/object_to_cam/ego_cam_c/{frame:06d}.json"
        write_json(path, {**rig.mat_to_js((H @ C if rectified else C) @ X), **metadata,
            "gt_use": "none", "residual": .001,
            "coordinate_frame": "rectified_camera" if rectified else "original_camera"})
        originals.append(path.read_bytes())
    assert convert.convert_episode(raw, cameras, fixed, 0, rectification=H[:3, :3] if rectified else None,
                                   require_independent=True) == len(frames)
    monkeypatch.setattr(sys, "argv", ["smooth", "--predictions", str(fixed), "--out", str(filtered)])
    smooth_rotation.main()
    poses, metadata = scorer.load_candidate_json(filtered, 0, "object_to_world", ["obj"],
                                                  len(frames), np.array(frames), True)
    for frame, pose in zip(frames, poses[:, 0]):
        np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_matrix(), X[:3, :3], atol=1e-10)
        np.testing.assert_allclose(pose[:3], X[:3, 3], atol=1e-10)
        record = json.loads((filtered / f"e000/ego_object_poses/e000/obj/object_to_world/{frame:06d}.json").read_text())
        assert record["smoothing"]["residual_validated"] is False
    assert not np.asarray(metadata["observed"]).any()
    assert np.asarray(metadata["inferred"]).all()
    assert metadata["gt_use"] == ["independent"]
    assert originals == [p.read_bytes() for p in sorted(raw.glob("**/object_to_cam/ego_cam_c/*.json"))]


def test_smoothing_even_window_uses_source_ids_and_preserves_gaps(tmp_path, monkeypatch):
    source, out = tmp_path / "flat-episode", tmp_path / "filtered"
    frames, angles = [0, 1, 2, 3, 10], [0, 20, 40, 60, 140]
    for frame, angle in zip(frames, angles):
        X = np.eye(4); X[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
        write_json(source / f"obj/object_to_world/{frame:06d}.json",
                   {**rig.mat_to_js(X), "coordinate_frame": "fixed_world", "residual": .001})
    monkeypatch.setattr(sys, "argv", ["smooth", "--predictions", str(source), "--out", str(out), "--window", "2"])
    smooth_rotation.main()
    paths = sorted((out / "obj/object_to_world").glob("*.json"))
    assert [int(p.stem) for p in paths] == frames
    for path, expected in zip(paths, [0, 10, 30, 50, 140]):
        record = json.loads(path.read_text())
        got = Rotation.from_quat(np.array(record["rotation"])[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)[2]
        assert got == pytest.approx(expected, abs=1e-10)
        assert "residual" not in record


@pytest.mark.parametrize("bad", ["original_camera", "rejected"])
def test_invalid_world_smoothing_track_creates_no_partial_output(tmp_path, monkeypatch, bad):
    source, out = tmp_path / "source", tmp_path / "out"
    record = {**rig.mat_to_js(np.eye(4)), "coordinate_frame": "fixed_world", "status": "observed"}
    write_json(source / "a/object_to_world/000000.json", record)
    record["coordinate_frame" if bad == "original_camera" else "status"] = bad
    write_json(source / "b/object_to_world/000000.json", record)
    monkeypatch.setattr(sys, "argv", ["smooth", "--predictions", str(source), "--out", str(out)])
    with pytest.raises(ValueError):
        smooth_rotation.main()
    assert not out.exists()


def diagnostic_fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    data, pred = tmp_path / "public", tmp_path / "pred"
    write_json(data / "meta/info.json", {"fps": 30})
    rows = []
    for i in range(12):
        frame = 100 + 4*i
        objects = []
        for b, name in enumerate(("pan", "brush")):
            R = Rotation.from_euler("z", 3*i + 17*b, degrees=True)
            xyz = np.array([.03*i + .2*b, .04*b, .7])
            q = R.as_quat()[[3, 0, 1, 2]]
            objects.append({"name": name, "pose": [*xyz, *q], "visible": True})
            write_json(pred / f"e000/ego_object_poses/e000/{name}/object_to_cam/ego_cam_c/{frame:06d}.json",
                       {"rotation": q.tolist(), "translation": xyz.tolist()})
        rows.append({"frame_index": frame, "capture_time": i/30, "observation.objects": objects})
    table = pa.Table.from_pylist(rows).replace_schema_metadata({b"objects": b'["pan", "brush"]'})
    path = data / "data/chunk-000/episode_000000.parquet"; path.parent.mkdir(parents=True)
    pq.write_table(table, path)
    env = os.environ.copy(); env["V2D_PUBLIC"] = str(data); env.pop("V2D_REPO", None)
    return pred, env


@pytest.mark.parametrize("script", ["diag_gt_orientation", "diag_relative_geometry", "diag_rotation_offset"])
def test_diagnostic_cli_uses_public_override_and_nonordinal_ids(tmp_path, script):
    pred, env = diagnostic_fixture(tmp_path)
    result = command(f"scripts/track3/{script}.py", pred, "0", env=env)
    assert result.returncode == 0, result.stderr
    if script == "diag_gt_orientation":
        assert "scanned 1 public episodes" in result.stdout
    elif script == "diag_relative_geometry":
        assert "frames with both predictions: 12/12" in result.stdout
        assert "median 0.00 cm" in result.stdout
        assert "t= 100" in result.stdout and "diff=  +0.00" in result.stdout
    else:
        assert "frames=12" in result.stdout and "skipped" not in result.stdout
        assert "mean offset   0.00 deg" in result.stdout


def test_local_kit_size_bound_before_read_or_output(tmp_path):
    archive, out = tmp_path / "huge.zip", tmp_path / "out"
    with archive.open("wb") as file:
        file.truncate(5_000_001)
    result = command("scripts/fetch_issue5_test_inputs.py", "--output", out, "--kit-zip", archive)
    assert result.returncode != 0 and "download bound" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("failure", ["subset", "changed_hash", "corrupt_bytes", "calibration", "timestamp"])
def test_validator_rejects_bad_inputs_under_optimized_python(tmp_path, failure):
    # Synthetic pinned fixture isolates integrity checks; no real asset/accuracy claim.
    repo, data, out = tmp_path / "repo", tmp_path / "data", tmp_path / "out"
    evidence = repo / "docs/team/issue5/evidence"
    contents = {"mesh/a/a.glb": b"mesh-a", "mesh/b/b.glb": b"mesh-b"}
    files = []
    for key, value in contents.items():
        path = data / key; path.parent.mkdir(parents=True); path.write_bytes(value)
        files.append({"path": key, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()})
    expected = {"dataset": "synthetic-test-only", "revision": "fixture", "files": files}
    write_json(evidence / "asset_download_manifest.json", expected)
    calibration = data / "meta/camera_calibration.json"; write_json(calibration, {})
    write_json(evidence / "local_results.json", {"camera_calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(), "episodes": []})
    supplied = json.loads(json.dumps(expected))
    message = "pinned asset"
    if failure == "subset":
        supplied["files"].pop(); message = "pinned asset metadata"
    elif failure == "changed_hash":
        supplied["files"][1]["sha256"] = "0" * 64; message = "pinned asset metadata"
    elif failure == "corrupt_bytes":
        (data / "mesh/b/b.glb").write_bytes(b"broken")
    elif failure == "calibration":
        write_json(calibration, {"changed": True}); message = "calibration/reference hash"
    elif failure == "timestamp":
        message = "invalid capture times"
    write_json(data / "asset_download_manifest.json", supplied)
    setup = f"import sys; sys.path.insert(0, {str(ROOT / 'scripts/eval')!r}); import validate_issue5_inputs as v; from pathlib import Path; v.REPO=Path({str(repo)!r}); "
    if failure == "timestamp":
        setup += "v.validate_pinned_inputs=lambda data: ({'dataset':'fixture','revision':'fixture','files':[]},'sha','sha'); v.load_rig=lambda _: type('Rig',(),{'left':'l','right':'r','baseline_m':.1,'T_lr':v.np.eye(4),'fx_rect':100,**{k:v.np.eye(4) for k in ('R1','R2','P1','P2','Q')}})(); v.load_reference=lambda *a: type('Ref',(),{'capture_time':v.np.array([0.,float('nan')])})(); "
    setup += f"v.run(Path({str(data)!r}),Path({str(out)!r}))"
    result = subprocess.run([sys.executable, "-O", "-c", setup], text=True, capture_output=True, timeout=45)
    assert result.returncode != 0 and message in result.stderr, result.stderr
    assert not (out / "local_results.json").exists()


def test_depth_probe_imports_checkout_stereo_module(tmp_path):
    env = os.environ.copy()
    env["V2D_PUBLIC"] = str(tmp_path / "missing-videos")
    env["V2D_CALIB"] = str(ROOT / "tests/fixtures/track3_camera_calibration.json")
    env["V2D_OUT"] = str(tmp_path / "probe")
    env.pop("V2D_RUNTIME", None)
    result = command("scripts/track3/probe_depth.py", "0", "1", env=env)
    assert result.returncode != 0 and "matching, nonempty" in result.stderr, result.stderr
    assert "baseline=" in result.stdout


def test_single_object_segmenter_does_not_assign_table_mask_to_pan():
    import torch
    class Inputs(dict):
        def to(self, device):
            return self
        @property
        def input_ids(self):
            return self["input_ids"]
    detector = {"boxes": torch.tensor([[0., 0., 8., 8.], [2., 2., 4., 4.]]),
                "text_labels": ["table", "mini dust pan"], "scores": torch.tensor([.9, .8])}
    class Processor:
        def __call__(self, **kwargs):
            return Inputs(input_ids=None, original_sizes=None)
        def post_process_grounded_object_detection(self, *args, **kwargs):
            return [detector]
        def post_process_masks(self, *args, **kwargs):
            masks = torch.ones((2, 1, 8, 8), dtype=torch.bool)
            masks[1] = False; masks[1, 0, 2:4, 2:4] = True
            return [masks]
    segmenter = registration.Segmenter.__new__(registration.Segmenter)
    segmenter.names = ["mini_dust_pan"]
    segmenter.device = "cpu"; segmenter.prompt = "mini dust pan."
    segmenter.phrase2idx = {"mini dust pan": 0}; segmenter._order = ["mini dust pan"]
    segmenter.gp = segmenter.sp = Processor()
    segmenter.gm = lambda *args, **kwargs: None
    segmenter.sm = lambda *args, **kwargs: SimpleNamespace(pred_masks=None)
    masks, detections = segmenter(np.zeros((8, 8, 3), np.uint8))
    assert masks.sum() == 4 and masks[0, 2:4, 2:4].all()
    assert not masks[0, 0, 0]
    assert [item[:2] for item in detections] == [(None, "table"), ("mini_dust_pan", "mini dust pan")]
    assert [x["box_xyxy"] for x in segmenter.last_candidates] == [[0., 0., 8., 8.], [2., 2., 4., 4.]]
    assert [x["mask_pixels"] for x in segmenter.last_candidates] == [64, 4]
    assert [x["object"] for x in segmenter.last_candidates] == [None, "mini_dust_pan"]
    detector.update(boxes=torch.empty((0, 4)), text_labels=[], scores=torch.empty(0))
    masks, detections = segmenter(np.zeros((8, 8, 3), np.uint8))
    assert not masks.any() and not detections and segmenter.last_candidates == []


@pytest.mark.parametrize("failure", [None, "mask", "residual", "read"])
def test_native_tracker_records_rejections_and_fails_zero_solve(tmp_path, monkeypatch, failure):
    import torch
    ref = SimpleNamespace(object_names=["mini_dust_pan"], frame_index=np.array([100, 104]),
                          capture_time=np.array([1., 1.1]))
    monkeypatch.setattr(registration, "load_reference", lambda *args: ref)
    monkeypatch.setattr(registration, "DEV", "cpu")
    monkeypatch.setattr(registration, "load_cad", lambda *args: np.random.default_rng(0).normal(size=(80, 3)) * .03)
    rig_fixture = SimpleNamespace(left="left", right="right", baseline_m=.1,
        **{k: np.eye(4) for k in ("R1", "R2", "P1", "P2", "Q")})
    monkeypatch.setattr(registration, "load_rig", lambda: rig_fixture)
    def read(*args):
        if failure == "read":
            raise RuntimeError("synthetic missing frame")
        return np.zeros((32, 32, 3), np.uint8)
    rig_fixture.rectify_pair = lambda a, b: (a, b)
    monkeypatch.setattr(registration, "read_frame", read)
    monkeypatch.setattr(registration, "stereo_depth", lambda *args: (np.ones((32, 32)), np.ones((32, 32), bool)))
    yy, xx = np.indices((32, 32)); cloud = np.stack([xx.ravel()*.01, yy.ravel()*.01, np.ones(1024)], axis=1)
    monkeypatch.setattr(registration, "backproject_full", lambda *args: cloud)
    segmenter = lambda *args: (
        np.full((1, 32, 32), failure != "mask", bool), [(None, "table", .9)] if failure == "mask" else [("mini_dust_pan", "mini dust pan", .9)])
    segmenter.last_candidates = [{"object": "mini_dust_pan", "label": "mini dust pan",
                                  "score": .9, "box_xyxy": [0., 0., 32., 32.], "mask_pixels": 1024}]
    monkeypatch.setattr(registration, "Segmenter", lambda names: segmenter)
    monkeypatch.setattr(registration, "register", lambda *args, **kwargs: (
        1., torch.eye(3), torch.tensor([0., 0., 1.]), .1 if failure == "residual" else .001))
    out = tmp_path / "pred"
    monkeypatch.setattr(sys, "argv", ["register", "--episode", "39", "--out", str(out)])
    if failure:
        with pytest.raises(SystemExit, match="incomplete tracking"):
            registration.main()
    else:
        registration.main()
    report = json.loads((out / "registration_report.json").read_text())
    assert report["accepted_by_object"] == {"mini_dust_pan": 0 if failure else 2}
    paths = list(out.glob("**/object_to_cam/*/*.json"))
    assert len(paths) == (0 if failure else 2)
    if failure == "read":
        assert report["input_errors"][0]["frame_index"] == 100
    else:
        assert sorted(report["frames"]) == ["100", "104"]
        assert report["frames"]["100"]["detection_candidates"] == segmenter.last_candidates
        detail = report["frames"]["100"]["objects"]["mini_dust_pan"]
        assert detail["reason"] == {None: None, "mask": "insufficient_masked_depth", "residual": "residual_gate"}[failure]


@pytest.mark.parametrize("gap", ["mask", "residual", "error", "source_ids"])
def test_native_tracker_reinitializes_after_rejection_or_source_gap(tmp_path, monkeypatch, gap):
    import torch
    frames = [0, 1, 4, 5] if gap == "source_ids" else [0, 1, 2, 3]
    ref = SimpleNamespace(object_names=["pan"], frame_index=np.array(frames),
                          capture_time=np.array(frames) / 20.)
    monkeypatch.setattr(registration, "load_reference", lambda *args: ref)
    monkeypatch.setattr(registration, "DEV", "cpu")
    monkeypatch.setattr(registration, "load_cad", lambda *args: np.ones((80, 3)))
    rig_fixture = SimpleNamespace(left="left", right="right", baseline_m=.1,
        **{k: np.eye(4) for k in ("R1", "R2", "P1", "P2", "Q")})
    rig_fixture.rectify_pair = lambda a, b: (a, b)
    monkeypatch.setattr(registration, "load_rig", lambda: rig_fixture)
    current = {}
    def read(path, frame):
        current["frame"] = frame
        return np.zeros((32, 32, 3), np.uint8)
    monkeypatch.setattr(registration, "read_frame", read)
    monkeypatch.setattr(registration, "stereo_depth", lambda *args: (np.ones((32, 32)), np.ones((32, 32), bool)))
    yy, xx = np.indices((32, 32))
    cloud = np.stack([xx.ravel()*.01, yy.ravel()*.01, np.ones(1024)], axis=1)
    monkeypatch.setattr(registration, "backproject_full", lambda *args: cloud)
    monkeypatch.setattr(registration, "Segmenter", lambda names: lambda *args: (
        np.full((1, 32, 32), not (gap == "mask" and current["frame"] == 1), bool), []))
    calls = {}
    def solve(*args, **kwargs):
        calls[current["frame"]] = kwargs["R0"] is not None
        if gap == "error" and current["frame"] == 1:
            raise RuntimeError("synthetic solve failure")
        return 1., torch.eye(3), torch.tensor([0., 0., 1.]), (
            .1 if gap == "residual" and current["frame"] == 1 else .001)
    monkeypatch.setattr(registration, "register", solve)
    out = tmp_path / "pred"
    monkeypatch.setattr(sys, "argv", ["register", "--episode", "39", "--out", str(out)])
    registration.main()
    assert calls[frames[0]] is False
    assert calls[frames[2]] is False, "a gap must restart global registration"
    assert calls[frames[3]] is True, "consecutive accepted observations may warm-start"
    report = json.loads((out / "registration_report.json").read_text())
    assert report["frames"][str(frames[2])]["objects"]["pan"]["warm_start_frame_index"] is None
    assert report["frames"][str(frames[3])]["objects"]["pan"]["warm_start_frame_index"] == frames[2]
    if gap != "source_ids":
        assert not list(out.glob("**/object_to_cam/*/000001.json"))


def prepared_mask_fixture(tmp_path, monkeypatch):
    import cv2
    calibration = tmp_path / "calibration.json"
    write_json(calibration, {"fixture": "synthetic calibration"})
    monkeypatch.setattr(registration, "CALIB_PATH", str(calibration))
    rig_fixture = SimpleNamespace(left="left", right="right", baseline_m=.1, size=(32, 32),
        **{k: np.eye(4) for k in ("R1", "R2", "P1", "P2", "Q")})
    directory = tmp_path / "observations"
    directory.mkdir()
    mask = directory / "pan.png"
    cv2.imwrite(str(mask), np.full((32, 32), 255, dtype=np.uint8))
    item = {"path": "pan.png", "sha256": hashlib.sha256(mask.read_bytes()).hexdigest()}
    record = {"schema_version": 1, "episode": 39, "objects": ["pan"], "camera_frame": "left",
              "coordinate_frame": "rectified_camera", "resolution": [32, 32], "gt_use": "none",
              "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
              "rectification": {k: np.eye(4).tolist() for k in ("R1", "R2", "P1", "P2", "Q")},
              "producer": {"kind": "generated test fixture"}, "frames": {"0": {"pan": item}}}
    path = directory / "manifest.json"
    write_json(path, record)
    return path, record, rig_fixture


@pytest.mark.parametrize("failure", ["axes", "calibration", "gt", "rectification", "bytes", "resolution", "path"])
def test_prepared_masks_reject_incompatible_or_changed_observations(tmp_path, monkeypatch, failure):
    import cv2
    path, record, rig_fixture = prepared_mask_fixture(tmp_path, monkeypatch)
    if failure == "axes":
        record["coordinate_frame"] = "original_camera"
    elif failure == "calibration":
        record["calibration_sha256"] = "0" * 64
    elif failure == "gt":
        record["gt_use"] = "gt_assisted"
    elif failure == "rectification":
        record["rectification"]["R1"][0][0] = 0
    elif failure in ("bytes", "resolution"):
        mask = path.parent / "pan.png"
        cv2.imwrite(str(mask), np.zeros((16, 16), np.uint8))
        if failure == "resolution":
            record["frames"]["0"]["pan"]["sha256"] = hashlib.sha256(mask.read_bytes()).hexdigest()
    else:
        record["frames"]["0"]["pan"]["path"] = "../calibration.json"
    write_json(path, record)
    with pytest.raises(ValueError):
        source = registration.PreparedMasks(path, 39, ["pan"], rig_fixture, require_no_gt=True)
        source.load(0, (32, 32, 3))


def test_native_tracker_consumes_frozen_masks_and_preserves_provenance(tmp_path, monkeypatch):
    import torch
    path, record, rig_fixture = prepared_mask_fixture(tmp_path, monkeypatch)
    ref = SimpleNamespace(object_names=["pan"], frame_index=np.array([0, 1]), capture_time=np.array([0., .05]))
    monkeypatch.setattr(registration, "load_reference", lambda *args: ref)
    monkeypatch.setattr(registration, "DEV", "cpu")
    monkeypatch.setattr(registration, "load_cad", lambda *args: np.ones((80, 3)))
    monkeypatch.setattr(registration, "load_rig", lambda: rig_fixture)
    rig_fixture.rectify_pair = lambda a, b: (a, b)
    monkeypatch.setattr(registration, "read_frame", lambda *args: np.zeros((32, 32, 3), np.uint8))
    monkeypatch.setattr(registration, "stereo_depth", lambda *args: (np.ones((32, 32)), np.ones((32, 32), bool)))
    yy, xx = np.indices((32, 32))
    cloud = np.stack([xx.ravel()*.01, yy.ravel()*.01, np.ones(1024)], axis=1)
    monkeypatch.setattr(registration, "backproject_full", lambda *args: cloud)
    monkeypatch.setattr(registration, "Segmenter", lambda *args: pytest.fail("perception must not rerun"))
    monkeypatch.setattr(registration, "register", lambda *args, **kwargs: (1., torch.eye(3), torch.tensor([0., 0., 1.]), .001))
    out = tmp_path / "pred"
    monkeypatch.setattr(sys, "argv", ["register", "--episode", "39", "--out", str(out), "--mask-manifest", str(path)])
    registration.main()
    report = json.loads((out / "registration_report.json").read_text())
    assert report["accepted_by_object"] == {"pan": 1}
    assert report["models"] is None
    pose_path = next(out.glob("**/object_to_cam/*/000000.json"))
    pose = json.loads(pose_path.read_text())
    assert pose["gt_use"] == "metadata_only"
    assert pose["mask_sha256"] == record["frames"]["0"]["pan"]["sha256"]
    assert pose["observation_source"]["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not list(out.glob("**/object_to_cam/*/000001.json"))
    assert report["frames"]["1"]["detections"] == [{"object": "pan", "status": "missing_prepared_mask"}]
