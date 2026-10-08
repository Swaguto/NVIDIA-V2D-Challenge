"""CPU tests for the issue-7 quality report on a synthetic scene archive."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refinement_report as rr


def axis_angle_matrix(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    a = np.linalg.norm(v)
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def cube(size=0.1):
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64) * size - size / 2
    f = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    return v, f


def synthetic_scene(T=60, fps=20.0, noise=0.004, seed=0):
    """Static cube 0.6 m ahead; a smoothly moving camera; a hand touching the
    cube's +x face (fingertips fixed on the face) in frames 20-39 and far
    away otherwise. "initial" adds per-frame noise to camera and object."""
    rng = np.random.default_rng(seed)
    verts, faces = cube()
    frames = np.arange(T)
    o2w = np.eye(4)
    o2w[:3, 3] = [0.0, 0.0, 0.6]
    w2c_ref = []
    for t in range(T):
        M = np.eye(4)
        M[:3, :3] = axis_angle_matrix([0, 0.004 * t, 0])
        M[:3, 3] = [0.002 * t, 0.0, 0.0]
        w2c_ref.append(M)
    w2c_ref = np.stack(w2c_ref)

    hand_local = rng.normal(scale=0.02, size=(778, 3))
    tips = list(rr.MANO_TIP_VERTS.values())
    for k, vid in enumerate(tips):
        hand_local[vid] = [0.05, -0.03 + 0.015 * k, 0.0]  # on the cube's +x face
    hand_local += [0.0, 0.0, 0.0]

    def state(perturb):
        w2c = w2c_ref.copy()
        o2c = np.einsum("tij,jk->tik", w2c_ref, o2w)
        hand_world = []
        for t in range(T):
            touching = 20 <= t < 40
            offset = np.array([0.0, 0.0, 0.6]) if touching else np.array([0.5, 0.3, 0.6])
            hand_world.append(hand_local + offset)
        hand_world = np.stack(hand_world)
        if perturb:
            for t in range(T):
                d = np.eye(4)
                d[:3, :3] = axis_angle_matrix(rng.normal(scale=0.003, size=3))
                d[:3, 3] = rng.normal(scale=noise, size=3)
                w2c[t] = d @ w2c[t]
                e = np.eye(4)
                e[:3, 3] = rng.normal(scale=noise, size=3)
                o2c[t] = e @ o2c[t]
                hand_world[t] = hand_world[t] + rng.normal(scale=noise, size=3)
        hand_cam = np.einsum("tij,tnj->tni", w2c_ref[:, :3, :3], hand_world) + w2c_ref[:, None, :3, 3]
        if perturb:
            # Hand vertices were generated with the true camera; re-express so
            # their world position under the perturbed camera keeps the noise.
            hand_cam = np.einsum("tij,tnj->tni", w2c[:, :3, :3], hand_world) + w2c[:, None, :3, 3]
        joints = hand_cam[:, :21]
        return {"camera_w2c": w2c, "object_0_o2c": o2c, "hand_0_verts": hand_cam.astype(np.float32),
                "hand_0_joints": joints.astype(np.float32)}

    scene = {
        "frame_indices": frames,
        "ref_frame_index": np.asarray(0),
        "fps": np.asarray(fps),
        "K": np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]),
        "image_size": np.array([640, 480]),
        "object_names": np.array(["box"]),
        "hand_sides": np.array(["right"]),
        "object_0_verts": verts.astype(np.float32),
        "object_0_faces": faces.astype(np.int32),
        "hand_0_frame_indices": frames,
        "hand_0_faces": np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32),
        "background_points": rng.normal(size=(500, 3)).astype(np.float32),
        "background_colors": rng.random((500, 3)).astype(np.float32),
    }
    for prefix, perturb in (("refined", False), ("initial", True)):
        for k, v in state(perturb).items():
            scene[f"{prefix}_{k}"] = v
    return scene


@pytest.fixture()
def scene_path(tmp_path):
    path = tmp_path / "scene.npz"
    np.savez_compressed(path, **synthetic_scene())
    return path


def test_rotation_log_roundtrip_including_near_pi():
    for v in ([0.0, 0.0, 0.0], [0.3, -0.2, 0.1], [0.0, np.pi - 1e-6, 0.0], [1.0, 1.0, 1.0]):
        R = axis_angle_matrix(v)
        got = rr.rotation_log(R)
        assert np.allclose(axis_angle_matrix(got), R, atol=1e-5)
    batch = np.stack([axis_angle_matrix([0.1 * i, 0, 0]) for i in range(4)])
    assert rr.rotation_log(batch).shape == (4, 3)


def test_translation_jitter_is_zero_for_constant_velocity_and_respects_gaps():
    frames = np.array([0, 1, 2, 3, 10, 11, 12])
    pos = np.stack([frames * 0.01, np.zeros(7), np.zeros(7)], axis=1)
    pos[4:] += 5.0  # a jump across the gap must not count
    out = rr.translation_jitter(pos, frames, fps=20.0)
    assert out["accel_rms_m_s2"] == pytest.approx(0.0, abs=1e-9)
    assert out["n"] == 3


def test_translation_jitter_units():
    frames = np.arange(5)
    pos = np.zeros((5, 3))
    pos[2, 0] = 0.001  # one 1 mm spike
    out = rr.translation_jitter(pos, frames, fps=10.0)
    # second differences: 1, -2, 1 mm -> 0.1, 0.2, 0.1 m/s^2 at 10 fps
    assert out["accel_rms_m_s2"] == pytest.approx(np.sqrt((0.01 + 0.04 + 0.01) / 3))


def test_rest_segment_drift_measures_motion_only_inside_segments():
    T = 30
    frames = np.arange(T)
    poses = np.tile(np.eye(4), (T, 1, 1))
    poses[:, 0, 3] = np.linspace(0, 0.03, T)  # 3 cm creep over the episode
    rest = np.ones(T, bool)
    rest[10:20] = False
    out = rr.rest_segment_drift(poses, np.zeros(3), rest, frames, min_len=5)
    assert len(out["segments"]) == 2
    assert out["rest_frames"] == 20
    assert out["endpoint_drift_cm_max"] == pytest.approx(9 * 0.03 / 29 * 100)


def test_evaluate_refined_beats_noisy_initial(scene_path):
    scene = rr.load_scene(scene_path)
    report = rr.evaluate(scene, None, rest_clearance=0.05, min_rest_len=5, penetration_m=0.005, surface_samples=4000)
    cam = report["camera"]
    # The synthetic camera centre accelerates slightly; the noise adds far more.
    assert cam["refined"]["translation"]["accel_rms_m_s2"] < 0.05
    assert cam["initial"]["translation"]["accel_rms_m_s2"] > 1.0
    assert cam["initial"]["rotation"]["ang_accel_rms_deg_s2"] > cam["refined"]["rotation"]["ang_accel_rms_deg_s2"]
    box = report["object_jitter"]["box"]
    assert box["refined"]["translation"]["accel_rms_m_s2"] < 1e-6
    assert box["refined"]["translation"]["accel_rms_m_s2"] < box["initial"]["translation"]["accel_rms_m_s2"]

    drift = report["drift"]["box"]
    # The hand is near the box only in frames 20-39: two rest segments.
    assert drift["refined"]["rest_frames"] == 40
    assert drift["refined"]["max_deviation_cm_max"] < 1e-6
    assert drift["initial"]["max_deviation_cm_max"] > 0.3

    contact = report["contact"]
    assert contact["run_source"].startswith("initial")
    assert contact["refined"]["contact_runs"] == 5  # one run per fingertip
    assert contact["refined"]["contact_point_spread_rms_mm"] < 1e-3
    assert contact["refined"]["contact_surface_distance_median_mm"] < 1.0
    assert contact["initial"]["contact_point_spread_rms_mm"] > contact["refined"]["contact_point_spread_rms_mm"]


def test_contact_runs_from_issue6_labels():
    rows = []
    for f in range(10):
        rows.append({"frame_id": f, "hand": "right", "fingertip": "index", "object_id": "box",
                     "raw_contact": f in (2, 3, 4, 7), "window_contact": f in (2, 3, 4, 5, 6, 7)})
    runs = rr.contact_runs_from_labels(rows, set(range(10)), min_len=3)
    assert [r.tolist() for r in runs[("right", "index", "box")]] == [[2, 3, 4, 5, 6, 7]]


def test_source_rest_intervals_are_fixed_even_when_hand_geometry_is_nearby():
    annotations = {"intervals": [{"object_id": "box", "frames": [20, 29],
                                  "evidence": "Source video shows stationary cube"}]}
    report = rr.evaluate(synthetic_scene(), None, 0.05, 5, 0.005, 1000, annotations)
    drift = report["source_reviewed_drift"]["box"]
    assert drift["initial"]["segments"][0]["frames"] == [20, 29]
    assert drift["refined"]["rest_frames"] == 10
    assert drift["refined"]["endpoint_drift_cm_max"] < 1e-6
    annotations["intervals"][0]["frames"] = [50, 65]
    with pytest.raises(ValueError, match="absent frames"):
        rr.evaluate(synthetic_scene(), None, 0.05, 5, 0.005, 1000, annotations)


def test_cli_outputs(scene_path, tmp_path):
    out = tmp_path / "report"
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"object_provenance": {"box": {str(f): "observed" if f % 2 else "inferred:world_hold"
                                                                for f in range(60)}}}))
    sys.argv = ["refinement_report.py", "--scene", str(scene_path), "--episode", 2, "--out", str(out),
                "--inputs", str(inputs), "--surface-samples", "2000"]
    sys.argv = [str(a) for a in sys.argv]
    rr.main()
    report = json.loads((out / "report.json").read_text())
    assert report["episode"] == 2
    assert report["object_coverage"]["box"]["observed_initialization"] == 30
    assert report["object_coverage"]["box"]["inferred_initialization"] == 30
    # Alternating inferred rows cannot supply observed acceleration triples.
    assert report["object_observed_jitter"]["box"]["initial"]["translation"]["n"] == 0
    md = (out / "report.md").read_text()
    assert "## Jitter" in md and "## Drift" in md and "## Contact consistency" in md
    html = (out / "viewer.html").read_text(encoding="utf-8")
    assert "three@0.160.0" in html and '"objects"' in html
    pose = json.loads((out / "scoring/refined/e002/ego_object_poses/e002/box/object_to_world/000001.json").read_text())
    assert pose["status"] == "observed" and pose["coordinate_frame"] == "fixed_world"
    assert json.loads((out / "scoring/refined/e002/ego_object_poses/e002/box/object_to_world/000000.json").read_text())["status"] == "inferred"
    # The exported world pose is camera-to-world composed with object-to-camera.
    scene = rr.load_scene(scene_path)
    expected = rr.invert_rigid(scene["refined_camera_w2c"][1]) @ scene["refined_object_0_o2c"][1]
    assert np.allclose(pose["translation"], expected[:3, 3], atol=1e-6)


def test_decimate_reduces_faces():
    rng = np.random.default_rng(0)
    verts = rng.random((3000, 3))
    faces = rng.integers(0, 3000, size=(6000, 3))
    v, f = rr.decimate(verts, faces, 1000)
    assert len(f) <= 6000 and f.max() < len(v)


def test_decimate_deduplicates_faces_and_preserves_winding():
    verts = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.tile([[2, 0, 1], [1, 0, 2]], (50, 1))
    simplified_verts, simplified_faces = rr.decimate(verts, faces, 1)
    assert len(simplified_faces) == 1
    triangle = simplified_verts[simplified_faces[0]]
    assert np.allclose(triangle, verts[faces[0]])


def test_contact_missing_geometry_stays_in_denominator():
    key = ("right", "index", "box")
    row = (np.zeros(3), np.array([1, 0, 0]), 0.001, 0.001)
    tables = {key: {0: row, 2: row}}
    result = rr.contact_precision(tables, {key: [np.array([0, 1, 2])]}, 0.005)
    assert result["expected_contact_rows"] == 3
    assert result["missing_contact_rows"] == 1
    assert result["contact_geometry_coverage"] == pytest.approx(2 / 3)
    assert result["contact_within_2cm_fraction"] == pytest.approx(2 / 3)


def test_completely_missing_contact_geometry_is_reported():
    key = ("left", "thumb", "box")
    result = rr.contact_precision({}, {key: [np.arange(3)]}, 0.005)
    assert result["missing_contact_rows"] == 3
    assert result["contact_within_2cm_fraction"] == 0.0
    assert result["contact_surface_distance_median_mm"] is None


def test_unobserved_hands_do_not_become_rest_segments():
    scene = synthetic_scene()
    keep = np.r_[0:10, 20:60]
    scene["hand_0_frame_indices"] = keep
    for state in rr.STATES:
        for kind in ("verts", "joints"):
            scene[f"{state}_hand_0_{kind}"] = scene[f"{state}_hand_0_{kind}"][keep]
    report = rr.evaluate(scene, None, 0.05, 5, 0.005, 1000)
    for state in rr.STATES:
        segments = report["drift"]["box"][state]["segments"]
        assert all(not (s["frames"][0] <= 10 <= s["frames"][1]) for s in segments)
    assert report["contact"]["independent_precision"] is None
