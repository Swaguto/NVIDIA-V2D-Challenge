from copy import deepcopy
import numpy as np
import pytest
from v2d_reliability.track3 import (discover_split, experiment_matrix, validate_trajectory,
    validate_pair, transform_poses, validate_time_mapping, validate_bound_sidecar)
from v2d_reliability.io import sha256
from v2d_reliability.core import fit_normalizer, score, SIGNALS


def trajectory():
    return {"schema_version": "track3_trajectory_v1", "episode_id": 3,
            "units": "m", "quaternion_order": "wxyz", "coordinate_frame": "episode_world",
            "frame_provenance": "synthetic fixture", "object_ids": ["cup"],
            "asset_sha256": {"cup": "a" * 64}, "frame_index": [0, 1, 2],
            "timestamps": [0., .05, .1], "visible": [[True], [False], [True]],
            "poses": [[[0, 0, 0, 1, 0, 0, 0]], [[0]*7], [[1, 0, 0, 1, 0, 0, 0]]]}


def test_oracle_ignores_invisible_zero_pose():
    r = trajectory()
    a, b, valid = validate_pair(r, deepcopy(r))
    assert np.array_equal(a[valid], b[valid]) and valid.sum() == 2


@pytest.mark.parametrize("key,value", [("units", "mm"), ("quaternion_order", "xyzw"),
    ("coordinate_frame", "camera"), ("timestamps", [0, 0, .1]), ("frame_index", [0, 2, 3]),
    ("asset_sha256", {}), ("visible", [[1], [0], [1]])])
def test_invalid_contract(key, value):
    r = trajectory(); r[key] = value
    with pytest.raises(ValueError): validate_trajectory(r)


@pytest.mark.parametrize("q", [[0, 0, 0, 0], [2, 0, 0, 0], [float("nan"), 0, 0, 0]])
def test_bad_visible_quaternion(q):
    r = trajectory(); r["poses"][0][0][3:] = q
    with pytest.raises(ValueError): validate_trajectory(r)


def test_candidate_cannot_hide_error_or_shift_time():
    r = trajectory(); c = deepcopy(r); c["visible"][0][0] = False
    with pytest.raises(ValueError): validate_pair(r, c)
    c = deepcopy(r); c["timestamps"] = [.01, .06, .11]
    with pytest.raises(ValueError): validate_pair(r, c)
    c = deepcopy(r)
    for key in ("poses", "timestamps", "frame_index", "visible"): c[key] = c[key][:-1]
    with pytest.raises(ValueError): validate_pair(r, c)


def test_rigid_transform_rotates_orientation_and_position():
    q = [2**-.5, 0, 0, 2**-.5]
    pose = np.array([[1, 0, 0, 1, 0, 0, 0.]])
    got = transform_poses(pose, q, [1, 2, 3])
    np.testing.assert_allclose(got[0, :3], [1, 3, 3])
    np.testing.assert_allclose(got[0, 3:], q)
    np.testing.assert_array_equal(pose[0], [1, 0, 0, 1, 0, 0, 0])


def test_sparse_ids_fixed_proxy_and_matched_matrix(tmp_path):
    data = tmp_path / "data/chunk-000"; data.mkdir(parents=True)
    for ep in [0, 3, 7, 11, 23, 31, 39]: (data / f"episode_{ep:06d}.parquet").write_bytes(b"fixture")
    split = discover_split(tmp_path)
    assert split["development"] == [3, 7]
    jobs = experiment_matrix(split, 100)
    assert len(jobs) == 2 * 3 * 4
    assert {j["environment_steps"] for j in jobs} == {100}
    assert all(j["status"] == "planned" for j in jobs)
    split["development"].append(0)
    with pytest.raises(ValueError): experiment_matrix(split, 100)


def test_explicit_resampling_and_warmup():
    m = {"source_timestamps": [0, .05, .1], "motion_timestamps": [0, .025, .05, .075, .1],
         "source_time_at_motion": [0, .025, .05, .075, .1],
         "simulation_timestamps": [5, 5.025, 5.05, 5.075, 5.1], "warmup_seconds": 5}
    assert validate_time_mapping(m)
    m["simulation_timestamps"][0] = 4.99
    with pytest.raises(ValueError): validate_time_mapping(m)


def test_sidecar_reference_binding_and_order(tmp_path):
    reference = tmp_path / "motion.parquet"; reference.write_bytes(b"fixture")
    d = {"schema_version": "observations_v1", "sequence_id": "episode_000003",
         "hand_ids": ["left", "right"], "object_ids": ["cup"], "timestamps": [0, .05],
         "signal_names": list(SIGNALS), "signals": np.ones((2, 2, 1, 4)).tolist(),
         "valid": np.ones((2, 2, 1, 4), dtype=bool).tolist(), "reference_sha256": sha256(reference)}
    s = score(d, fit_normalizer([d], [d["sequence_id"]]), variant="baseline")
    args = (reference, d["sequence_id"], d["timestamps"], d["hand_ids"], d["object_ids"])
    assert np.all(validate_bound_sidecar(s, *args) == 1)
    with pytest.raises(ValueError): validate_bound_sidecar(s, reference, d["sequence_id"], d["timestamps"], ["right", "left"], ["cup"])
    reference.write_bytes(b"changed")
    with pytest.raises(ValueError): validate_bound_sidecar(s, *args)


def test_comparison_requires_matched_budgets_and_retains_failures():
    from v2d_reliability.track3 import summarize_matched, VARIANTS
    rows = [{"episode_id": 2, "seed": seed, "variant": variant, "environment_steps": 100,
             "status": "completed", "rollout_complete": True, "score_kind": "proxy",
             "reference_sha256": "ref", "physics_config_hash": "physics", "evaluation_config_hash": "eval",
             "metrics": {"add_auc": .6}} for seed in (17, 29, 43) for variant in VARIANTS]
    assert summarize_matched(rows, [2])["summary"][0]["metric_mean"] == pytest.approx(.6)
    with pytest.raises(ValueError): summarize_matched(rows[:-1], [2])
    rows[-1]["environment_steps"] = 99
    with pytest.raises(ValueError): summarize_matched(rows, [2])
    rows[-1].update(environment_steps=100, status="failed", rollout_complete=False, metrics={})
    result = summarize_matched(rows, [2])["summary"][-1]
    assert result["failed_runs"] == 1 and result["metric_mean"] is None


@pytest.mark.parametrize("episode,allowed", [(2, True), (0, False)])
def test_cli_fit_accepts_fixed_split_and_rejects_proxy(tmp_path, episode, allowed):
    from v2d_reliability.cli import main
    from v2d_reliability.io import write_json, read_json
    from v2d_reliability.track3 import PROXY_EPISODES
    data = tmp_path / "data"; data.mkdir()
    for ep in (*PROXY_EPISODES, 2): (data / f"episode_{ep:06d}.parquet").write_bytes(b"fixture")
    split = tmp_path / "split.json"; write_json(split, discover_split(tmp_path))
    doc = {"schema_version": "observations_v1", "sequence_id": f"episode_{episode:06d}",
           "hand_ids": ["left"], "object_ids": ["cup"], "timestamps": [0, .05],
           "signal_names": list(SIGNALS), "signals": np.ones((2, 1, 1, 4)).tolist(),
           "valid": np.ones((2, 1, 1, 4), dtype=bool).tolist()}
    observations = tmp_path / "observations.json"; write_json(observations, doc)
    output = tmp_path / "normalizer.json"
    argv = ["fit", "--split", str(split), "--observations", str(observations), "--output", str(output)]
    if allowed:
        assert main(argv) == 0
        assert read_json(output)["development_ids"] == ["episode_000002"]
    else:
        with pytest.raises(ValueError): main(argv)
        assert not output.exists()
