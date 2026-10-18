"""Source authority, missing-point and camera contracts for fresh GPU hand fits."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location("recover_source_hand", Path(__file__).parents[1]/"recover_source_hand.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def recovery_request(tmp_path):
    image = tmp_path/"source.png"
    Image.new("RGB", (128, 64)).save(image)
    evidence = tmp_path/"evidence.json"
    evidence.write_text('{"estimated":true}')
    bundle = tmp_path/"episode.npz"
    np.savez(bundle, frame_indices=[5], timestamps_s=[.25], hand_directly_observed=np.array([[True, False]]), intrinsics=[[50.,0,60.],[0,50.,30.],[0,0,1.]])
    arrays = tmp_path/"observations.npz"
    points = np.full((21, 2), np.nan)
    points[:4] = [[60, 25], [65, 25], [65, 30], [60, 30]]
    np.savez(arrays, frame_indices=[5], landmarks_px=points[None])
    keypoints = np.zeros((21, 3))
    keypoints[:4, :2] = points[:4]
    keypoints[:4, 2] = .5
    row = dict(frame_idx=5, timestamp_s=.25, track_id=3, is_right=True, original_direct_observation=False,
        evidence_kind="checked_inferred_image_flow", identity_anchor_frames=[0, 1, 2, 3, 4], keypoints=keypoints.tolist(),
        crop=dict(cx=64, cy=32, size=40), fixed_betas=[0.]*10, fixed_hand_scale=.9, image=str(image), image_sha256=m.digest(image))
    data = dict(schema=m.SCHEMA, source_hand="right", camera=dict(fx=50., fy=50., cx=60., cy=30., width=128, height=64),
        evidence_receipt=str(evidence), evidence_sha256=m.digest(evidence), evidence_arrays=str(arrays), evidence_arrays_sha256=m.digest(arrays),
        canonical_bundle=str(bundle), canonical_bundle_sha256=m.digest(bundle), frames=[row])
    path = tmp_path/"request.json"
    path.write_text(json.dumps(data))
    return path, data


def test_validation_runs_without_loading_gpu_dependencies(recovery_request):
    path, _ = recovery_request
    assert len(m.validate_request(path)["frames"]) == 1


def test_digest_preserves_sha256_on_python_without_file_digest(tmp_path,monkeypatch):
    data=b"private-source-contract"*100000
    path=tmp_path/'large-source.bin';path.write_bytes(data)
    expected=hashlib.sha256(data).hexdigest()
    monkeypatch.delattr(hashlib,'file_digest',raising=False)
    assert m.digest(path)==expected


@pytest.mark.parametrize("change", [{"original_direct_observation":True}, {"timestamp_s":.3}, {"is_right":"right"}, {"is_right":False}])
def test_reject_changed_canonical_authority(recovery_request, change):
    path, data = recovery_request
    data["frames"][0].update(change)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):m.validate_request(path)


def test_reject_fabricated_missing_finger(recovery_request):
    path, data = recovery_request
    data["frames"][0]["keypoints"][5] = [63, 28, .5]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fabricated"):m.validate_request(path)


def test_reject_changed_calibration(recovery_request):
    path, data = recovery_request
    data["camera"]["fx"] = 55.
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="calibration"):m.validate_request(path)


def test_reject_changed_source_and_duplicate_frame(recovery_request):
    path, data = recovery_request
    data["frames"].append(copy.deepcopy(data["frames"][0]))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate"):m.validate_request(path)
    data["frames"].pop()
    Image.new("RGB", (128, 64), "red").save(data["frames"][0]["image"])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source image"):m.validate_request(path)


def test_mask_cannot_invent_unseen_joints(recovery_request):
    _, data = recovery_request
    points = data["frames"][0]["keypoints"]
    mask = np.ones((64,128), bool)
    supported = m.associated_keypoints(points, mask)
    assert np.count_nonzero(supported[:,2]) == 4
    with pytest.raises(ValueError):m.associated_keypoints(points, np.zeros_like(mask))


def test_record_preserves_estimation_flags_shape_and_calibrated_ray(recovery_request):
    _, data = recovery_request
    row = data["frames"][0]
    pred = dict(hand_pose=np.tile(np.eye(3),(15,1,1)), global_orient=np.eye(3), betas=np.ones(10), pred_cam_t_full=[.1,.05,2.], scaled_focal_length=100.)
    record = m.make_record(row, pred, data["camera"], "abc")
    assert record["fresh_image_supported"] and record["estimated"]
    assert not record["directly_observed"] and not record["metric_valid"] and not record["visual_valid"]
    assert record["mano"]["betas"] == [0.]*10
    np.testing.assert_allclose(record["cam_t"], [.18, .09, 1.])
    assert row["original_direct_observation"] is False
