"""Reference schema controls: preserve uncertainty and convert quaternion order."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/dex3_physics"))
from export_reference import export

spec = importlib.util.spec_from_file_location(
    "bundle_fixture", ROOT / "robotic_grounding/tests/test_track3_bundle.py"
)
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


def case(tmp_path):
    _, _, bundle = fixture_module.fixture(tmp_path)
    retarget = tmp_path / "retarget"
    retarget.mkdir()
    with np.load(bundle / "episode.npz", allow_pickle=False) as source:
        target = {key: source[key].copy() for key in ["frame_indices", "timestamps_s"]}
        target["robot_valid"] = source["hand_valid"].copy()
    q = np.full((3, 2, 14), np.nan)
    q[1, :, :3] = [1, 2, 3]
    q[1, :, 3:7] = Rotation.from_euler("xyz", [0.3, -0.2, 0.1]).as_quat()
    q[1, :, 7:] = 0.05
    target["robot_qpos"] = q
    for si, side in enumerate(["left", "right"]):
        target[f"robot_{side}_frame_poses"] = q[:, si, None, :7]
    np.savez_compressed(retarget / "retarget.npz", **target)
    manifest = {
        "episode_bundle_manifest_sha256": fixture_module.bundle.digest(
            bundle / "manifest.json"
        ),
        "payload": {"sha256": fixture_module.bundle.digest(retarget / "retarget.npz")},
        "joints": {
            side: {
                "finger_joint_names": [f"{side}_joint{i}" for i in range(7)],
                "frame_names": [f"{side}_hand_palm_link"],
            }
            for side in ["left", "right"]
        },
    }
    (retarget / "manifest.json").write_text(json.dumps(manifest))
    return bundle, retarget, q


def test_real_schema_roundtrip_preserves_unknowns_and_explicit_wxyz(tmp_path):
    bundle, retarget, q = case(tmp_path)
    out = tmp_path / "reference"
    report = export(bundle, retarget, out)
    row = pq.read_table(out / report["motion_path"]).to_pylist()[0]
    poses = np.array(row["ee_pose_w"])
    assert row["schema_version"] == "motion_v1"
    assert row["motion_kind"] == "dual_hand"
    assert row["robot_name"] == "dex3"
    assert np.isnan(poses[[0, 2]]).all()
    np.testing.assert_allclose(poses[1, 0, 3:], q[1, 0, [6, 3, 4, 5]], atol=1e-7)
    with (
        np.load(out / "source-masks.npz") as masks,
        np.load(bundle / "episode.npz") as source,
    ):
        for key in [
            "frame_indices",
            "timestamps_s",
            "hand_valid",
            "hand_directly_observed",
            "contact_known",
            "contact_probability",
            "wrench_known",
        ]:
            np.testing.assert_array_equal(masks[key], source[key])
    assert row["hand_contact_active"] == [[], []]
    assert not report["training_ready"] and not report["physical_task_accepted"]


def test_changed_targets_are_rejected_before_export(tmp_path):
    bundle, retarget, _ = case(tmp_path)
    with (retarget / "retarget.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="provenance"):
        export(bundle, retarget, tmp_path / "reference")
    assert not (tmp_path / "reference").exists()
