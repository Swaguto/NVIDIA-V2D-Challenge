"""Reject altered private inputs and unbounded/mutable policy runs before Docker."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from run_scored_policy import TRAINING, validate


def test_no_training_with_both_hands_parked():
    from run_scored_policy import require_command_anchors

    require_command_anchors({"selected_anchor_slots": [4, 0]})
    for slots in ([0, 0], [False, 1], [-1, 2], [2]):
        with pytest.raises(ValueError, match="command anchors"):
            require_command_anchors({"selected_anchor_slots": slots})


@pytest.fixture
def config(tmp_path):
    c = {
        "episode": 3,
        "iterations": 100,
        "max_seconds": 1800,
        "training_image": TRAINING,
    }
    for key in ("bundle", "support", "context", "robot_assets", "kit"):
        p = tmp_path / key
        p.mkdir()
        c[key] = str(p)
    files = {}
    for side in ("left", "right"):
        name = f"urdfs/g1/dex3_{side}.urdf"
        path = Path(c["robot_assets"]) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test URDF " + side)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    lock = tmp_path / "robot-assets.lock.json"
    lock.write_text(
        json.dumps({"schema": "v2d.dex3.robot-assets.v1", "files_sha256": files})
    )
    c["robot_asset_lock"] = str(lock)
    source = Path(c["context"]) / "code/producer.py"
    source.parent.mkdir()
    source.write_text("# committed preparation code\n")
    (
        Path(c["context"])
        / "code/robotic_grounding/source/robotic_grounding/robotic_grounding/assets"
    ).mkdir(parents=True)
    manifest = {
        "schema": "v2d.scored-policy-context.v1",
        "source_commit": "test",
        "files": {"code/producer.py": hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    manifest.update(
        {
            name: False
            for name in (
                "private_inputs_included",
                "robot_assets_included",
                "model_weights_included",
                "credentials_included",
            )
        }
    )
    for name in ("check_native_scene.py", "audit_initial_contacts.py"):
        path = Path(c["context"]) / "code/scripts/dex3_training" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# committed diagnostic fixture\n")
        manifest["files"][path.relative_to(c["context"]).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (Path(c["context"]) / "context-manifest.json").write_text(json.dumps(manifest))
    (Path(c["bundle"]) / "manifest.json").write_text(
        json.dumps(
            {
                "episode": 3,
                "provided_meshes_read": False,
                "reference_pose_labels_read": False,
            }
        )
    )
    evaluator = Path(c["kit"]) / "examples/run_policy_evaluation.py"
    evaluator.parent.mkdir()
    evaluator.touch()
    return c


def test_locked_label_free_case_is_ready_for_bounded_training(config):
    assert validate(config) == "test"
    for changed in (
        {"iterations": 101},
        {"max_seconds": 0},
        {"training_image": "image:latest"},
    ):
        with pytest.raises(ValueError):
            validate(dict(config, **changed))


@pytest.mark.parametrize("kind", ["robot", "code"])
def test_modified_asset_or_preparation_code_is_rejected(config, kind):
    root, relative = (
        ("robot_assets", "urdfs/g1/dex3_left.urdf")
        if kind == "robot"
        else ("context", "code/producer.py")
    )
    (Path(config[root]) / relative).write_text("changed")
    with pytest.raises(ValueError, match="closure differs"):
        validate(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("episode", 10),
        ("provided_meshes_read", True),
        ("reference_pose_labels_read", True),
    ],
)
def test_other_episode_and_reference_labels_cannot_be_used_silently(config, key, value):
    p = Path(config["bundle"]) / "manifest.json"
    metadata = json.loads(p.read_text())
    metadata[key] = value
    p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="matching label-free"):
        validate(config)


def test_robot_file_lock_cannot_escape_the_mount(config, tmp_path):
    outside = tmp_path / "external-secret"
    outside.write_text("not an asset")
    p = Path(config["robot_asset_lock"])
    lock = json.loads(p.read_text())
    lock["files_sha256"]["../external-secret"] = hashlib.sha256(
        outside.read_bytes()
    ).hexdigest()
    p.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="closure differs"):
        validate(config)


def test_nested_readonly_mount_target_must_exist_before_docker(config):
    (
        Path(config["context"])
        / "code/robotic_grounding/source/robotic_grounding/robotic_grounding/assets"
    ).rmdir()
    with pytest.raises(ValueError, match="mount target"):
        validate(config)
