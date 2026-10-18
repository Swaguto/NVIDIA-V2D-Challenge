"""Reject altered private inputs and unbounded/mutable policy runs before Docker."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from run_scored_policy import (
    TRAINING,
    collision_seed_manifest,
    require_matching_collision_geometry,
    validate,
    verify_reused_collision_cache,
)


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
    for name in (
        "check_native_scene.py",
        "audit_initial_contacts.py",
        "audit_reference_contacts.py",
        "audit_hand_reference_contacts.py",
    ):
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


def test_old_context_cannot_silently_omit_full_hand_contact_diagnostics(config):
    p = Path(config["context"]) / "context-manifest.json"
    source = json.loads(p.read_text())
    del source["files"]["code/scripts/dex3_training/audit_hand_reference_contacts.py"]
    p.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="native contact diagnostic"):
        validate(config)


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


def write_collision_pack(
    root, *, frozen=True, object_mesh="same geometry", targets="commands"
):
    root.mkdir()
    contents = {
        "objects/pot.obj": object_mesh,
        "objects/pot.urdf": "same mass and collision material",
        "robot/hand.urdf": "same robot",
        "support.usda": "same table",
        "targets.npz": targets,
    }
    if frozen:
        contents["collision-cache/convex/example.npz"] = "same frozen hull bytes"
    files = {}
    for name, content in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"files": files, "objects": [{"name": "pot", "mesh": "objects/pot.obj"}]}
    if frozen:
        manifest["collision_cache"] = {
            "directory": "collision-cache",
            "runtime_commit": "same-pinned-runtime",
            "files": ["collision-cache/convex/example.npz"],
        }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_paired_anchor_commands_can_reuse_identical_physics_geometry(tmp_path):
    seed, target = tmp_path / "seed", tmp_path / "candidate"
    write_collision_pack(seed)
    write_collision_pack(target, targets="changed robot anchor commands")
    require_matching_collision_geometry(seed, target)
    proof = verify_reused_collision_cache(seed, target)
    assert proof["collision_cache_bytes_exact"] is True
    assert proof["native_preflight_and_contact_audits_still_required"] is True


def test_different_objects_cannot_reuse_a_control_collision_cache(tmp_path):
    seed, target = tmp_path / "seed", tmp_path / "candidate"
    write_collision_pack(seed)
    write_collision_pack(target, frozen=False, object_mesh="different geometry")
    with pytest.raises(ValueError, match="identical ordered"):
        require_matching_collision_geometry(seed, target)


def test_modified_or_undeclared_collision_seed_is_rejected(tmp_path):
    seed = tmp_path / "seed"
    write_collision_pack(seed)
    extra = seed / "collision-cache/convex/extra.npz"
    extra.write_text("unbound hulls")
    with pytest.raises(ValueError, match="seed closure"):
        collision_seed_manifest(seed)
    extra.unlink()
    (seed / "collision-cache/convex/example.npz").write_text("altered hulls")
    with pytest.raises(ValueError, match="closure differs"):
        collision_seed_manifest(seed)


def test_regenerated_hulls_fail_even_when_surface_assets_match(tmp_path):
    seed, target = tmp_path / "seed", tmp_path / "candidate"
    write_collision_pack(seed)
    manifest = write_collision_pack(target)
    name = "collision-cache/convex/example.npz"
    (target / name).write_text("different regenerated hulls")
    manifest["files"][name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="paired control"):
        verify_reused_collision_cache(seed, target)
