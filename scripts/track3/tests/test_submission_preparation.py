import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import prepare_scored_submission as submission


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def case(tmp_path):
    bundle, policy = tmp_path / "bundle", tmp_path / "policy"
    write(
        bundle / "manifest.json",
        {
            "episode": 1,
            "units": "metres",
            "provided_meshes_read": False,
            "reference_pose_labels_read": False,
            "objects": [{"name": "pot"}],
        },
    )
    mesh = policy / "frozen/objects/pot.obj"
    mesh.parent.mkdir(parents=True)
    # A deliberately offset body origin must survive staging unchanged.
    mesh.write_bytes(
        b"v 0.123 0.234 0.345\nv 0.14 0.25 0.36\nv 0.15 0.27 0.38\nf 1 2 3\n"
    )
    write(
        policy / "frozen/manifest.json",
        {
            "episode": 1,
            "objects": [{"name": "pot", "mesh": "objects/pot.obj"}],
            "full_episode_covered": True,
            "source_frame_indices": [0, 1],
            "source_observations_changed": False,
            "contact_labels_fabricated": False,
            "command_selected_anchor_slots": [1, 0],
            "files": {"objects/pot.obj": submission.sha(mesh)},
        },
    )
    write(
        policy / "receipt.json",
        {
            "episode": 1,
            "baseline_complete": True,
            "training": {"iterations": 100},
            "bundle_manifest_sha256": submission.sha(bundle / "manifest.json"),
        },
    )
    checkpoint = policy / "ppo/model_100.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"opaque checkpoint bytes")
    report = {
        "checkpoint_sha256": submission.sha(checkpoint),
        "step_count": 2,
        "start_frame": 0,
        "non_finite_worlds": [],
    }
    write(policy / "official-100/episode_000001.json", report)
    write(policy / "official-0/episode_000001.json", {})
    (policy / "official-100/episode_000001.parquet").write_bytes(
        b"opaque trajectory with source metadata"
    )
    return {
        "episode": 1,
        "iteration": 100,
        "policy_root": str(policy),
        "bundle": str(bundle),
    }


def test_rejects_modified_geometry_and_checkpoint(tmp_path):
    row = case(tmp_path)
    submission.inspect_episode(row, ["pot"], 2)
    mesh = Path(row["policy_root"]) / "frozen/objects/pot.obj"
    original = mesh.read_bytes()
    mesh.write_bytes(original + b"# changed body frame\n")
    with pytest.raises(ValueError, match="closure differs"):
        submission.inspect_episode(row, ["pot"], 2)
    mesh.write_bytes(original)
    (Path(row["policy_root"]) / "ppo/model_100.safetensors").write_bytes(
        b"different checkpoint"
    )
    with pytest.raises(ValueError, match="coverage/checkpoint"):
        submission.inspect_episode(row, ["pot"], 2)


def test_complete_roster_required_before_staging(tmp_path, monkeypatch):
    kit = tmp_path / "kit"
    write(
        kit / "data/track_3_evaluation_objects.json",
        {"episodes": {"1": ["pot"], "3": ["spoon"]}},
    )
    catalog = tmp_path / "catalog.json"
    write(
        catalog,
        {"schema": "v2d.scored-submission-catalog.v1", "episodes": [case(tmp_path)]},
    )
    monkeypatch.setattr(submission, "official_frames", lambda *args: {1: 2, 3: 2})
    out = tmp_path / "staged"
    with pytest.raises(ValueError, match="exact complete"):
        submission.prepare(
            kit, catalog, out, "https://github.com/owner/repo/commit/" + "a" * 40
        )
    assert not out.exists()


def test_staging_preserves_bytes_and_uses_official_packer(tmp_path, monkeypatch):
    row = case(tmp_path)
    kit = tmp_path / "kit"
    write(kit / "data/track_3_evaluation_objects.json", {"episodes": {"1": ["pot"]}})
    packer = kit / "tools/pack_submission.py"
    packer.parent.mkdir()
    packer.write_text("# untouched official tool fixture\n")
    catalog = tmp_path / "catalog.json"
    write(catalog, {"schema": "v2d.scored-submission-catalog.v1", "episodes": [row]})
    monkeypatch.setattr(submission, "official_frames", lambda *args: {1: 2})
    calls = []

    def pack(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"official packed result")
        return __import__("subprocess").CompletedProcess(command, 0)

    monkeypatch.setattr(submission.subprocess, "run", pack)
    out = tmp_path / "staged"
    receipt = submission.prepare(
        kit, catalog, out, "https://github.com/owner/repo/commit/" + "a" * 40
    )
    assert receipt["official_packer_passed"] is True
    assert receipt["submission_uploaded"] is False
    assert receipt["captain_approval_required"] is True
    assert "--skip-crosscheck" not in calls[0]
    assert calls[0][1] == str(packer)
    assert (out / "meshes/episode_000001/pot.obj").read_bytes() == (
        Path(row["policy_root"]) / "frozen/objects/pot.obj"
    ).read_bytes()
    assert (
        out / "results/episode_000001.parquet"
    ).read_bytes() == b"opaque trajectory with source metadata"
    assert (
        receipt["submission_sha256"]
        == hashlib.sha256(b"official packed result").hexdigest()
    )


def legacy_case(tmp_path):
    row = case(tmp_path)
    policy, bundle = Path(row["policy_root"]), Path(row["bundle"])
    (bundle / "episode.npz").write_bytes(b"original episode array bytes")
    frozen = policy / "frozen"
    manifest = json.loads((frozen / "manifest.json").read_text())
    manifest.update(
        command_anchor_policy="visual",
        source_files_sha256={
            "bundle/episode.npz": submission.sha(bundle / "episode.npz")
        },
    )
    write(frozen / "manifest.json", manifest)
    input_hash = submission.sha(frozen / "manifest.json")
    for n in [0, 100]:
        metadata = {
            "format": "learner_state",
            "version": "1",
            "user.iteration": str(n),
            "user.environment_steps": str(n * 1536),
            "user.dex3_input_manifest_sha256": input_hash,
            "user.resolved_config_json": json.dumps(
                {"training": {"world_count": 64, "rollout_steps": 24}}
            ),
        }
        header = json.dumps({"__metadata__": metadata}).encode()
        checkpoint = policy / f"ppo/model_{n}.safetensors"
        checkpoint.write_bytes(struct.pack("<Q", len(header)) + header)
        write(
            policy / f"official-{n}/episode_000001.json",
            {
                "checkpoint_sha256": submission.sha(checkpoint),
                "step_count": 2,
                "start_frame": 0,
                "non_finite_worlds": [],
            },
        )
    original = tmp_path / "original-comparison.json"
    write(
        original,
        {
            "schema": "v2d.scored-sam3d-anchor-comparison.v1",
            "episodes": [1],
            "source_commit": "a" * 40,
            "image": "example/image@sha256:" + "b" * 64,
            "training": {
                "visual-e001": {"iterations": 100, "environment_steps": 153600}
            },
        },
    )
    readback = tmp_path / "recovered-readback.json"
    write(
        readback,
        {
            "schema": "v2d.dex3-full-task-policy-readback.v1",
            "learned_policy": True,
            "checkpoint_sha256": submission.sha(policy / "ppo/model_100.safetensors"),
            "source_manifest_sha256": input_hash,
            "frame_count": 2,
            "source_frame_count": 2,
            "zero_assistance_control_steps_verified": 2,
            "virtual_object_assistance": 0.0,
            "virtual_object_actuator_caps_zero": True,
            "evaluation_reset_stabilization_steps": 0,
            "source_motion_speed": 1.0,
        },
    )
    row["legacy_sources"] = {
        "comparison_receipt": str(original),
        "frozen": str(frozen),
        "evaluation": str(policy),
        "readback_report": str(readback),
    }
    # No synthetic normalized receipt is saved for this original layout.
    (policy / "receipt.json").unlink()
    return row


def test_original_layout_is_bound_without_rewriting_outputs(tmp_path):
    row = legacy_case(tmp_path)
    report = Path(row["policy_root"]) / "official-100/episode_000001.json"
    original = report.read_bytes()
    checked = submission.inspect_episode(row, ["pot"], 2)
    assert (
        checked["legacy_source_verification"]["frozen_inputs_and_original_bundle_bound"]
        is True
    )
    assert report.read_bytes() == original
    assert not (Path(row["policy_root"]) / "receipt.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"virtual_object_assistance": 1.0},
        {"zero_assistance_control_steps_verified": 1},
        {"source_manifest_sha256": "wrong"},
    ],
)
def test_legacy_adapter_cannot_bypass_unassisted_or_input_binding(tmp_path, change):
    row = legacy_case(tmp_path)
    p = Path(row["legacy_sources"]["readback_report"])
    native = json.loads(p.read_text())
    write(p, dict(native, **change))
    with pytest.raises(ValueError, match="unassisted evaluation binding"):
        submission.inspect_episode(row, ["pot"], 2)
