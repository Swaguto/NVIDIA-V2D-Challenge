import hashlib
import json
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
