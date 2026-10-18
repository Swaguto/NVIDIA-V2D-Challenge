import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from replay_scored_episode import commands, preflight


def fixture(root):
    inputs = root / "inputs"
    inputs.mkdir()
    data = inputs / "targets.npz"
    data.write_bytes(b"locked target bytes")
    manifest = inputs / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                episode=3,
                full_episode_covered=True,
                source_observations_changed=False,
                contact_labels_fabricated=False,
                source_frame_indices=[0, 1],
                files={data.name: hashlib.sha256(data.read_bytes()).hexdigest()},
            )
        )
    )
    checkpoint = root / "policies/model_100.safetensors"
    checkpoint.parent.mkdir()
    metadata = dict(format="learner_state", version="1")
    metadata.update(
        {
            "user." + k: str(v)
            for k, v in dict(
                iteration=100,
                environment_steps=153600,
                dex3_input_manifest_sha256=hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
            ).items()
        }
    )
    header = json.dumps(dict(__metadata__=metadata)).encode()
    checkpoint.write_bytes(struct.pack("<Q", len(header)) + header)
    report = root / "expected.json"
    report.write_text(
        json.dumps(
            dict(
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                step_count=2,
                start_frame=0,
                world_count=8,
                non_finite_worlds=[],
            )
        )
    )
    receipt = root / "producer.json"
    receipt.write_text(
        json.dumps(
            dict(
                schema="v2d.scored-single-episode-policy.v1",
                episode=3,
                baseline_complete=True,
                training_image="example/image@sha256:" + "a" * 64,
            )
        )
    )
    kit = root / "kit"
    (kit / "examples").mkdir(parents=True)
    (kit / "examples/run_policy_evaluation.py").write_text("# official fixture\n")
    return inputs, checkpoint, report, receipt, kit, root / "fresh-output"


def test_preflight_never_creates_output_and_uses_readonly_offline_mounts(tmp_path):
    args = fixture(tmp_path)
    plan = preflight(*args)
    assert not args[-1].exists()
    stages, common = commands(plan)
    assert "--pull=never" in common
    assert common[common.index("--network") + 1] == "none"
    mounts = [common[i + 1] for i, value in enumerate(common) if value == "-v"]
    assert all(
        next(m for m in mounts if target in m).endswith(":ro")
        for target in (":/data", ":/checkpoint/model.safetensors", ":/kit")
    )
    assert [label for label, _, _ in stages] == ["official", "native", "audit"]
    assert plan["training_steps_added"] == 0


@pytest.mark.parametrize(
    "kind", ["targets", "checkpoint", "short_report", "mutable_image", "output_overlap"]
)
def test_changed_sources_or_invalid_replay_contract_fail_before_docker(tmp_path, kind):
    args = list(fixture(tmp_path))
    if kind == "targets":
        (args[0] / "targets.npz").write_bytes(b"changed")
    elif kind == "checkpoint":
        args[1].write_bytes(args[1].read_bytes() + b"changed")
    elif kind == "short_report":
        r = json.loads(args[2].read_text())
        r["step_count"] = 1
        args[2].write_text(json.dumps(r))
    elif kind == "mutable_image":
        r = json.loads(args[3].read_text())
        r["training_image"] = "image:latest"
        args[3].write_text(json.dumps(r))
    else:
        args[-1] = args[0] / "output"
    with pytest.raises(ValueError):
        preflight(*args)
    assert not args[-1].exists()
