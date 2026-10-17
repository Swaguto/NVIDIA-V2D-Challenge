import json
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from prepare_physics_probe import PROBE_KEY, prepare
from v2d_dex3_flashchord.resume_contract import validate


def fixture(path, fps=20):
    sim = dict(fps=fps, substeps=20, integrator="implicitfast")
    config = dict(sim=sim, env=dict(sim=sim), action=dict(effort_limit=0.76))
    header = {
        "__metadata__": dict(
            format="learner_state",
            version="1",
            **{
                "user.resolved_config_json": json.dumps(config),
                "user.environment_steps": "13107200",
            },
        ),
        "state.0000": dict(dtype="F32", shape=[2], data_offsets=[0, 8]),
    }
    text = json.dumps(header).encode()
    payload = struct.pack("<ff", 1.25, -2.5)
    path.write_bytes(struct.pack("<Q", len(text)) + text + payload)
    return header, payload, config


def test_evaluation_probe_preserves_every_tensor_and_original_training_provenance(
    tmp_path,
):
    source, out = tmp_path / "trained.safetensors", tmp_path / "probe.safetensors"
    original, payload, config = fixture(source)
    receipt = prepare(source, out, 40)
    data = out.read_bytes()
    size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : size + 8])
    meta = header["__metadata__"]
    assert data[8 + size :] == payload
    assert header["state.0000"] == original["state.0000"]
    changed = json.loads(meta["user.resolved_config_json"])
    changed["sim"]["substeps"] = 20
    changed["env"]["sim"]["substeps"] = 20
    assert changed == config
    assert json.loads(receipt["original_training_resolved_config_json"]) == config
    assert receipt["original_training_environment_steps"] == 13_107_200
    assert receipt["evaluation_physics_fps"] == 800
    assert receipt["evaluation_only"]
    assert "user." + PROBE_KEY in meta


def test_probe_cannot_change_the_control_timeline_or_overwrite_a_checkpoint(tmp_path):
    source = tmp_path / "source.safetensors"
    fixture(source, fps=100)
    with pytest.raises(ValueError, match="same20Hz"):
        prepare(source, tmp_path / "bad.safetensors", 40)
    fixture(source)
    with pytest.raises(ValueError, match="finer timestep"):
        prepare(source, tmp_path / "bad.safetensors", 10)
    with pytest.raises(FileExistsError):
        prepare(source, source, 40)


def test_deployment_probe_is_not_an_exact_training_resume(tmp_path):
    with pytest.raises(ValueError, match="evaluation-only"):
        validate({PROBE_KEY: "{}"}, {}, None, None, None)
