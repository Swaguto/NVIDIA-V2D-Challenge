"""Fork deployment metadata for a finer-timestep evaluation, preserving all tensors.

The output is marked evaluation-only and is rejected by the training resume guard.
Original training configuration and counters remain recorded in its provenance.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct


PROBE_KEY = "dex3_evaluation_physics_probe_json"


def prepare(source, out, substeps):
    data = source.read_bytes()
    if len(data) < 8:
        raise ValueError("Truncated checkpoint")
    size = struct.unpack("<Q", data[:8])[0]
    if not 0 < size <= 16 * 1024 * 1024 or size + 8 > len(data):
        raise ValueError("Invalid checkpoint header size")
    header = json.loads(data[8 : size + 8])
    meta = header["__metadata__"]
    if meta.get("format") != "learner_state" or meta.get("version") != "1":
        raise ValueError("Unsupported checkpoint format")
    if "user." + PROBE_KEY in meta:
        raise ValueError("Fork from the original trained checkpoint")
    original = json.loads(meta["user.resolved_config_json"])
    config = deepcopy(original)
    old = config["sim"]["substeps"]
    if (
        config["sim"] != config["env"]["sim"]
        or config["sim"]["fps"] != 20
        or not isinstance(substeps, int)
        or substeps not in (2 * old, 4 * old)
    ):
        raise ValueError(
            "Only a twofold or fourfold finer timestep at the same20Hz is allowed"
        )
    config["sim"]["substeps"] = substeps
    config["env"]["sim"]["substeps"] = substeps
    payload = data[8 + size :]
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        lo, hi = tensor["data_offsets"]
        if not 0 <= lo <= hi <= len(payload):
            raise ValueError("Invalid tensor data range")
    receipt = dict(
        schema="v2d.evaluation-physics-probe.v1",
        source_checkpoint_sha256=hashlib.sha256(data).hexdigest(),
        original_training_resolved_config_json=meta["user.resolved_config_json"],
        original_training_environment_steps=int(meta["user.environment_steps"]),
        original_training_physics_fps=20 * old,
        evaluation_physics_fps=20 * substeps,
        control_fps=20,
        all_tensor_bytes_unchanged=True,
        tensor_payload_sha256=hashlib.sha256(payload).hexdigest(),
        evaluation_only=True,
        physical_task_accepted=False,
    )
    meta["user.resolved_config_json"] = json.dumps(config, sort_keys=True)
    meta["user." + PROBE_KEY] = json.dumps(receipt, sort_keys=True)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with out.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.write(payload)
    receipt["probe_checkpoint_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--substeps", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.checkpoint, args.out, args.substeps)
    with args.receipt.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
