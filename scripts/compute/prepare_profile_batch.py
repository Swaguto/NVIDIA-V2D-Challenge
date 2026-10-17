"""Prepare two identical policy readbacks to measure repeated runtime cost offline."""

import argparse
import json
from pathlib import Path

from evaluation_gate import metadata, sha256
from run_batch import preflight


def prepare(inputs, checkpoint, context, outputs, image):
    inputs, checkpoint, context, outputs = (
        Path(p).resolve() for p in (inputs, checkpoint, context, outputs)
    )
    saved = metadata(checkpoint)
    if saved["user.dex3_input_manifest_sha256"] != sha256(inputs / "manifest.json"):
        raise ValueError("Checkpoint uses different private inputs")
    if saved["user.dex3_image_context_sha256"] != sha256(context):
        raise ValueError("Checkpoint uses a different image context")
    cfg = json.loads(saved["user.resolved_config_json"])
    if cfg["env"]["sim"]["fps"] != 20:
        raise ValueError("Expected the recorded 20 Hz control configuration")
    checkpoint_mount = "/profile-checkpoint/" + checkpoint.name
    plan = {
        "schema": "v2d.compute-batch.v1",
        "image": image,
        "image_context_sha256": sha256(context),
        "inputs": str(inputs),
        "outputs": str(outputs),
        "input_manifest_sha256": sha256(inputs / "manifest.json"),
        "cache_volume": "dex3-training-cache",
        "max_wall_seconds": 330,
        "mounts": [
            {
                "source": str(checkpoint.parent),
                "target": "/profile-checkpoint",
                "read_only": True,
            },
            {
                "source": str(context.parent),
                "target": "/profile-context",
                "read_only": True,
            },
        ],
        "required_files": [
            {"path": checkpoint_mount, "sha256": sha256(checkpoint)},
            {"path": "/profile-context/" + context.name, "sha256": sha256(context)},
        ],
        "stages": [
            {
                "name": name,
                "kind": "container",
                "purpose": "evaluate",
                "timeout_seconds": 150,
                "script": "/opt/dex3-extension/record_policy_readback.py",
                "args": [
                    "--checkpoint",
                    checkpoint_mount,
                    "--out",
                    "/outputs/" + name,
                    "--world-count",
                    "4",
                ],
            }
            for name in ("readback-first", "readback-repeat")
        ],
        "new_training_steps": 0,
        "purpose": "Compare repeated identical evaluation wall time, sampled memory and GPU utilization. Cache hit and total billed-cost savings need GPU/provider validation.",
    }
    preflight(plan)
    return plan


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "checkpoint", "context", "outputs", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--image", required=True)
    a = p.parse_args()
    plan = prepare(a.inputs, a.checkpoint, a.context, a.outputs, a.image)
    with a.out.open("x") as stream:
        stream.write(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(preflight(plan), indent=2))
