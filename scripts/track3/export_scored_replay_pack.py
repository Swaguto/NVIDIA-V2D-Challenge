"""Create a private portable frozen-policy replay folder, without footage/weights.

Copies only the verified frozen input closure, checkpoint and producer/evaluator
receipts. Does not upload, publish, provision compute or run Docker.
"""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from replay_scored_episode import preflight


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(inputs, checkpoint, expected_report, policy_receipt, kit, out, code_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("Use the complete committed replay source SHA")
    inputs, checkpoint, expected_report, policy_receipt, kit, out = (
        Path(p).resolve()
        for p in (inputs, checkpoint, expected_report, policy_receipt, kit, out)
    )
    plan = preflight(inputs, checkpoint, expected_report, policy_receipt, kit, out)
    manifest = json.loads((inputs / "manifest.json").read_text())
    standalone = {
        "targets.npz",
        "source-confidence.npz",
        "reference.parquet",
        "support.usda",
        "collision-cache-receipt.json",
        "command-proposal-report.json",
    }
    for name in manifest["files"]:
        path = Path(name)
        geometry = path.parts[0] in {"objects", "robot"} and path.suffix.lower() in {
            ".obj",
            ".urdf",
            ".stl",
            ".ply",
            ".glb",
        }
        collision = name.startswith("collision-cache/convex/") and path.suffix == ".npz"
        if name not in standalone and not geometry and not collision:
            raise ValueError("Private replay export refuses non-replay data: " + name)
    out.mkdir(parents=True, exist_ok=False)
    copies = [(inputs / "manifest.json", "inputs/manifest.json")]
    copies += [(inputs / name, "inputs/" + name) for name in manifest["files"]]
    checkpoint_name = f"checkpoint/model_{plan['iteration']}.safetensors"
    copies += [
        (checkpoint, checkpoint_name),
        (expected_report, "expected-official.json"),
        (policy_receipt, "producer-receipt.json"),
    ]
    files = {}
    for source, relative in copies:
        target = out / relative
        if not target.resolve().is_relative_to(out):
            raise ValueError("Export path escaped private pack")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(source) != sha(target):
            raise ValueError("Private copy differs from original bytes")
        files[relative] = dict(size=target.stat().st_size, sha256=sha(target))
    recipe = dict(
        schema="v2d.private-scored-replay-recipe.v1",
        episode=plan["episode"],
        code_commit=code_commit,
        image=plan["image"],
        inputs="inputs",
        checkpoint=checkpoint_name,
        expected_report="expected-official.json",
        policy_receipt="producer-receipt.json",
        source_frames=plan["source_frames"],
        world_count=8,
        private_pack=True,
        pack_published=False,
        footage_included=False,
        inference_model_weights_included=False,
        credentials_included=False,
        learned_policy_checkpoint_included=True,
        training_steps_added=0,
        physical_task_accepted=False,
        independent_teammate_reproduction_verified=False,
    )
    (out / "replay-recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    readme = (
        "PRIVATE SCORED POLICY REPLAY PACK\n\n"
        f"Episode {plan['episode']}, {plan['source_frames']} source frames, eight evaluation worlds.\n"
        f"Use repository code commit {code_commit}.\n"
        "Obtain the unchanged official submission kit separately.\n"
        "Read docs/team/dex3-isaac-review/SCORED_POLICY_REPLAY.md in that checkout.\n"
        "Authenticate to private GHCR and explicitly pull the image in replay-recipe.json.\n"
        "Pass the recipe's relative file paths as absolute paths to replay_scored_episode.py.\n"
        "Default preflight performs no Docker/GPU work; --execute runs the repeat on your GPU host.\n"
        "Keep the input files read-only and choose a fresh output outside the input/checkpoint directories.\n"
        "Back up results and stop the GPU instance whenever idle.\n\n"
        "This pack reproduces an archived experiment, including its failed grasp.\n"
        "It is not a successful task or a complete reconstruction/training reproduction.\n"
        "Small floating-point/simulation differences can remain. No footage, inference model\n"
        "weights or credentials are intentionally included. Keep this pack private.\n"
    )
    (out / "README.txt").write_text(readme)
    for name in ("replay-recipe.json", "README.txt"):
        path = out / name
        files[name] = dict(size=path.stat().st_size, sha256=sha(path))
    # Recheck the relocated pack using its own copied metadata and input bytes.
    relocated = preflight(
        out / "inputs",
        out / checkpoint_name,
        out / "expected-official.json",
        out / "producer-receipt.json",
        kit,
        out / "fresh-replay-output",
    )
    if (
        relocated["input_manifest_sha256"] != plan["input_manifest_sha256"]
        or relocated["checkpoint_sha256"] != plan["checkpoint_sha256"]
    ):
        raise ValueError("Portable replay contract changed during export")
    receipt = dict(
        schema="v2d.private-scored-replay-pack.v1",
        episode=plan["episode"],
        source_input_manifest_sha256=plan["input_manifest_sha256"],
        checkpoint_sha256=plan["checkpoint_sha256"],
        code_commit=code_commit,
        relocated_preflight_passed=True,
        private_pack=True,
        published=False,
        files=files,
        total_bytes=sum(item["size"] for item in files.values()),
    )
    (out / "pack-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "inputs",
        "checkpoint",
        "expected-report",
        "policy-receipt",
        "kit",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    receipt = export(
        args.inputs,
        args.checkpoint,
        args.expected_report,
        args.policy_receipt,
        args.kit,
        args.out,
        args.code_commit,
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2))
