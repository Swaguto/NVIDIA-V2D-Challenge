"""Verify the official kit pin and saved policy/input contract before GPU evaluation.

This is an offline compatibility check. Passing it is not a policy evaluation,
full-episode result, or submission acceptance. The official runner remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import zipfile


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metadata(path: Path) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated checkpoint header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(64 * 1024 * 1024, path.stat().st_size - 8):
            raise ValueError("Invalid checkpoint header length")
        header = json.loads(stream.read(length))
    return header["__metadata__"]


def verify_kit(kit: Path) -> tuple[dict, dict]:
    with zipfile.ZipFile(kit) as archive:
        names = archive.namelist()
        manifests = [n for n in names if n.endswith("/KIT_MANIFEST.json")]
        if len(manifests) != 1 or len(names) != len(set(names)):
            raise ValueError("Expected one kit manifest and unique members")
        prefix = manifests[0].removesuffix("KIT_MANIFEST.json")
        manifest = json.loads(archive.read(manifests[0]))
        for name, expected in manifest["files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe kit member")
            if hashlib.sha256(archive.read(prefix + name)).hexdigest() != expected:
                raise ValueError(f"Official kit member checksum mismatch: {name}")
        roster = json.loads(
            archive.read(prefix + "data/track_3_evaluation_objects.json")
        )["episodes"]
    return manifest, roster


def audit(kit: Path, runtime: Path, checkpoint: Path, inputs: Path) -> dict:
    manifest, roster = verify_kit(kit)
    pin = manifest["evaluator"]["commit"]
    runtime_pin = subprocess.check_output(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True
    ).strip()
    if pin != runtime_pin:
        raise ValueError("Runtime does not match the kit evaluator commit")
    evaluator = runtime / manifest["evaluator"]["source_path"]
    if sha(evaluator) != manifest["files"][manifest["evaluator"]["kit_path"]]:
        raise ValueError("Runtime evaluator differs from the official kit")
    saved = metadata(checkpoint)
    config = json.loads(saved["user.resolved_config_json"])
    schema = json.loads(saved["user.policy_schema_json"])
    source = json.loads((inputs / "manifest.json").read_text())
    if saved["user.dex3_input_manifest_sha256"] != sha(inputs / "manifest.json"):
        raise ValueError("Checkpoint is bound to different targets/assets")
    for relative, expected in source["files"].items():
        file = (inputs / relative).resolve()
        if not file.is_relative_to(inputs.resolve()) or sha(file) != expected:
            raise ValueError(f"Input member checksum/path mismatch: {relative}")
    episode = source["episode"]
    names = [o["name"] for o in source["objects"]]
    scored = str(episode) in roster
    if scored and names != roster[str(episode)]:
        raise ValueError("Input objects do not match the official scored slot order")
    return dict(
        schema="v2d.dex3-official-evaluation-contract.v1",
        kit_sha256=sha(kit),
        kit_members_verified=len(manifest["files"]),
        official_runtime_commit=pin,
        official_evaluator_sha256=sha(evaluator),
        official_batch_runner_sha256=manifest["files"][
            "examples/run_policy_evaluation.py"
        ],
        official_evaluation_entrypoint="examples/run_policy_evaluation.py",
        checkpoint_sha256=sha(checkpoint),
        input_manifest_sha256=sha(inputs / "manifest.json"),
        iteration=int(saved["user.iteration"]),
        environment_steps=int(saved["user.environment_steps"]),
        embodiment=config["embodiment"]["_target_"],
        action_dimension=schema["action"]["dimension"],
        observation_dimension=schema["observation"]["dimension"],
        episode=episode,
        object_slot_names=names,
        episode_in_scored_roster=scored,
        selected_frame_count=source["selected_frame_count"],
        source_frame_count=source["source_frame_count"],
        full_source_episode_covered=source["selected_frame_count"]
        == source["source_frame_count"],
        runtime_extension_required=True,
        static_contract_pass=True,
        official_batch_runner_executed=False,
        policy_task_success_verified=False,
        submission_ready=False,
        scope="Kit and saved configuration checks only. Run the untouched official batch runner in the extension image, then validate its trajectories and full scored roster with the official packer.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kit", "runtime", "checkpoint", "inputs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.kit, args.runtime, args.checkpoint, args.inputs)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
