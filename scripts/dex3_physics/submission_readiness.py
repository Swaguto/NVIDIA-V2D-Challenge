"""Inventory Track 3 artifacts against a supplied official kit, without uploading."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


def inventory(
    kit: Path, checkpoints: dict, meshes: Path | None, results: Path | None
) -> dict:
    """Read the kit's roster, preserving object-slot order and split membership."""
    with zipfile.ZipFile(kit) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.endswith("/data/track_3_evaluation_objects.json")
        ]
        if len(names) != 1:
            raise ValueError("Expected exactly one official Track 3 roster")
        roster_bytes = archive.read(names[0])
        episodes = json.loads(roster_bytes)["episodes"]
    missing_checkpoints, missing_meshes, missing_rollouts, present_checkpoints = (
        [],
        [],
        [],
        [],
    )
    for episode, objects in sorted(episodes.items(), key=lambda item: int(item[0])):
        ep = int(episode)
        stem = f"episode_{ep:06d}"
        checkpoint = checkpoints.get(str(ep))
        if checkpoint and Path(checkpoint).expanduser().is_file():
            path = Path(checkpoint).expanduser()
            present_checkpoints.append(
                {
                    "episode": ep,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                }
            )
        else:
            missing_checkpoints.append(ep)
        for slot, name in enumerate(objects):
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
                raise ValueError("Unsafe object name in roster")
            found = meshes is not None and any(
                (meshes / stem / f"{name}{suffix}").is_file()
                for suffix in [".glb", ".obj", ".ply", ".stl", ".off"]
            )
            if not found:
                missing_meshes.append(
                    {"episode": ep, "object_slot": slot, "object": name}
                )
        if results is None or not (results / f"{stem}.parquet").is_file():
            missing_rollouts.append(ep)
    extra = sorted(set(checkpoints) - set(episodes))
    return {
        "schema": "v2d.track3-artifact-inventory.v1",
        "kit_archive_sha256": hashlib.sha256(kit.read_bytes()).hexdigest(),
        "kit_roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
        "scored_episodes": [int(key) for key in episodes],
        "required_mesh_pairs": sum(len(objects) for objects in episodes.values()),
        "checkpoint_template": {
            key: f"/path/to/checkpoints/episode_{int(key):06d}.safetensors" for key in episodes
        },
        "present_checkpoint_files": present_checkpoints,
        "missing_checkpoints": missing_checkpoints,
        "missing_meshes": missing_meshes,
        "missing_rollouts": missing_rollouts,
        "checkpoint_keys_outside_scored_roster": extra,
        "artifact_presence_pass": not (
            missing_checkpoints or missing_meshes or missing_rollouts or extra
        ),
        "checkpoint_evaluator_compatibility_verified": False,
        "official_packer_validation_performed": False,
        "submission_ready": False,
        "scope": "File presence only; run the official evaluator and combined packer for numerical/schema/mesh validation. No submission is sent.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kit", type=Path, required=True, help="Official submission-kit zip"
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        help="Official evaluator mapping: episode string -> checkpoint path",
    )
    parser.add_argument("--meshes", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = json.loads(args.checkpoints.read_text()) if args.checkpoints else {}
    if not isinstance(checkpoints, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in checkpoints.items()
    ):
        parser.error(
            "Checkpoint mapping must contain string episode keys and path values"
        )
    report = inventory(args.kit, checkpoints, args.meshes, args.results)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "scored_episodes",
                    "required_mesh_pairs",
                    "missing_checkpoints",
                    "missing_meshes",
                    "missing_rollouts",
                    "submission_ready",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
