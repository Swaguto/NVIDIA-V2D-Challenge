"""Freeze the actual upstream scene's collision hulls into a new private pack.

Run once on CPU before training. Subsequent training/evaluation refuses missing,
modified, or undeclared cache files. Source observations and robot targets stay
byte-identical; only the explicit collision-geometry closure is added.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

import numpy as np
import warp as wp

from flash_chord.assets import mesh as collision_mesh
from flash_chord.scene.builder import build_scene
from flash_chord.scene.collision import CollisionPolicy
from v2d_dex3_flashchord import RUNTIME_COMMIT
from v2d_dex3_flashchord.embodiment import Dex3Hands, Dex3HandsConfig
from v2d_dex3_flashchord.reference import load_dex3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(inputs, output, seed_cache=None):
    inputs, output = inputs.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(inputs):
        raise ValueError("Use a new output directory outside the source pack")
    source_sha = sha(inputs / "manifest.json")
    source_manifest = json.loads((inputs / "manifest.json").read_text())
    for relative, expected in source_manifest["files"].items():
        file = (inputs / relative).resolve()
        if not file.is_relative_to(inputs) or sha(file) != expected:
            raise ValueError("Source input hash/path mismatch")
    output.mkdir(parents=True)
    for relative in source_manifest["files"]:
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inputs / relative, dest)
    shutil.copy2(inputs / "manifest.json", output / "manifest.json")
    cache_root = output / "collision-cache"
    if cache_root.exists():
        raise ValueError("Freeze from the original unbound preparation pack")
    (cache_root / "convex").mkdir(parents=True)
    if seed_cache:
        for file in (seed_cache / "convex").glob("*.npz"):
            shutil.copy2(file, cache_root / "convex" / file.name)
    reference = load_dex3(output / "reference.parquet", require_frozen_collisions=False)
    os.environ["FLASH_CHORD_CACHE_DIR"] = str(cache_root)
    embodiment = Dex3Hands(
        Dex3HandsConfig(
            left_urdf_path=output / "robot/urdfs/g1/dex3_left.urdf",
            right_urdf_path=output / "robot/urdfs/g1/dex3_right.urdf",
            reference_manifest_sha256=source_sha,
        )
    )
    used = set()
    original_load, original_save = (
        collision_mesh._load_cache,
        collision_mesh._save_cache,
    )

    def load(path):
        used.add(Path(path).resolve())
        return original_load(path)

    def save(path, hulls):
        used.add(Path(path).resolve())
        return original_save(path, hulls)

    collision_mesh._load_cache, collision_mesh._save_cache = load, save
    try:
        with wp.ScopedDevice("cpu"):
            scene = build_scene(
                embodiment,
                reference,
                world_count=1,
                support_usda=output / "support.usda",
                collision=CollisionPolicy(ground=False),
                decompose_objects=True,
                contact_friction=1.0,
            )
    finally:
        collision_mesh._load_cache, collision_mesh._save_cache = (
            original_load,
            original_save,
        )
    if not used or any(not f.is_relative_to(cache_root) for f in used):
        raise ValueError("Scene did not produce the expected local collision cache")
    for file in cache_root.rglob("*.npz"):
        if file.resolve() not in used:
            file.unlink()  # Only an unused copy in the newly created output pack.
    cache_files = []
    for file in sorted(used):
        with np.load(file, allow_pickle=False) as data:
            cache_files.append(
                dict(
                    path=file.relative_to(output).as_posix(),
                    sha256=sha(file),
                    convex_hulls=int(data["n"]),
                )
            )
    receipt = dict(
        schema="v2d.dex3-frozen-collision-geometry.v1",
        source_reference_manifest_sha256=source_sha,
        runtime_commit=RUNTIME_COMMIT,
        producer_sha256=sha(Path(__file__)),
        cache_files=cache_files,
        scene_body_count=int(scene.model.body_count),
        scene_shape_count=int(scene.model.shape_count),
        library_versions={
            name: importlib.metadata.version(name)
            for name in ["newton", "coacd", "numpy", "trimesh"]
        },
        source_observations_or_robot_targets_changed=False,
        physical_task_accepted=False,
    )
    (output / "collision-cache-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    manifest = json.loads((output / "manifest.json").read_text())
    # Freezing geometry must preserve the full-task command/confidence contract.
    # Downgrading v3/v4 to a confidence window rejects legitimate unknown source
    # frames and removes the loader's inferred-target validation.
    if manifest["schema"] == "v2d.dex3-training-window.v1":
        manifest["schema"] = "v2d.dex3-training-window.v2"
    manifest["collision_cache"] = dict(
        directory="collision-cache",
        runtime_commit=RUNTIME_COMMIT,
        files=[r["path"] for r in cache_files],
    )
    manifest["files"] = {
        p.relative_to(output).as_posix(): sha(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p != output / "manifest.json"
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    load_dex3(output / "reference.parquet")
    return dict(
        input_manifest_sha256=sha(output / "manifest.json"),
        cache_files=cache_files,
        physical_task_accepted=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed-cache", type=Path)
    args = p.parse_args()
    print(json.dumps(freeze(args.inputs, args.out, args.seed_cache), indent=2))
