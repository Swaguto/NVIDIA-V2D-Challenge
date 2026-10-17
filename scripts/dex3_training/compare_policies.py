"""Compare development evaluations only after checking their frozen contracts."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
from v2d_dex3_flashchord.reset_curriculum import (
    validate as validate_reset_curriculum,
    objective_weights,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(path):
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated checkpoint header")
        length = struct.unpack("<Q", raw)[0]
        if length > 16 * 1024 * 1024:
            raise ValueError("Checkpoint header exceeds limit")
        header = json.loads(stream.read(length))
    return header["__metadata__"]


def compare(inputs, checkpoints, reports, controls):
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = digest(manifest_path)
    cache = manifest.get("collision_cache")
    if not cache or not cache.get("files"):
        raise ValueError("Comparison requires frozen collision geometry")
    for name, sha in manifest["files"].items():
        path = inputs / name
        if not path.resolve().is_relative_to(inputs.resolve()) or digest(path) != sha:
            raise ValueError(f"Private input bytes differ: {name}")
    control = json.loads(controls.read_text())
    if control["input_manifest_sha256"] != manifest_sha or not all(
        control.get(key) is True
        for key in (
            "collision_cache_locked",
            "installed_dispatch_loads_dual_hand",
            "unknown_contact_reward_rejected",
            "implicit_resampling_rejected",
            "changed_manifest_rejected",
        )
    ):
        raise ValueError("Installed runtime controls do not match this pack")
    metas = [metadata(path) for path in checkpoints]
    for key in (
        "user.resolved_config_json",
        "user.dex3_runtime_patches_sha256",
        "user.dex3_image_context_sha256",
        "user.policy_schema_json",
    ):
        if not metas[0].get(key) or metas[0][key] != metas[1].get(key):
            raise ValueError(f"Checkpoint contracts differ: {key}")
    config = json.loads(metas[0]["user.resolved_config_json"])
    if (
        config["env"]["voc_scale"] != 0
        or config["voc_scale_override"] != 0
        or config["reset"]["voc_decay_steps"] != 0
    ):
        raise ValueError("Comparison requires unassisted policies")
    if config["curriculum"] is not None:
        validate_reset_curriculum(
            config["curriculum"], objective_weights(config["objective"])
        )
    for name in (
        "contact_wrench_support",
        "missed_contact",
        "unintended_contact",
        "force_closure",
    ):
        term = config["objective"][name]
        if term["enabled"] or term["weight"] != 0:
            raise ValueError("Unknown contact labels were used")
    rows = []
    world_counts = set()
    for checkpoint, report_path, meta in zip(checkpoints, reports, metas, strict=True):
        report = json.loads(report_path.read_text())
        metrics = report["metrics"]
        if (
            meta["user.dex3_input_manifest_sha256"] != manifest_sha
            or report["checkpoint_sha256"] != digest(checkpoint)
            or report["environment_steps"] != int(meta["user.environment_steps"])
            or report["step_count"] != manifest["selected_frame_count"]
            or report["reset_mode"] != "explicit"
            or report["settling_steps"] != 0
            or report["start_frame"] != 0
            or report["sim_control_fps"] != 20
            or report["non_finite_worlds"]
        ):
            raise ValueError("Evaluation receipt differs from the frozen contract")
        world_counts.add(report["world_count"])
        rows.append(
            dict(
                iteration=int(meta["user.iteration"]),
                environment_steps=report["environment_steps"],
                checkpoint_sha256=digest(checkpoint),
                report_sha256=digest(report_path),
                success_rate=report["reference_tracking"]["success_rate"],
                chord_sr=metrics["chord_sr"],
                mppe_cm=metrics["mppe_cm"],
                mean_add_m=metrics["mean_add_m"],
                object_position_error_per_body_m=metrics[
                    "object_position_error_per_body_m"
                ],
                world_count=report["world_count"],
                step_count=report["step_count"],
            )
        )
    if (
        len(world_counts) != 1
        or rows[0]["environment_steps"] != 0
        or rows[1]["environment_steps"] <= 0
    ):
        raise ValueError("Need equal evaluation coverage, initial then learned policy")
    return dict(
        schema="v2d.dex3-controlled-policy-comparison.v1",
        scope=(
            "full development episode with inferred robot targets and original source confidence masks; not an official scored roster"
            if manifest.get("full_episode_covered") is True
            and manifest["selected_frame_count"] == manifest.get("source_frame_count")
            else "selected development window; not full episode or official scored roster"
        ),
        comparison_source_sha256=digest(Path(__file__)),
        input_manifest_sha256=manifest_sha,
        runtime_controls_sha256=digest(controls),
        image_context_sha256=metas[0]["user.dex3_image_context_sha256"],
        frozen_collision_files_sha256={
            name: manifest["files"][name] for name in cache["files"]
        },
        same_saved_experiment_config=True,
        collision_geometry_identical=True,
        controlled_comparison=True,
        virtual_object_assistance=0,
        rows=rows,
        primary_success_improved=rows[1]["success_rate"] > rows[0]["success_rate"],
        bit_exact_training_reproducibility_claimed=False,
        checkpoint_compatibility_verified_with_installed_extension=True,
        unmodified_stock_runtime_compatibility_claimed=False,
        physical_task_accepted=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--learned-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-report", type=Path, required=True)
    parser.add_argument("--learned-report", type=Path, required=True)
    parser.add_argument("--runtime-controls", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.inputs,
        [args.initial_checkpoint, args.learned_checkpoint],
        [args.initial_report, args.learned_report],
        args.runtime_controls,
    )
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
