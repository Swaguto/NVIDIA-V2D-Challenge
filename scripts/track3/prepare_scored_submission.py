"""Stage complete, unchanged policy outputs and run the untouched Track 3 packer.

The explicit catalog chooses checkpoints before packing. This tool never selects
by score, edits poses, aligns meshes, uploads files, or accepts physical tasks.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from audit_scored_coverage import official_frames
from run_scored_policy import checkpoint_metadata


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked(root, name, digest):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path) != digest:
        raise ValueError("Frozen input closure differs: " + name)
    return path


def inspect_legacy_source(row, policy, bundle, frames):
    """Adapt the original comparison layout by checking its actual evidence.

    No receipt or evaluator output is rewritten. Recovery evaluations can live
    elsewhere; their checkpoint/input bindings must still match exactly.
    """
    source = row["legacy_sources"]
    original = Path(source["comparison_receipt"])
    receipt = json.loads(original.read_text())
    ep, iteration = row["episode"], row["iteration"]
    frozen = Path(source["frozen"])
    manifest = json.loads((frozen / "manifest.json").read_text())
    label = f"{manifest['command_anchor_policy']}-e{ep:03d}"
    checkpoint = policy / f"ppo/model_{iteration}.safetensors"
    metadata = checkpoint_metadata(checkpoint)
    initial = checkpoint_metadata(policy / "ppo/model_0.safetensors")
    resolved = json.loads(metadata["resolved_config_json"])
    steps = (
        iteration
        * resolved["training"]["world_count"]
        * resolved["training"]["rollout_steps"]
    )
    evaluation = Path(source["evaluation"])
    readback = Path(source["readback_report"])
    native = json.loads(readback.read_text())
    if (
        receipt.get("schema") != "v2d.scored-sam3d-anchor-comparison.v1"
        or ep not in receipt.get("episodes", [])
        or not re.fullmatch(r"[0-9a-f]{40}", receipt.get("source_commit", ""))
        or not re.fullmatch(
            r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}", receipt.get("image", "")
        )
        or receipt.get("training", {}).get(label)
        != {"iterations": iteration, "environment_steps": steps}
        or int(metadata["iteration"]) != iteration
        or int(metadata["environment_steps"]) != steps
        or int(initial["iteration"]) != 0
        or int(initial["environment_steps"]) != 0
        or metadata["dex3_input_manifest_sha256"] != sha(frozen / "manifest.json")
        or initial["dex3_input_manifest_sha256"]
        != metadata["dex3_input_manifest_sha256"]
        or manifest.get("source_files_sha256", {}).get("bundle/episode.npz")
        != sha(bundle / "episode.npz")
        or native.get("schema") != "v2d.dex3-full-task-policy-readback.v1"
        or native.get("learned_policy") is not True
        or native.get("checkpoint_sha256") != sha(checkpoint)
        or native.get("source_manifest_sha256") != sha(frozen / "manifest.json")
        or native.get("frame_count") != frames
        or native.get("source_frame_count") != frames
        or native.get("zero_assistance_control_steps_verified") != frames
        or native.get("virtual_object_assistance") != 0.0
        or native.get("virtual_object_actuator_caps_zero") is not True
        or native.get("evaluation_reset_stabilization_steps") != 0
        or native.get("source_motion_speed") != 1.0
    ):
        raise ValueError(
            "Legacy baseline lost its checkpoint, source or unassisted evaluation binding"
        )
    first = json.loads(
        (evaluation / "official-0" / f"episode_{ep:06d}.json").read_text()
    )
    if (
        first.get("checkpoint_sha256") != sha(policy / "ppo/model_0.safetensors")
        or first.get("step_count") != frames
        or first.get("start_frame") != 0
        or first.get("non_finite_worlds")
    ):
        raise ValueError("Legacy initial-policy control is incomplete")
    # An in-memory normalized view, never a fabricated/replaced producer receipt.
    normalized = {
        "baseline_complete": True,
        "episode": ep,
        "training": {"iterations": iteration},
        "bundle_manifest_sha256": sha(bundle / "manifest.json"),
    }
    proof = {
        "artifact_origin": "verified original comparison layout; no retraining or output edits",
        "comparison_receipt_sha256": sha(original),
        "readback_report_sha256": sha(readback),
        "preparation_source_commit": receipt["source_commit"],
        "training_image": receipt["image"],
        "checkpoint_and_initial_control_verified": True,
        "frozen_inputs_and_original_bundle_bound": True,
        "full_unassisted_native_readback_verified": True,
    }
    return normalized, frozen, evaluation, proof


def inspect_episode(row, names, frames):
    ep = row["episode"]
    iteration = row["iteration"]
    if type(ep) is not int or type(iteration) is not int or iteration < 1:
        raise ValueError("Catalog needs explicit integer episodes and checkpoints")
    policy, bundle = Path(row["policy_root"]), Path(row["bundle"])
    origin = None
    if "legacy_sources" in row:
        receipt, frozen, evaluation, origin = inspect_legacy_source(
            row, policy, bundle, frames
        )
    else:
        receipt = json.loads((policy / "receipt.json").read_text())
        frozen, evaluation = policy / "frozen", policy
    provenance = json.loads((bundle / "manifest.json").read_text())
    manifest = json.loads((frozen / "manifest.json").read_text())
    if (
        receipt.get("baseline_complete") is not True
        or receipt["episode"] != ep
        or receipt["training"]["iterations"] != iteration
        or receipt["bundle_manifest_sha256"] != sha(bundle / "manifest.json")
        or provenance["episode"] != ep
        or provenance["units"] != "metres"
        or provenance.get("provided_meshes_read") is not False
        or provenance.get("reference_pose_labels_read") is not False
        or [o["name"] for o in provenance["objects"]] != names
        or manifest["episode"] != ep
        or [o["name"] for o in manifest["objects"]] != names
        or manifest.get("full_episode_covered") is not True
        or manifest["source_frame_indices"] != list(range(frames))
        or manifest.get("source_observations_changed") is not False
        or manifest.get("contact_labels_fabricated") is not False
        or sum(manifest.get("command_selected_anchor_slots", [0, 0])) == 0
    ):
        raise ValueError("Incomplete or mismatched own-geometry baseline")
    for name, digest in manifest["files"].items():
        locked(frozen, name, digest)
    meshes = []
    for obj in manifest["objects"]:
        path = locked(frozen, obj["mesh"], manifest["files"][obj["mesh"]])
        if path.suffix.lower() not in {".obj", ".glb", ".ply", ".stl"}:
            raise ValueError("Unsupported reconstructed mesh format")
        meshes.append((obj["name"], path))
    checkpoint = policy / f"ppo/model_{iteration}.safetensors"
    final = evaluation / f"official-{iteration}"
    stem = f"episode_{ep:06d}"
    report = json.loads((final / (stem + ".json")).read_text())
    if (
        report["checkpoint_sha256"] != sha(checkpoint)
        or report["step_count"] != frames
        or report["start_frame"] != 0
        or report.get("non_finite_worlds")
    ):
        raise ValueError("Final evaluator coverage/checkpoint differs or is nonfinite")
    if not (evaluation / "official-0" / (stem + ".json")).is_file():
        raise ValueError("Initial-policy control report missing")
    return {
        "episode": ep,
        "iteration": iteration,
        "meshes": meshes,
        "report": final / (stem + ".json"),
        "trajectory": final / (stem + ".parquet"),
        "checkpoint_sha256": sha(checkpoint),
        "frozen_manifest_sha256": sha(frozen / "manifest.json"),
        "bundle_manifest_sha256": sha(bundle / "manifest.json"),
        "legacy_source_verification": origin,
    }


def prepare(kit, catalog_path, out, code_commit):
    if not re.fullmatch(
        r"https://github\.com/[^/]+/[^/]+/commit/[0-9a-f]{40}", code_commit
    ):
        raise ValueError("Use an immutable GitHub commit URL")
    catalog = json.loads(catalog_path.read_text())
    if catalog.get("schema") != "v2d.scored-submission-catalog.v1":
        raise ValueError("Unexpected submission catalog")
    roster = json.loads((kit / "data/track_3_evaluation_objects.json").read_text())[
        "episodes"
    ]
    frames = official_frames(kit, roster)
    rows = catalog["episodes"]
    ids = [row["episode"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(frames):
        raise ValueError(
            "Catalog must cover the exact complete official episode roster"
        )
    inspected = [
        inspect_episode(row, roster[str(row["episode"])], frames[row["episode"]])
        for row in rows
    ]
    out.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "v2d.scored-submission-preparation.v1",
        "catalog_sha256": sha(catalog_path),
        "code_commit": code_commit,
        "official_packer_sha256": sha(kit / "tools/pack_submission.py"),
        "official_roster_sha256": sha(kit / "data/track_3_evaluation_objects.json"),
        "episodes": [],
        "official_packer_passed": False,
        "physical_task_accepted": False,
        "hidden_score_claimed": False,
        "submission_uploaded": False,
        "captain_approval_required": True,
    }
    try:
        results, meshes = out / "results", out / "meshes"
        results.mkdir()
        for item in inspected:
            sources = [item["report"], item["trajectory"]]
            pairs = [(path, results / path.name) for path in sources]
            for name, path in item["meshes"]:
                pairs.append(
                    (
                        path,
                        meshes
                        / f"episode_{item['episode']:06d}"
                        / (name + path.suffix),
                    )
                )
            copies = {}
            for source, target in pairs:
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = sha(source)
                shutil.copyfile(source, target)
                if sha(target) != digest:
                    raise ValueError("Artifact changed during staging")
                copies[str(target.relative_to(out))] = digest
            receipt["episodes"].append(
                {
                    k: v
                    for k, v in item.items()
                    if k not in {"meshes", "report", "trajectory"}
                }
                | {"staged_sha256": copies}
            )
        command = [
            sys.executable,
            str(kit / "tools/pack_submission.py"),
            "--track",
            "track_3",
            "--results",
            str(results),
            "--meshes",
            str(meshes),
            "--code-commit",
            code_commit,
            "-o",
            str(out / "submission.parquet"),
        ]
        receipt["command"] = command
        with (out / "official-packer.log").open("wb") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=900
            )
        receipt["official_packer_returncode"] = result.returncode
        result.check_returncode()
        receipt["submission_sha256"] = sha(out / "submission.parquet")
        receipt["official_packer_passed"] = True
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kit", "catalog", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.kit.resolve(),
                args.catalog.resolve(),
                args.out.resolve(),
                args.code_commit,
            ),
            indent=2,
        )
    )
