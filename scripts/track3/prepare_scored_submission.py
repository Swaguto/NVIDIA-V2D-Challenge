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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked(root, name, digest):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path) != digest:
        raise ValueError("Frozen input closure differs: " + name)
    return path


def inspect_episode(row, names, frames):
    ep = row["episode"]
    iteration = row["iteration"]
    if type(ep) is not int or type(iteration) is not int or iteration < 1:
        raise ValueError("Catalog needs explicit integer episodes and checkpoints")
    policy, bundle = Path(row["policy_root"]), Path(row["bundle"])
    receipt = json.loads((policy / "receipt.json").read_text())
    provenance = json.loads((bundle / "manifest.json").read_text())
    frozen = policy / "frozen"
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
    final = policy / f"official-{iteration}"
    stem = f"episode_{ep:06d}"
    report = json.loads((final / (stem + ".json")).read_text())
    if (
        report["checkpoint_sha256"] != sha(checkpoint)
        or report["step_count"] != frames
        or report["start_frame"] != 0
        or report.get("non_finite_worlds")
    ):
        raise ValueError("Final evaluator coverage/checkpoint differs or is nonfinite")
    if not (policy / "official-0" / (stem + ".json")).is_file():
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
