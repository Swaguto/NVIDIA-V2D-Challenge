"""Continue one scored policy by a bounded chunk with repeated full evaluations.

An outer compute lease owns provisioning, verified backups and provider stop.
This runner keeps the control intact and never promotes or submits a policy.
"""

import argparse
import json
import re
import subprocess
import time
from copy import deepcopy
from pathlib import Path

from run_scored_policy import checkpoint_metadata, collision_seed_manifest, sha


def prepare(config):
    if not re.fullmatch(r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}", config["training_image"]):
        raise ValueError("Continuation image must have an immutable SHA256 digest")
    if (
        type(config["chunk_iterations"]) is not int
        or not 1 <= config["chunk_iterations"] <= 100
    ):
        raise ValueError("Continuation needs one to 100 new iterations")
    if type(config["max_seconds"]) is not int or not 0 < config["max_seconds"] <= 1800:
        raise ValueError("Continuation needs a deadline no longer than 30 minutes")
    parent = Path(config["parent"]).resolve()
    receipt = json.loads((parent / "receipt.json").read_text())
    if (
        receipt.get("baseline_complete") is not True
        or receipt["training_image"] != config["training_image"]
    ):
        raise ValueError("Need a complete baseline in the identical pinned image")
    iteration = receipt["training"]["iterations"]
    checkpoint = parent / f"ppo/model_{iteration}.safetensors"
    metadata = checkpoint_metadata(checkpoint)
    resolved = json.loads(metadata["resolved_config_json"])
    stride = resolved["training"]["world_count"] * resolved["training"]["rollout_steps"]
    if (
        int(metadata["iteration"]) != iteration
        or int(metadata["environment_steps"]) != iteration * stride
    ):
        raise ValueError("Parent learner counters do not match its baseline receipt")
    collision_seed_manifest(parent / "frozen")
    if sha(parent / "frozen/manifest.json") != metadata["dex3_input_manifest_sha256"]:
        raise ValueError("Parent checkpoint does not bind these exact frozen inputs")
    if not (Path(config["kit"]) / "examples/run_policy_evaluation.py").is_file():
        raise ValueError("Untouched official evaluator is required")
    profile = deepcopy(resolved)
    profile["output_dir"] = "/outputs/ppo"
    profile["resume_context_manifest"] = "/outputs/image-context-manifest.json"
    profile["training"].update(
        resume=True,
        checkpoint=f"/parent/ppo/model_{iteration}.safetensors",
        max_iterations=iteration + config["chunk_iterations"],
        save_interval=25,
    )
    return parent, receipt, checkpoint, metadata, profile, stride


def summarize(control, candidate, expected_frames):
    if len(control) != 3 or len(candidate) != 3:
        raise ValueError("Need three full readbacks for each checkpoint")
    result = {}
    for name, rows in [("control", control), ("candidate", candidate)]:
        for row in rows:
            if (
                row.get("world_count") != 8
                or row.get("non_finite_worlds") != []
                or row.get("start_frame") != 0
                or row.get("step_count") != expected_frames
            ):
                raise ValueError(
                    "Comparison requires finite eight-world frame-zero evaluations"
                )
        auc = [r["metrics"]["add_auc"] for r in rows]
        sr = [r["metrics"]["maniptrans_sr"] for r in rows]
        result[name] = {
            "mean_auc": sum(auc) / 3,
            "minimum_auc": min(auc),
            "maximum_auc": max(auc),
            "mean_mp_sr": sum(sr) / 3,
            "minimum_mp_sr": min(sr),
            "maximum_mp_sr": max(sr),
        }
    # A repeatability screen is not a confidence interval or a training-seed test.
    result["auc_ranges_overlap"] = not (
        result["candidate"]["minimum_auc"] > result["control"]["maximum_auc"]
        or result["control"]["minimum_auc"] > result["candidate"]["maximum_auc"]
    )
    result["policy_promoted"] = False
    result["independent_training_repeat_still_required"] = True
    result["physical_task_accepted"] = False
    return result


def run(config):
    parent, baseline, checkpoint, metadata, profile, stride = prepare(config)
    out = Path(config["out"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "profile.yaml").write_text(
        json.dumps(profile, indent=2, allow_nan=False) + "\n"
    )
    deadline = time.monotonic() + config["max_seconds"]
    start_iteration = int(metadata["iteration"])
    end_iteration = profile["training"]["max_iterations"]
    ep = baseline["episode"]
    receipt = {
        "schema": "v2d.scored-policy-continuation.v1",
        "episode": ep,
        "parent_checkpoint_sha256": sha(checkpoint),
        "frozen_manifest_sha256": sha(parent / "frozen/manifest.json"),
        "training_image": config["training_image"],
        "parent_iteration": start_iteration,
        "end_iteration": end_iteration,
        "new_environment_steps": config["chunk_iterations"] * stride,
        "repeated_evaluations_per_checkpoint": 3,
        "source_observations_or_collision_geometry_changed": False,
        "policy_promoted": False,
        "physical_task_accepted": False,
        "private_assets_published": False,
        "official_leaderboard_score_claimed": False,
        "stages": [],
    }

    def save():
        (out / "receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )

    def container(script, args, label, gpu=False, limit=300):
        name = f"v2d-continuation-e{ep:03d}-{label}"
        command = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--network",
            "none",
            "--name",
            name,
            "--shm-size",
            "8g",
        ]
        command += (
            ["--gpus", "all"]
            if gpu
            else ["--cpus", "2", "-e", "NVIDIA_VISIBLE_DEVICES=void"]
        )
        for root, target in [
            (parent, "/parent:ro"),
            (parent / "frozen", "/data:ro"),
            (Path(config["kit"]), "/kit:ro"),
            (out, "/outputs"),
        ]:
            command += ["-v", str(root.resolve()) + ":" + target]
        command += [
            "-v",
            "dex3-training-cache:/root/.cache",
            "-e",
            "OMP_NUM_THREADS=2",
            "-e",
            "OPENBLAS_NUM_THREADS=2",
            config["training_image"],
            script,
            *map(str, args),
        ]
        remaining = int(deadline - time.monotonic() - 10)
        if remaining <= 0:
            raise TimeoutError("Continuation deadline")
        begin = time.monotonic()
        try:
            with (out / (label + ".log")).open("wb") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=min(limit, remaining),
                    check=False,
                )
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                timeout=30,
                check=False,
            )
            raise
        receipt["stages"].append(
            {
                "name": label,
                "returncode": result.returncode,
                "wall_seconds": time.monotonic() - begin,
            }
        )
        save()
        result.check_returncode()

    try:
        container(
            "-c",
            [
                "from pathlib import Path; import shutil; shutil.copyfile('/opt/dex3-extension/image-context-manifest.json','/outputs/image-context-manifest.json')"
            ],
            "image-context",
        )
        if (
            sha(out / "image-context-manifest.json")
            != metadata["dex3_image_context_sha256"]
        ):
            raise ValueError("Image context differs from the parent checkpoint")
        container(
            "/opt/dex3-extension/train.py",
            [
                "--config-path",
                "/outputs",
                "--config-name",
                "profile",
                "hydra.run.dir=/outputs/hydra",
            ],
            "train",
            True,
            600,
        )
        lineage = json.loads((out / "ppo/resume-lineage.json").read_text())
        if (
            lineage["parent_checkpoint_sha256"] != sha(checkpoint)
            or lineage["unchanged_experiment"] is not True
        ):
            raise ValueError("Continuation changed the experiment or parent learner")
        rows = [
            json.loads(s) for s in (out / "ppo/metrics.jsonl").read_text().splitlines()
        ]
        new_checkpoint = out / f"ppo/model_{end_iteration}.safetensors"
        new_metadata = checkpoint_metadata(new_checkpoint)
        if (
            len(rows) != config["chunk_iterations"]
            or rows[0]["environment_steps"] != (start_iteration + 1) * stride
            or rows[-1]["environment_steps"] != end_iteration * stride
            or int(new_metadata["environment_steps"]) != end_iteration * stride
        ):
            raise ValueError("Continuation counters differ from the exact new chunk")
        comparisons = {}
        for label, model in [
            ("control", f"/parent/ppo/model_{start_iteration}.safetensors"),
            ("candidate", f"/outputs/ppo/model_{end_iteration}.safetensors"),
        ]:
            reports = []
            for repeat in range(1, 4):
                folder = f"{label}-readback-{repeat}"
                container(
                    "/opt/dex3-extension/record_policy_readback.py",
                    [
                        "--checkpoint",
                        model,
                        "--out",
                        "/outputs/" + folder,
                        "--world-count",
                        "8",
                    ],
                    folder,
                    True,
                    180,
                )
                reports.append(
                    json.loads((out / folder / "evaluation-report.json").read_text())
                )
            comparisons[label] = reports
            (out / f"{label}-checkpoint.json").write_text(
                json.dumps({str(ep): model}) + "\n"
            )
            container(
                "/kit/examples/run_policy_evaluation.py",
                [
                    "--checkpoints",
                    f"/outputs/{label}-checkpoint.json",
                    "--results",
                    f"/outputs/{label}-official",
                    "--runtime-root",
                    "/opt/flash-chord",
                    "--world-count",
                    "8",
                ],
                label + "-official",
                True,
                180,
            )
        frozen = json.loads((parent / "frozen/manifest.json").read_text())
        receipt["comparison"] = summarize(
            comparisons["control"],
            comparisons["candidate"],
            frozen["source_frame_count"],
        )
        receipt["completed"] = True
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        save()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
