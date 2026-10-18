"""Retarget a complete reconstructed episode, train a short baseline and evaluate.

Run under a separate timed compute lease that backs up results and stops the VM.
This command never provisions compute, uploads a submission, or accepts a grasp.
It retains the initial policy and the trained policy with separate evaluations.
"""

import argparse
import hashlib
import json
import re
import struct
import subprocess
import time
from pathlib import Path

TRAINING = "ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:bcde4a54c47743564d65f95725c4891e814077962b3f1b0a4a522ecb7d576922"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(root, files):
    root = root.resolve()
    if not files:
        raise ValueError("Expected a nonempty file closure")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError("Asset or source closure differs: " + name)


def validate(config):
    if type(config["max_seconds"]) is not int or not 0 < config["max_seconds"] <= 7200:
        raise ValueError(
            "Policy run needs a positive deadline no longer than two hours"
        )
    if type(config["iterations"]) is not int or not 1 <= config["iterations"] <= 100:
        raise ValueError("A baseline chunk must contain one to 100 iterations")
    if config.get("anchor_policy", "visual") not in {"visual", "metric"}:
        raise ValueError("Unknown command anchor policy")
    if not re.fullmatch(r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}", config["training_image"]):
        raise ValueError("Training image needs an immutable SHA256 digest")
    for name in ("bundle", "support", "context", "robot_assets", "kit"):
        if not Path(config[name]).is_dir():
            raise ValueError("Missing private/source directory: " + name)
    context = Path(config["context"])
    source = json.loads((context / "context-manifest.json").read_text())
    if source.get("schema") != "v2d.scored-policy-context.v1" or any(
        source.get(key) is not False
        for key in (
            "private_inputs_included",
            "robot_assets_included",
            "model_weights_included",
            "credentials_included",
        )
    ):
        raise ValueError("Expected a committed code-only policy context")
    verify_files(context, source["files"])
    if not (
        context
        / "code/robotic_grounding/source/robotic_grounding/robotic_grounding/assets"
    ).is_dir():
        raise ValueError("Code context needs an empty nested robot asset mount target")
    lock = json.loads(Path(config["robot_asset_lock"]).read_text())
    if lock.get("schema") != "v2d.dex3.robot-assets.v1" or not all(
        f"urdfs/g1/dex3_{side}.urdf" in lock["files_sha256"]
        for side in ("left", "right")
    ):
        raise ValueError("Need both hands and their locked robot meshes")
    verify_files(Path(config["robot_assets"]), lock["files_sha256"])
    metadata = json.loads((Path(config["bundle"]) / "manifest.json").read_text())
    if metadata["episode"] != config["episode"] or any(
        metadata.get(key) is not False
        for key in ("provided_meshes_read", "reference_pose_labels_read")
    ):
        raise ValueError("Need the matching label-free reconstructed episode")
    if not (Path(config["kit"]) / "examples/run_policy_evaluation.py").is_file():
        raise ValueError("Official evaluator is missing")
    return source["source_commit"]


def checkpoint_metadata(path):
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 16 * 1024 * 1024:
            raise ValueError("Checkpoint header exceeds limit")
        metadata = json.loads(stream.read(length))["__metadata__"]
    if metadata.get("format") != "learner_state" or metadata.get("version") != "1":
        raise ValueError("Expected a complete learner-state checkpoint")
    return {
        key.removeprefix("user."): value
        for key, value in metadata.items()
        if key.startswith("user.")
    }


def run(config):
    commit = validate(config)
    out = Path(config["out"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    iterations, ep = config["iterations"], config["episode"]
    deadline = time.monotonic() + config["max_seconds"]
    receipt = {
        "schema": "v2d.scored-single-episode-policy.v1",
        "episode": ep,
        "preparation_source_commit": commit,
        "training_image": config["training_image"],
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "bundle_manifest_sha256": sha(Path(config["bundle"]) / "manifest.json"),
        "robot_asset_lock_sha256": sha(Path(config["robot_asset_lock"])),
        "anchor_policy": config.get("anchor_policy", "visual"),
        "stages": [],
        "physical_task_accepted": False,
        "official_leaderboard_score_claimed": False,
        "private_assets_published": False,
    }

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def container(script, args, label, *, ik=False, pack=None, gpu=False, limit=600):
        name = f"v2d-policy-e{ep:03d}-{label}"
        command = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--network",
            "none",
            "--shm-size",
            "8g",
        ]
        command += (
            ["--gpus", "all"]
            if gpu
            else ["--cpus", "2", "-e", "NVIDIA_VISIBLE_DEVICES=void"]
        )
        mounts = [
            (Path(config["context"]) / "code", "/project", True),
            (
                Path(config["robot_assets"]),
                "/project/robotic_grounding/source/robotic_grounding/robotic_grounding/assets",
                True,
            ),
            (Path(config["robot_assets"]), "/robot-assets", True),
            (Path(config["bundle"]), "/bundle", True),
            (Path(config["support"]), "/support", True),
            (Path(config["kit"]), "/kit", True),
            (out, "/outputs", False),
        ]
        if pack:
            mounts.append((out / pack, "/data", True))
        for source, dest, readonly in mounts:
            command += [
                "-v",
                str(source.resolve()) + ":" + dest + (":ro" if readonly else ""),
            ]
        command += [
            "-v",
            "dex3-training-cache:/root/.cache",
            "-e",
            "OMP_NUM_THREADS=2",
            "-e",
            "OPENBLAS_NUM_THREADS=2",
        ]
        if ik:
            command += ["--entrypoint", "/opt/ik/bin/python"]
        command += [config["training_image"], script, *map(str, args)]
        remaining = int(deadline - time.monotonic() - 10)
        if remaining <= 0:
            raise TimeoutError("Policy batch deadline")
        start = time.monotonic()
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
            receipt["stages"].append({"name": label, "timed_out": True})
            save()
            raise
        stage = {
            "name": label,
            "returncode": result.returncode,
            "wall_seconds": time.monotonic() - start,
        }
        receipt["stages"].append(stage)
        save()
        print(json.dumps(stage), flush=True)
        result.check_returncode()

    try:
        container(
            "/project/robotic_grounding/scripts/retarget/track3_to_dex3.py",
            ["--bundle", "/bundle", "--out", "/outputs/retarget", "--max-iter", "100"],
            "retarget",
            ik=True,
            limit=240,
        )
        container(
            "/project/scripts/track3/prepare_scored_case.py",
            [
                "--bundle",
                "/bundle",
                "--retarget",
                "/outputs/retarget",
                "--support",
                "/support",
                "--out",
                "/outputs/prepared",
                "--anchor-policy",
                receipt["anchor_policy"],
            ],
            "case",
            ik=True,
            limit=180,
        )
        container(
            "/opt/dex3-extension/prepare_window.py",
            [
                "--case",
                "/outputs/prepared/case",
                "--static-inputs",
                "/outputs/prepared/static-inputs",
                "--robot-assets",
                "/robot-assets",
                "--out",
                "/outputs/unfrozen",
                "--full-episode",
            ],
            "portable",
            limit=180,
        )
        container(
            "/opt/dex3-extension/freeze_collision_cache.py",
            ["--inputs", "/data", "--out", "/outputs/frozen"],
            "freeze",
            pack="unfrozen",
            limit=600,
        )
        before = json.loads((out / "unfrozen/manifest.json").read_text())
        after = json.loads((out / "frozen/manifest.json").read_text())
        if before["schema"] != after["schema"] or any(
            after["files"].get(n) != h for n, h in before["files"].items()
        ):
            raise ValueError("Collision freeze changed source targets or assets")
        container(
            "/opt/dex3-extension/check_native_scene.py",
            ["--inputs", "/data", "--out", "/outputs/native-preflight.json"],
            "preflight",
            pack="frozen",
            limit=120,
        )
        container(
            "/opt/dex3-extension/train.py",
            [
                "--config-dir",
                "/opt/dex3-extension/configs",
                "experiment=dex3_scored_baseline",
                "output_dir=/outputs/ppo",
                "hydra.run.dir=/outputs/hydra",
                f"training.max_iterations={iterations}",
                f"training.save_interval={min(25, iterations)}",
            ],
            "train",
            pack="frozen",
            gpu=True,
            limit=600,
        )
        metadata = checkpoint_metadata(out / f"ppo/model_{iterations}.safetensors")
        resolved = json.loads(metadata["resolved_config_json"])
        steps = (
            resolved["training"]["world_count"]
            * resolved["training"]["rollout_steps"]
            * iterations
        )
        rows = [
            json.loads(s) for s in (out / "ppo/metrics.jsonl").read_text().splitlines()
        ]
        if (
            len(rows) != iterations
            or int(metadata["iteration"]) != iterations
            or int(metadata["environment_steps"]) != steps
            or rows[-1]["environment_steps"] != steps
        ):
            raise ValueError("Training counters differ from the requested short chunk")
        receipt["training"] = {
            "iterations": iterations,
            "environment_steps": steps,
            "seed": resolved["training"]["seed"],
        }
        save()
        for iteration in (0, iterations):
            (out / f"checkpoint-{iteration}.json").write_text(
                json.dumps({str(ep): f"/outputs/ppo/model_{iteration}.safetensors"})
                + "\n"
            )
            container(
                "/kit/examples/run_policy_evaluation.py",
                [
                    "--checkpoints",
                    f"/outputs/checkpoint-{iteration}.json",
                    "--results",
                    f"/outputs/official-{iteration}",
                    "--runtime-root",
                    "/opt/flash-chord",
                    "--world-count",
                    "8",
                ],
                f"official-{iteration}",
                pack="frozen",
                gpu=True,
                limit=240,
            )
        container(
            "/opt/dex3-extension/record_policy_readback.py",
            [
                "--checkpoint",
                f"/outputs/ppo/model_{iterations}.safetensors",
                "--out",
                "/outputs/readback",
                "--world-count",
                "8",
            ],
            "readback",
            pack="frozen",
            gpu=True,
            limit=240,
        )
        container(
            "/opt/dex3-extension/audit_scored_policy.py",
            [
                "--run",
                "/outputs/readback",
                "--inputs",
                "/data",
                "--out",
                "/outputs/native-audit.json",
            ],
            "audit",
            pack="frozen",
            limit=90,
        )
        receipt["baseline_complete"] = True
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
