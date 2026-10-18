"""Preflight or repeat one frozen scored policy in its exact existing image.

No training, cloud provisioning, image pulling, input edits or submission upload.
Use a separately supervised compute lease that backs up results and stops idle VMs.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from run_scored_policy import checkpoint_metadata, verify_files


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(inputs, checkpoint, expected_report, policy_receipt, kit, out):
    paths = [
        Path(p).resolve()
        for p in (inputs, checkpoint, expected_report, policy_receipt, kit, out)
    ]
    inputs, checkpoint, expected_report, policy_receipt, kit, out = paths
    if out.exists() or any(
        p.is_relative_to(out) or out.is_relative_to(p)
        for p in (inputs, checkpoint.parent, kit)
    ):
        raise ValueError(
            "Replay needs a fresh output outside its read-only input mounts"
        )
    manifest = json.loads((inputs / "manifest.json").read_text())
    verify_files(inputs, manifest["files"])
    if (
        manifest.get("full_episode_covered") is not True
        or manifest.get("source_observations_changed") is not False
        or manifest.get("contact_labels_fabricated") is not False
    ):
        raise ValueError(
            "Need a complete frozen episode with preserved source provenance"
        )
    frames = manifest["source_frame_indices"]
    if len(frames) < 2 or frames != list(range(len(frames))):
        raise ValueError("Replay must cover the complete ordered source clock")
    metadata = checkpoint_metadata(checkpoint)
    if (
        metadata["dex3_input_manifest_sha256"] != sha(inputs / "manifest.json")
        or int(metadata["iteration"]) < 1
    ):
        raise ValueError("Checkpoint belongs to different frozen inputs")
    expected = json.loads(expected_report.read_text())
    if (
        expected["checkpoint_sha256"] != sha(checkpoint)
        or expected["step_count"] != len(frames)
        or expected["start_frame"] != 0
        or expected.get("non_finite_worlds")
        or expected.get("world_count") != 8
    ):
        raise ValueError(
            "Saved evaluation does not bind the complete eight-world control"
        )
    receipt = json.loads(policy_receipt.read_text())
    ep = manifest["episode"]
    if receipt.get("schema") == "v2d.scored-single-episode-policy.v1":
        if receipt.get("episode") != ep or receipt.get("baseline_complete") is not True:
            raise ValueError("Need the completed matching policy receipt")
        image = receipt["training_image"]
    elif receipt.get("schema") == "v2d.scored-sam3d-anchor-comparison.v1":
        label = f"{manifest['command_anchor_policy']}-e{ep:03d}"
        if ep not in receipt["episodes"] or receipt.get("training", {}).get(
            label, {}
        ).get("iterations") != int(metadata["iteration"]):
            raise ValueError("Legacy comparison does not bind this trained episode")
        image = receipt["image"]
    else:
        raise ValueError("Unknown policy producer receipt")
    if not re.fullmatch(r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("Replay requires the immutable producer image")
    evaluator = kit / "examples/run_policy_evaluation.py"
    if not evaluator.is_file():
        raise ValueError("Untouched official evaluator is missing")
    return dict(
        schema="v2d.scored-frozen-policy-replay-plan.v1",
        episode=ep,
        inputs=str(inputs),
        checkpoint=str(checkpoint),
        expected_report=str(expected_report),
        policy_receipt=str(policy_receipt),
        kit=str(kit),
        out=str(out),
        image=image,
        input_manifest_sha256=sha(inputs / "manifest.json"),
        checkpoint_sha256=sha(checkpoint),
        expected_report_sha256=sha(expected_report),
        policy_receipt_sha256=sha(policy_receipt),
        official_evaluator_sha256=sha(evaluator),
        source_frames=len(frames),
        world_count=8,
        iteration=int(metadata["iteration"]),
        environment_steps=int(metadata["environment_steps"]),
        training_steps_added=0,
        new_cloud_instance_created=False,
        private_assets_published=False,
        physical_task_accepted=False,
        bit_exact_retraining_claimed=False,
    )


def commands(plan):
    common = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network",
        "none",
        "--shm-size",
        "8g",
    ]
    for source, target, ro in [
        (plan["inputs"], "/data", True),
        (plan["checkpoint"], "/checkpoint/model.safetensors", True),
        (plan["kit"], "/kit", True),
        (plan["out"], "/outputs", False),
    ]:
        common += ["-v", source + ":" + target + (":ro" if ro else "")]
    common += [
        "-v",
        "dex3-training-cache:/root/.cache",
        "-e",
        "OMP_NUM_THREADS=2",
        "-e",
        "OPENBLAS_NUM_THREADS=2",
    ]
    return [
        (
            "official",
            True,
            [
                "/kit/examples/run_policy_evaluation.py",
                "--checkpoints",
                "/outputs/checkpoints.json",
                "--results",
                "/outputs/official",
                "--runtime-root",
                "/opt/flash-chord",
                "--world-count",
                "8",
            ],
        ),
        (
            "native",
            True,
            [
                "/opt/dex3-extension/record_policy_readback.py",
                "--checkpoint",
                "/checkpoint/model.safetensors",
                "--out",
                "/outputs/readback",
                "--world-count",
                "8",
            ],
        ),
        (
            "audit",
            False,
            [
                "/opt/dex3-extension/audit_scored_policy.py",
                "--run",
                "/outputs/readback",
                "--inputs",
                "/data",
                "--out",
                "/outputs/native-audit.json",
            ],
        ),
    ], common


def execute(plan):
    out = Path(plan["out"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints.json").write_text(
        json.dumps({str(plan["episode"]): "/checkpoint/model.safetensors"}) + "\n"
    )
    report = dict(
        plan=plan, stages=[], replay_complete=False, physical_task_accepted=False
    )
    prefix = "v2d-scored-replay-" + hashlib.sha256(str(out).encode()).hexdigest()[:12]
    stages, common = commands(plan)
    try:
        # Recheck the exact frozen closure immediately before GPU execution.
        verify_files(
            Path(plan["inputs"]),
            json.loads((Path(plan["inputs"]) / "manifest.json").read_text())["files"],
        )
        for key, path in [
            ("input_manifest_sha256", Path(plan["inputs"]) / "manifest.json"),
            ("checkpoint_sha256", Path(plan["checkpoint"])),
            (
                "official_evaluator_sha256",
                Path(plan["kit"]) / "examples/run_policy_evaluation.py",
            ),
            ("expected_report_sha256", Path(plan["expected_report"])),
            ("policy_receipt_sha256", Path(plan["policy_receipt"])),
        ]:
            if sha(path) != plan[key]:
                raise ValueError("Replay source changed after preflight")
        for label, gpu, args in stages:
            name = prefix + "-" + label
            command = (
                common
                + ["--name", name]
                + (
                    ["--gpus", "all"]
                    if gpu
                    else ["--cpus", "2", "-e", "NVIDIA_VISIBLE_DEVICES=void"]
                )
                + [plan["image"], *args]
            )
            try:
                with (out / (label + ".log")).open("wb") as log:
                    result = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=240,
                        check=False,
                    )
                report["stages"].append(dict(name=label, returncode=result.returncode))
                result.check_returncode()
            finally:
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
        actual = json.loads(
            (out / "official" / f"episode_{plan['episode']:06d}.json").read_text()
        )
        native = json.loads((out / "readback/report.json").read_text())
        if (
            actual["checkpoint_sha256"] != plan["checkpoint_sha256"]
            or actual["step_count"] != plan["source_frames"]
            or actual["start_frame"] != 0
            or actual.get("non_finite_worlds")
            or actual.get("world_count") != 8
            or native.get("source_manifest_sha256") != plan["input_manifest_sha256"]
            or native.get("checkpoint_sha256") != plan["checkpoint_sha256"]
            or native.get("frame_count") != plan["source_frames"]
            or native.get("source_motion_speed") != 1.0
            or native.get("evaluation_reset_stabilization_steps") != 0
            or native.get("zero_assistance_control_steps_verified")
            != plan["source_frames"]
            or native.get("virtual_object_assistance") != 0.0
            or native.get("virtual_object_actuator_caps_zero") is not True
        ):
            raise ValueError(
                "Replay lost the full unchanged unassisted evaluation contract"
            )
        expected = json.loads(Path(plan["expected_report"]).read_text())["metrics"]
        if any(
            not math.isfinite(value)
            for metrics in (expected, actual["metrics"])
            for value in metrics.values()
            if type(value) in (int, float)
        ):
            raise ValueError("Replay metrics must be finite before comparison")
        report["metric_comparison"] = {
            key: dict(
                expected=value,
                replay=actual["metrics"][key],
                delta=actual["metrics"][key] - value,
            )
            for key, value in expected.items()
            if type(value) in (int, float)
            and type(actual["metrics"].get(key)) in (int, float)
        }
        report["replay_complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (out / "replay-receipt.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report


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
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the preflighted replay on the current GPU host",
    )
    args = parser.parse_args()
    plan = preflight(
        args.inputs,
        args.checkpoint,
        args.expected_report,
        args.policy_receipt,
        args.kit,
        args.out,
    )
    print(json.dumps(execute(plan) if args.execute else plan, indent=2))
