"""Preflight or execute one bounded, evaluation-gated Docker batch.

Default invocation is offline: no Docker, GPU, network or cloud commands. Paid
execution needs an explicit reconciled budget receipt. This runner never starts
a VM. Use it inside the operator's verified backup-and-stop lifecycle wrapper.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from evaluation_gate import compare, read_evaluation, sha256


def number(value, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Expected a finite budget or duration")
    if value < 0 or (positive and value == 0):
        raise ValueError("Invalid budget or duration")
    return value


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact or mount path escapes its declared root")
    return path


def mounts(plan):
    result = {"/data": Path(plan["inputs"]), "/outputs": Path(plan["outputs"])}
    for item in plan.get("mounts", []):
        target = item["target"]
        if (
            target
            not in {
                "/baseline",
                "/profile-checkpoint",
                "/profile-context",
                "/kit",
                "/initial-policy",
                "/initial-context",
            }
            or not target.startswith("/")
            or ":" in target
            or "," in target
            or ".." in Path(target).parts
            or target == "/"
            or any(
                Path(target).is_relative_to(old) or Path(old).is_relative_to(target)
                for old in result
            )
            or item.get("read_only") is not True
        ):
            raise ValueError(
                "Extra mounts must have distinct absolute read-only targets"
            )
        source = Path(item["source"])
        if not source.is_absolute() or not source.exists():
            raise ValueError("Extra mount source does not exist")
        result[target] = source
    for source in result.values():
        if "," in str(source) or not source.is_absolute():
            raise ValueError("Mount source must be absolute and contain no comma")
    return result


def artifact(path, bindings):
    path = Path(path)
    for target, source in sorted(bindings.items(), key=lambda x: -len(x[0])):
        if path == Path(target):
            return source.resolve()
        if path.is_relative_to(target):
            return within(source / path.relative_to(target), source)
    raise ValueError("Artifact is outside the declared container mounts")


def verify_inputs(inputs, expected):
    inputs = Path(inputs)
    manifest_path = inputs / "manifest.json"
    if sha256(manifest_path) != expected:
        raise ValueError("Input manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("files") or not manifest.get("collision_cache", {}).get(
        "files"
    ):
        raise ValueError("Input manifest needs frozen files and collision geometry")
    for name, digest in manifest["files"].items():
        member = Path(name)
        if (
            member.is_absolute()
            or ".." in member.parts
            or sha256(within(inputs / member, inputs)) != digest
        ):
            raise ValueError("Private input checksum/path mismatch: " + name)
    for name in manifest["collision_cache"]["files"]:
        if name not in manifest["files"]:
            raise ValueError("Collision cache is not bound by the input manifest")
    return len(manifest["files"])


def preflight(plan):
    if plan.get("schema") != "v2d.compute-batch.v1":
        raise ValueError("Unknown batch schema")
    if not re.fullmatch(r"[a-zA-Z0-9./_-]+@sha256:[0-9a-f]{64}", plan["image"]):
        raise ValueError("Use an immutable image digest")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", plan["cache_volume"]):
        raise ValueError("Invalid named cache volume")
    bindings = mounts(plan)
    inputs, outputs = Path(plan["inputs"]).resolve(), Path(plan["outputs"]).resolve()
    if outputs.exists() or any(
        outputs.is_relative_to(path.resolve()) or path.resolve().is_relative_to(outputs)
        for target, path in bindings.items()
        if target != "/outputs"
    ):
        raise ValueError(
            "Outputs must be fresh and separate from every read-only mount"
        )
    count = verify_inputs(inputs, plan["input_manifest_sha256"])
    for file in plan.get("required_files", []):
        path = artifact(file["path"], bindings)
        if str(file["path"]).startswith("/outputs/") or sha256(path) != file["sha256"]:
            raise ValueError("Required baseline/checkpoint/code checksum mismatch")
    wall = number(plan["max_wall_seconds"], positive=True)
    timeout_total, names, awaiting_gate, saw_evaluation = 0, set(), False, False
    baseline_proofs = []
    if not plan.get("stages"):
        raise ValueError("Batch is empty")
    for stage in plan["stages"]:
        name = stage["name"]
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,60}", name) or name in names:
            raise ValueError("Stage names must be unique simple names")
        names.add(name)
        if stage["kind"] == "container":
            if stage["purpose"] not in ("check", "train", "evaluate"):
                raise ValueError("Unknown container purpose")
            if (
                not stage["script"].startswith("/")
                or not isinstance(stage["args"], list)
                or not all(isinstance(x, str) for x in stage["args"])
            ):
                raise ValueError(
                    "Container command must be an explicit script and argv"
                )
            timeout_total += number(stage["timeout_seconds"], positive=True)
            if stage["purpose"] == "train":
                if awaiting_gate:
                    raise ValueError(
                        "Evaluate and pass a gate before another training chunk"
                    )
                awaiting_gate, saw_evaluation = True, False
                if not str(stage.get("training_metrics", "")).startswith("/outputs/"):
                    raise ValueError("Training must declare its new metrics JSONL")
            if stage["purpose"] == "evaluate":
                saw_evaluation = True
            for path in (
                [stage["training_metrics"]] if stage.get("training_metrics") else []
            ) + stage.get("failure_artifacts", []):
                if not str(path).startswith("/outputs/"):
                    raise ValueError("Failure signals must come from fresh outputs")
                artifact(path, bindings)
        elif stage["kind"] == "gate":
            if not awaiting_gate or not saw_evaluation:
                raise ValueError("A gate must follow training and evaluation")
            if number(stage.get("min_auc_gain", 0.005), positive=True) <= 0:
                raise ValueError("Gate needs a positive improvement")
            for key in (
                "baseline",
                "candidate",
                "baseline_checkpoint",
                "candidate_checkpoint",
            ):
                artifact(stage[key], bindings)
            if stage["baseline"].startswith("/outputs/") != stage[
                "baseline_checkpoint"
            ].startswith("/outputs/"):
                raise ValueError(
                    "Baseline report and checkpoint must have the same provenance"
                )
            if not stage["baseline"].startswith("/outputs/"):
                _, _, meta = read_evaluation(
                    artifact(stage["baseline"], bindings),
                    artifact(stage["baseline_checkpoint"], bindings),
                    inputs,
                )
                if (
                    meta["user.dex3_image_context_sha256"]
                    != plan["image_context_sha256"]
                ):
                    raise ValueError(
                        "Baseline checkpoint and planned image context differ"
                    )
                baseline_proofs.append(
                    sha256(artifact(stage["baseline_checkpoint"], bindings))
                )
            if not stage["candidate"].startswith("/outputs/") or not stage[
                "candidate_checkpoint"
            ].startswith("/outputs/"):
                raise ValueError(
                    "Candidate gate artifacts must come from the current batch"
                )
            awaiting_gate, saw_evaluation = False, False
        else:
            raise ValueError("Unknown stage kind")
    if awaiting_gate:
        raise ValueError("Every training chunk needs a final evaluation and gate")
    if timeout_total + 30 > wall:
        raise ValueError(
            "Stage timeouts must fit inside the batch limit with 30s cleanup reserve"
        )
    return {
        "schema": "v2d.compute-preflight.v1",
        "input_files_verified": count,
        "input_manifest_sha256": plan["input_manifest_sha256"],
        "stages": len(names),
        "stage_timeout_total_seconds": timeout_total,
        "max_wall_seconds": wall,
        "jax_compilation_cache_dir": "/root/.cache/jax",
        "cache_volume": plan["cache_volume"],
        "baseline_checkpoints_verified": baseline_proofs,
        "gpu_executed": False,
        "billing_authorized": False,
    }


def authorize(plan, budget, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    if not budget.get("freeze_file") or Path(budget["freeze_file"]).exists():
        raise ValueError("Paid execution is frozen pending billing reconciliation")
    if any(
        budget.get(key) is not True
        for key in (
            "billing_reconciled",
            "billing_rules_verified",
            "paid_execution_approved",
        )
    ):
        raise ValueError("A reconciled, explicitly approved budget is required")
    if (
        not budget.get("approval_reference")
        or not budget.get("billing_source_reference")
        or not budget.get("instance_id")
    ):
        raise ValueError(
            "Budget needs approval, billing evidence and instance identity"
        )
    if not isinstance(budget.get("currency"), str) or not re.fullmatch(
        r"[A-Z]{3}", budget["currency"]
    ):
        raise ValueError("Budget usage and rates need a verified currency code")
    if number(budget["stop_latency_reserve_seconds"]) < 30:
        raise ValueError(
            "Reserve remaining backup/shutdown time, including at least 30s cleanup"
        )
    observed = dt.datetime.fromisoformat(budget["usage_verified_utc"])
    expires = dt.datetime.fromisoformat(budget["approval_expires_utc"])
    if (
        observed.tzinfo is None
        or expires.tzinfo is None
        or not 0 <= (now - observed).total_seconds() <= 900
        or expires <= now
    ):
        raise ValueError("Billing snapshot or approval is stale")
    # This is a planning reserve after metering rules have been verified. It is
    # not a provider-enforced account cap or an assertion about invoice timing.
    elapsed = (now - observed).total_seconds()
    seconds = max(
        number(budget["minimum_session_seconds"]),
        elapsed
        + number(plan["max_wall_seconds"], True)
        + number(budget["stop_latency_reserve_seconds"]),
    )
    quantum = number(budget["billing_quantum_seconds"], True)
    billed_seconds = math.ceil(seconds / quantum) * quantum
    reserve = billed_seconds / 3600 * (
        number(budget["compute_hourly_rate"], True)
        + number(budget["storage_hourly_rate"])
    ) + number(budget["fixed_session_charge"])
    spent, ceiling = (
        number(budget["observed_account_usage"]),
        number(budget["approved_account_usage_ceiling"], True),
    )
    if spent + reserve > ceiling:
        raise ValueError("Insufficient approved allowance for the entire bounded batch")
    if not budget.get("verified_backup_and_stop_wrapper"):
        raise ValueError("Run through a verified backup-and-stop lifecycle wrapper")
    return {
        "instance_id": budget["instance_id"],
        "approval_reference": budget["approval_reference"],
        "estimated_batch_reserve": reserve,
        "currency": budget["currency"],
        "provider_enforced_spending_cap": False,
        "metering_must_be_reconciled_after_run": True,
    }


def docker_command(plan, stage, name):
    cmd = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--gpus",
        "all",
        "--shm-size=8g",
        "--memory=24g",
        "--memory-swap=24g",
        "-e",
        "WANDB_MODE=disabled",
        "-e",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "-e",
        "JAX_COMPILATION_CACHE_DIR=/root/.cache/jax",
        "-e",
        "JAX_EXPLAIN_CACHE_MISSES=true",
        "-e",
        "FLASH_CHORD_CACHE_DIR=/data/collision-cache",
        "--mount",
        "type=volume,src=" + plan["cache_volume"] + ",dst=/root/.cache",
    ]
    for target, source in mounts(plan).items():
        mount = "type=bind,src=" + str(source) + ",dst=" + target
        cmd += ["--mount", mount + (",readonly" if target != "/outputs" else "")]
    # All published training images use this Python entrypoint. Explicitly pin
    # it here so a mismatched default command cannot silently train instead.
    return cmd + [
        "--entrypoint",
        "/opt/flash-env/bin/python",
        plan["image"],
        stage["script"],
        *stage["args"],
    ]


def training_failure(stage, bindings, final=False):
    for name in stage.get("failure_artifacts", []):
        if artifact(name, bindings).exists():
            return "Numerical failure artifact: " + name
    if not stage.get("training_metrics"):
        return None
    path = artifact(stage["training_metrics"], bindings)
    if not path.exists():
        return "Training exited without its metrics JSONL" if final else None
    data = path.read_bytes()
    if final and (not data or not data.endswith(b"\n")):
        return "Training exited without complete metrics records"
    # The final line can be in flight. Only consume complete logger writes.
    previous_step = -1
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        row = json.loads(line)
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                return "Nonfinite training metric"
        if row.get("numerical_failures/cumulative_world_transitions", 0) > 0:
            return "Numerical failure reported during training"
        step = row.get("environment_steps")
        if isinstance(step, bool) or not isinstance(step, int) or step <= previous_step:
            return "Missing or nonmonotonic training step counter"
        previous_step = step
    return None


def gpu_sample():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0:
            return [
                [float(x.strip()) for x in line.split(",")]
                for line in result.stdout.splitlines()
            ]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def run_container(plan, stage, name, output, deadline):
    started = time.monotonic()
    limit = min(started + stage["timeout_seconds"], deadline - 30)
    command = docker_command(plan, stage, name)
    reason, samples, cleanup = None, [], None
    process = None
    try:
        with (output / (stage["name"] + ".log")).open("xb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                reason = (
                    "Paid execution freeze was activated"
                    if plan.get("execution_freeze_file")
                    and Path(plan["execution_freeze_file"]).exists()
                    else training_failure(stage, mounts(plan))
                )
                if reason or time.monotonic() >= limit:
                    reason = reason or "Stage or batch wall-time limit reached"
                    break
                sample = gpu_sample()
                if sample is not None:
                    samples.append(sample)
                time.sleep(min(2, max(0, limit - time.monotonic())))
            # Check the final flushed metrics even if the process exited
            # between two samples. A zero exit code cannot hide bad metrics.
            reason = reason or training_failure(
                stage, mounts(plan), final=process.poll() is not None
            )
    finally:
        if process is not None and (reason or process.poll() is None):
            try:
                cleanup = subprocess.run(
                    ["docker", "rm", "-f", name],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                ).returncode
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
    return {
        "name": stage["name"],
        "purpose": stage["purpose"],
        "returncode": process.returncode,
        "stopped_reason": reason,
        "elapsed_seconds": time.monotonic() - started,
        "forced_container_cleanup_returncode": cleanup,
        "gpu_samples": samples,
        "gpu_monitoring_available": bool(samples),
        "peak_sampled_gpu_memory_mib": max(
            (gpu[0] for sample in samples for gpu in sample), default=None
        ),
    }


def execute(plan, budget, run_container_fn=run_container):
    checked = preflight(plan)
    approved = authorize(plan, budget)
    plan = dict(plan, execution_freeze_file=budget["freeze_file"])
    root = Path(plan["outputs"])
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "v2d.compute-batch-receipt.v1",
        "preflight": checked,
        "budget": approved,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stages": [],
        "vm_started_by_runner": False,
        "instance_shutdown_verified": False,
        "lifecycle_owner": budget["verified_backup_and_stop_wrapper"],
        "completed": False,
    }
    deadline = time.monotonic() + plan["max_wall_seconds"]
    prefix = "v2d-batch-" + uuid.uuid4().hex[:10]

    def save():
        (root / "batch-receipt.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    try:
        for stage in plan["stages"]:
            if Path(budget["freeze_file"]).exists():
                raise RuntimeError("Paid execution freeze was activated")
            if time.monotonic() >= deadline - 30:
                raise RuntimeError("Batch wall-time exhausted")
            if stage["kind"] == "gate":
                bindings = mounts(plan)
                result = compare(
                    *(
                        artifact(stage[k], bindings)
                        for k in (
                            "baseline",
                            "candidate",
                            "baseline_checkpoint",
                            "candidate_checkpoint",
                        )
                    ),
                    inputs=plan["inputs"],
                    min_auc_gain=stage.get("min_auc_gain", 0.005),
                )
                (root / (stage["name"] + ".json")).write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n"
                )
                report["stages"].append({"name": stage["name"], "gate": result})
                save()
                if not result["continue_training"]:
                    report["stopped_reason"] = (
                        "Evaluation gate declined further compute"
                    )
                    return report
            else:
                result = run_container_fn(
                    plan, stage, prefix + "-" + stage["name"], root, deadline
                )
                report["stages"].append(result)
                save()
                if result["returncode"] != 0 or result["stopped_reason"]:
                    report["stopped_reason"] = (
                        result["stopped_reason"] or "Stage failed; no automatic retry"
                    )
                    return report
        report["completed"] = True
        return report
    except BaseException as error:
        report["stopped_reason"] = repr(error)
        raise
    finally:
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--report", type=Path, help="Save an offline preflight receipt")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--budget", type=Path)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    if a.execute:
        if sys.platform != "linux" or not a.budget:
            p.error("Execution needs Linux and an explicit private budget receipt")

        def interrupted(signum, frame):
            raise SystemExit(128 + signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, interrupted)
        result = execute(plan, json.loads(a.budget.read_text()))
    else:
        result = preflight(plan)
        if a.report:
            with a.report.open("x") as stream:
                stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if not a.execute or result["completed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
