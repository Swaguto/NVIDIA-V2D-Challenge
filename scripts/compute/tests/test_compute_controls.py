"""Cost controls must fail closed before a GPU process, and on bad results."""

import datetime as dt
import json
import struct
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import evaluation_gate as gate
import prepare_training_batch as prepare
import run_batch as batch


def checkpoint(path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"__metadata__": meta}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def evaluation(path, policy, inputs, auc=0.8, mppe=2):
    path.mkdir(parents=True)
    meta = gate.metadata(policy)
    metrics = {
        "add_auc": auc,
        "chord_sr": 1,
        "spider_sr_uncentered": 1,
        "maniptrans_sr": 1,
        "relative_position_error_cm": 3,
        "mppe_cm": mppe,
        "world_count": 4,
        "step_count": 2,
        "object_body_names": ["pot"],
    }
    tracking = {"reference_end_fraction": 1, "success_rate": 1}
    common = {
        "checkpoint_sha256": gate.sha256(policy),
        "environment_steps": int(meta["user.environment_steps"]),
        "metrics": metrics,
        "reference_tracking": tracking,
    }
    raw = dict(
        common,
        step_count=2,
        world_count=4,
        non_finite_worlds=[],
        reset_mode="explicit",
        settling_steps=0,
        start_frame=0,
        motion_start_frame=0,
        motion_end_frame=-1,
        sim_control_fps=20,
        sim_physics_fps=400,
        object_orientation_threshold_rad=0.5,
        object_position_threshold_m=0.1,
    )
    readback = dict(
        common,
        schema="v2d.dex3-full-task-policy-readback.v1",
        source_manifest_sha256=gate.sha256(inputs / "manifest.json"),
        frame_count=2,
        source_frame_count=2,
        virtual_object_assistance=0,
        virtual_object_actuator_caps_zero=True,
        evaluation_reset_stabilization_steps=0,
        zero_assistance_control_steps_verified=2,
        diagnostic_source_sha256=gate.sha256(
            Path(gate.__file__).parents[1] / "dex3_training/record_policy_readback.py"
        ),
    )
    (path / "report.json").write_text(json.dumps(readback))
    (path / "evaluation-report.json").write_text(json.dumps(raw))


@pytest.fixture
def setup(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "hull.bin").write_bytes(b"frozen geometry")
    (inputs / "manifest.json").write_text(
        json.dumps(
            {
                "files": {"hull.bin": gate.sha256(inputs / "hull.bin")},
                "collision_cache": {"files": ["hull.bin"]},
                "objects": [{"name": "pot"}],
                "selected_frame_count": 2,
                "source_frame_count": 2,
            }
        )
    )
    context = tmp_path / "context/context-manifest.json"
    context.parent.mkdir()
    context.write_text('{"clean_code_only_context":true}')
    policy = tmp_path / "policies/model_2000.safetensors"
    cfg = {
        "training": {
            "world_count": 256,
            "rollout_steps": 32,
            "max_iterations": 2000,
            "save_interval": 100,
            "resume": False,
            "checkpoint": None,
        },
        "env": {"sim": {"fps": 20}},
    }
    meta = {
        "user.dex3_input_manifest_sha256": gate.sha256(inputs / "manifest.json"),
        "user.dex3_image_context_sha256": gate.sha256(context),
        "user.dex3_runtime_patches_sha256": "1" * 64,
        "user.policy_schema_json": '{"action":14}',
        "user.resolved_config_json": json.dumps(cfg),
        "user.iteration": "2000",
        "user.environment_steps": str(2000 * 8192),
    }
    checkpoint(policy, meta)
    baseline = tmp_path / "baseline"
    evaluation(baseline, policy, inputs)
    plan = prepare.prepare_training(
        inputs,
        policy,
        context,
        baseline,
        tmp_path / "new-outputs",
        "ghcr.io/team/code@sha256:" + "a" * 64,
        "dex3_grip_estimated_cws",
    )
    now = dt.datetime.now(dt.timezone.utc)
    budget = {
        "freeze_file": str(tmp_path / "unresolved-billing.lock"),
        "billing_reconciled": True,
        "billing_rules_verified": True,
        "paid_execution_approved": True,
        "approval_reference": "synthetic test approval",
        "billing_source_reference": "synthetic metering fixture",
        "instance_id": "fixture",
        "usage_verified_utc": now.isoformat(),
        "approval_expires_utc": (now + dt.timedelta(hours=1)).isoformat(),
        "minimum_session_seconds": 0,
        "billing_quantum_seconds": 1,
        "stop_latency_reserve_seconds": 30,
        "compute_hourly_rate": 2,
        "storage_hourly_rate": 0.03,
        "fixed_session_charge": 0,
        "observed_account_usage": 10,
        "approved_account_usage_ceiling": 20,
        "currency": "USD",
        "verified_backup_and_stop_wrapper": "fixture lifecycle",
    }
    return plan, budget, inputs, policy, meta


def test_offline_preflight_calls_no_process_and_checks_inputs(setup, monkeypatch):
    plan, _, inputs, _, _ = setup

    def forbidden(*a, **kw):
        raise AssertionError("Offline preflight tried to invoke a process")

    monkeypatch.setattr(batch.subprocess, "run", forbidden)
    monkeypatch.setattr(batch.subprocess, "Popen", forbidden)
    report = batch.preflight(plan)
    assert report["gpu_executed"] is False and report["input_files_verified"] == 1
    (inputs / "hull.bin").write_bytes(b"different geometry")
    with pytest.raises(ValueError, match="checksum/path"):
        batch.preflight(plan)


@pytest.mark.parametrize(
    "change",
    [
        "freeze",
        "unreconciled",
        "unapproved",
        "stale",
        "no-allowance",
        "no-lifecycle",
        "unknown-metering",
    ],
)
def test_budget_denial_precedes_output_creation_or_any_process(setup, change):
    plan, budget, _, _, _ = setup
    if change == "freeze":
        Path(budget["freeze_file"]).write_text("pending reconciliation")
    elif change == "unreconciled":
        budget["billing_reconciled"] = False
    elif change == "unapproved":
        budget["paid_execution_approved"] = False
    elif change == "unknown-metering":
        budget["billing_rules_verified"] = False
    elif change == "no-lifecycle":
        budget["verified_backup_and_stop_wrapper"] = ""
    elif change == "no-allowance":
        budget["approved_account_usage_ceiling"] = 10.01
    else:
        budget["usage_verified_utc"] = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
        ).isoformat()

    def forbidden(*a, **kw):
        raise AssertionError("Denied budget invoked a GPU stage")

    with pytest.raises(ValueError):
        batch.execute(plan, budget, forbidden)
    assert not Path(plan["outputs"]).exists()


def test_missing_final_gate_and_oversized_stage_are_refused(setup):
    plan = deepcopy(setup[0])
    plan["stages"].pop()
    with pytest.raises(ValueError, match="final evaluation and gate"):
        batch.preflight(plan)
    plan = deepcopy(setup[0])
    plan["stages"][0]["timeout_seconds"] = 5000
    with pytest.raises(ValueError, match="timeouts must fit"):
        batch.preflight(plan)


def test_path_escape_and_overlapping_mounts_are_refused(setup):
    plan = deepcopy(setup[0])
    plan["mounts"].append(
        {"source": plan["inputs"], "target": "/outputs/steal", "read_only": True}
    )
    with pytest.raises(ValueError, match="distinct"):
        batch.preflight(plan)
    with pytest.raises(ValueError, match="escapes"):
        batch.artifact("/outputs/../../outside", batch.mounts(setup[0]))
    plan["mounts"][-1]["target"] = "/opt/flash-chord"
    with pytest.raises(ValueError, match="distinct"):
        batch.preflight(plan)


def simulated_executor(inputs, meta, improve):
    calls = []

    def run(plan, stage, name, output, deadline):
        calls.append(stage["name"])
        bindings = batch.mounts(plan)
        if stage["purpose"] == "train":
            iteration = int(
                next(
                    x.split("=", 1)[1]
                    for x in stage["args"]
                    if x.startswith("training.max_iterations=")
                )
            )
            new_meta = dict(
                meta,
                **{
                    "user.iteration": str(iteration),
                    "user.environment_steps": str(iteration * 8192),
                },
            )
            policy = output / stage["name"] / "ppo" / f"model_{iteration}.safetensors"
            checkpoint(policy, new_meta)
        else:
            args = stage["args"]
            policy = batch.artifact(args[args.index("--checkpoint") + 1], bindings)
            directory = batch.artifact(args[args.index("--out") + 1], bindings)
            evaluation(
                directory,
                policy,
                inputs,
                auc=(0.82 if stage["name"] == "evaluate-short" else 0.84)
                if improve
                else 0.79,
            )
        return {"name": stage["name"], "returncode": 0, "stopped_reason": None}

    return calls, run


def test_regression_prevents_conditional_long_training(setup):
    plan, budget, inputs, _, meta = setup
    calls, run = simulated_executor(inputs, meta, False)
    result = batch.execute(plan, budget, run)
    assert calls == ["train-short", "evaluate-short"]
    assert (
        not result["completed"]
        and not result["stages"][-1]["gate"]["continue_training"]
    )
    assert (Path(plan["outputs"]) / "batch-receipt.json").exists()


def test_improvement_unlocks_one_bounded_chunk_and_retains_both_gates(setup):
    plan, budget, inputs, _, meta = setup
    calls, run = simulated_executor(inputs, meta, True)
    result = batch.execute(plan, budget, run)
    assert calls == ["train-short", "evaluate-short", "train-longer", "evaluate-longer"]
    assert result["completed"] and result["instance_shutdown_verified"] is False
    assert len([x for x in result["stages"] if "gate" in x]) == 2


def test_stage_failure_never_retries_or_runs_later_stages(setup):
    plan, budget, _, _, _ = setup
    calls = []

    def fail(plan, stage, *args):
        calls.append(stage["name"])
        return {"name": stage["name"], "returncode": 9, "stopped_reason": None}

    report = batch.execute(plan, budget, fail)
    assert calls == ["train-short"] and not report["completed"]


def test_tradeoff_nan_and_incompatible_coverage_cannot_pass(setup, tmp_path):
    _plan, _, inputs, policy, meta = setup
    new_policy = tmp_path / "candidate/model_2100.safetensors"
    checkpoint(
        new_policy,
        dict(
            meta,
            **{"user.iteration": "2100", "user.environment_steps": str(2100 * 8192)},
        ),
    )
    candidate = tmp_path / "candidate/readback"
    evaluation(candidate, new_policy, inputs, auc=0.82, mppe=3)
    args = (tmp_path / "baseline", candidate, policy, new_policy, inputs)
    assert gate.compare(*args)["continue_training"] is False
    for name in ("report.json", "evaluation-report.json"):
        path = candidate / name
        data = json.loads(path.read_text())
        data["metrics"]["mppe_cm"] = 2
        data["metrics"]["add_auc"] = float("nan")
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        gate.compare(*args)


def test_timeout_terminates_real_child_and_requests_container_removal(
    setup, monkeypatch
):
    plan = setup[0]
    output = Path(plan["outputs"])
    output.mkdir()
    stage = dict(plan["stages"][0], timeout_seconds=0.05)
    monkeypatch.setattr(
        batch,
        "docker_command",
        lambda *a: [sys.executable, "-c", "import time;time.sleep(30)"],
    )
    monkeypatch.setattr(batch, "gpu_sample", lambda: None)
    cleanup = []

    def remove(argv, **kw):
        cleanup.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch.subprocess, "run", remove)
    result = batch.run_container(
        plan, stage, "fixture-container", output, batch.time.monotonic() + 60
    )
    assert "wall-time" in result["stopped_reason"] and result["returncode"] != 0
    assert cleanup == [["docker", "rm", "-f", "fixture-container"]]


def test_zero_exit_with_numerical_failure_is_rejected(setup):
    plan = setup[0]
    stage = plan["stages"][0]
    path = batch.artifact(stage["training_metrics"], batch.mounts(plan))
    path.parent.mkdir(parents=True)
    path.write_text('{"numerical_failures/cumulative_world_transitions":1}\n')
    assert batch.training_failure(stage, batch.mounts(plan))
    path.write_text('{"reward":NaN}\n')
    assert batch.training_failure(stage, batch.mounts(plan))


def test_jax_cache_is_inside_persistent_volume_and_image_never_pulls(setup):
    plan = setup[0]
    command = batch.docker_command(plan, plan["stages"][0], "test")
    assert "JAX_COMPILATION_CACHE_DIR=/root/.cache/jax" in command
    assert "type=volume,src=dex3-training-cache,dst=/root/.cache" in command
    assert "--pull=never" in command


def test_successful_process_cannot_hide_bad_final_metrics(setup, monkeypatch):
    plan = setup[0]
    output = Path(plan["outputs"])
    output.mkdir()
    stage = dict(plan["stages"][0], timeout_seconds=0.3)
    metric_path = batch.artifact(stage["training_metrics"], batch.mounts(plan))
    metric_path.parent.mkdir(parents=True)
    script = (
        "from pathlib import Path;Path("
        + repr(str(metric_path))
        + ").write_text('{\"reward\":NaN}\\n')"
    )
    monkeypatch.setattr(
        batch, "docker_command", lambda *a: [sys.executable, "-c", script]
    )
    monkeypatch.setattr(batch, "gpu_sample", lambda: None)
    monkeypatch.setattr(
        batch.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
    )
    result = batch.run_container(
        plan, stage, "fixture-container", output, batch.time.monotonic() + 60
    )
    assert result["returncode"] == 0
    assert result["stopped_reason"] == "Nonfinite training metric"


def test_report_with_different_coverage_is_refused_before_any_gpu(setup):
    plan = setup[0]
    baseline = batch.artifact("/baseline", batch.mounts(plan))
    path = baseline / "evaluation-report.json"
    data = json.loads(path.read_text())
    data["world_count"] = 1
    path.write_text(json.dumps(data))
    # Even if an operator refreshes the file's checksum, its semantics fail.
    for item in plan["required_files"]:
        if item["path"].endswith("evaluation-report.json"):
            item["sha256"] = gate.sha256(path)
    with pytest.raises(ValueError, match="readback contract"):
        batch.preflight(plan)
