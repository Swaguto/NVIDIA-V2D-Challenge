"""Instrument the upstream checkpoint evaluator with read-only physics snapshots.

The separate official kit run provides evaluator provenance. This diagnostic
adds measured hand/contact state and does not claim an official score.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from flash_chord.evaluation import harness
from diagnose_task import ReadbackEnv


def record(checkpoint, out, world_count=2):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    original = harness.record_object_pose_rollout
    captured = []
    assistance_checks = []

    def instrument(env, jax_env, *args, **kwargs):
        readback = ReadbackEnv(jax_env, env)
        if env.config.voc.max_force != 0 or env.config.voc.max_torque != 0:
            raise ValueError("Policy readback must disable virtual object actuators")
        if env.config.reset.voc_decay_steps != 0:
            raise ValueError("Policy readback must disable training reset stabilization")
        step = readback.step

        def unassisted_step(action):
            for scales in (env.voc_scale, env.object_control_scale):
                if np.any(scales.numpy() != 0):
                    raise ValueError("Nonzero object assistance during policy evaluation")
            result = step(action)
            if np.any(env.object_control_scale.numpy() != 0):
                raise ValueError("Nonzero object assistance after policy evaluation step")
            assistance_checks.append(True)
            return result

        readback.step = unassisted_step
        captured.append((readback, kwargs["reference"]))
        return original(env, readback, *args, **kwargs)

    harness.record_object_pose_rollout = instrument
    try:
        outcome = harness.evaluate_checkpoint(
            checkpoint,
            world_count=world_count,
            object_trajectories_output=out / "objects.parquet",
            eval_script=Path(__file__).resolve(),
        )
    finally:
        harness.record_object_pose_rollout = original
    if len(captured) != 1:
        raise RuntimeError("Expected one unchanged upstream rollout")
    readback, reference = captured[0]
    harness.write_evaluation_report(out / "evaluation-report.json", outcome)
    inputs = Path(outcome.parquet).parent
    with np.load(inputs / "targets.npz") as f:
        source_count = len(f["frame_indices"])
    report = dict(
        schema="v2d.dex3-full-task-policy-readback.v1",
        metrics=asdict(outcome.metrics),
        reference_tracking=asdict(outcome.reference_tracking),
        checkpoint_sha256=outcome.checkpoint_sha256,
        environment_steps=outcome.environment_steps,
        frame_count=outcome.step_count,
        control_fps=outcome.sim_control_fps,
        physics_fps=outcome.sim_physics_fps,
        source_fps=reference.metadata.source_fps,
        source_frame_count=source_count,
        source_manifest_sha256=hashlib.sha256(
            (inputs / "manifest.json").read_bytes()
        ).hexdigest(),
        learned_policy=True,
        virtual_object_assistance=0.0,
        zero_assistance_control_steps_verified=len(assistance_checks),
        virtual_object_actuator_caps_zero=True,
        evaluation_reset_stabilization_steps=0,
        source_motion_speed=1.0,
        native_motor_verification=readback.save(out),
        physical_task_accepted=False,
        official_evaluation_score_claimed=False,
        diagnostic_source_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    )
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--world-count", type=int, default=2)
    a = p.parse_args()
    print(json.dumps(record(a.checkpoint, a.out, a.world_count), indent=2))
