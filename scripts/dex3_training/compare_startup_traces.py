"""Compare hash-bound passive startup traces without running physics.

An equal subset of arrays cannot establish equal startup state. Missing stages,
missing required buffers, nonfinite values and altered evidence stay explicit.
A first difference localizes recorded divergence; it never establishes causality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

REQUIRED_STAGES = (
    "after-construction", "before-capture-step", "after-capture-step",
    "after-explicit-reset", "before-first-control-interval",
    "after-first-control-interval",
)
REQUIRED_FIELDS = (
    "state_0.joint_q", "state_0.joint_qd", "state_0.body_q", "state_0.body_qd",
    "native.qpos", "native.qvel", "native.qacc_warmstart", "native.time",
    "env.timestep", "env.episode_step", "env.observation",
    "action.joint_target", "control.joint_target_pos", "control.joint_f",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_trace(directory, max_uncompressed_bytes=512 * 1024**2):
    directory = Path(directory)
    metadata = directory / "startup-state-trace.json"
    arrays_path = directory / "startup-state-trace.npz"
    report = json.loads(metadata.read_text())
    if report.get("schema") != "v2d.passive-startup-state-trace.v1":
        raise ValueError("Unknown passive trace schema")
    if (report.get("simulation_buffers_written") is not False
        or report.get("extra_physics_steps") != 0
        or report.get("reset_or_warmstart_behavior_changed") is not False
        or report.get("official_evaluator_source_changed") is not False):
        raise ValueError("Need a separately labelled passive trace without simulation changes")
    if digest(arrays_path) != report["arrays_sha256"]:
        raise ValueError("Startup arrays no longer match their receipt")
    with zipfile.ZipFile(arrays_path) as archive:
        if sum(x.file_size for x in archive.infolist()) > max_uncompressed_bytes:
            raise ValueError("Startup trace exceeds the CPU readback limit")
    values, keys = [], set()
    with np.load(arrays_path, allow_pickle=False) as archive:
        for index, snapshot in enumerate(report["snapshots"]):
            if snapshot["index"] != index or not isinstance(snapshot["stage"], str):
                raise ValueError("Invalid startup lifecycle ordering")
            fields = {}
            for name, field in snapshot["fields"].items():
                key = field["key"]
                if key in keys:
                    raise ValueError("A startup buffer was reused across snapshots")
                keys.add(key)
                array = archive[key]
                if array.dtype.hasobject or array.dtype.kind not in "biufc":
                    raise ValueError("Startup buffers must be numeric")
                if list(array.shape) != field["shape"] or array.dtype.str != field["dtype"]:
                    raise ValueError("Startup array shape/dtype differs from receipt")
                if hashlib.sha256(array.tobytes(order="C")).hexdigest() != field["sha256"]:
                    raise ValueError("Startup field hash differs from receipt")
                if bool(np.isfinite(array).all()) != field["finite"]:
                    raise ValueError("Startup finite-value flag differs from evidence")
                fields[name] = array.copy()
            if set(fields) & set(snapshot["missing_fields"]):
                raise ValueError("A startup buffer is both recorded and missing")
            values.append(fields)
        if keys != set(archive.files):
            raise ValueError("Startup arrays contain unbound fields")
    return report, values, dict(metadata_sha256=digest(metadata), arrays_sha256=digest(arrays_path))


def compare(left, right):
    lr, la, lh = load_trace(left)
    rr, ra, rh = load_trace(right)
    left_stages = [s["stage"] for s in lr["snapshots"]]
    right_stages = [s["stage"] for s in rr["snapshots"]]
    compatible = left_stages == right_stages
    coverage, rows, first_difference = [], [], None
    producer_equal = lr["producer_sha256"] == rr["producer_sha256"]
    if not producer_equal:
        coverage.append(dict(side="both", reason="different_trace_producers"))
    for side, report in (("left", lr), ("right", rr)):
        stages = [s["stage"] for s in report["snapshots"]]
        for stage in REQUIRED_STAGES:
            if stage not in stages:
                coverage.append(dict(side=side, stage=stage, reason="missing_required_stage"))
        for snapshot in report["snapshots"]:
            required = list(REQUIRED_FIELDS)
            if snapshot["stage"] == "before-first-control-interval":
                required.append("pending_policy_action")
            for field in required:
                value = snapshot["fields"].get(field)
                if value is None:
                    coverage.append(dict(side=side, stage=snapshot["stage"], field=field, reason="missing_required_buffer"))
                elif not value["finite"]:
                    coverage.append(dict(side=side, stage=snapshot["stage"], field=field, reason="nonfinite_required_buffer"))
    exact = compatible
    if compatible:
        for index, (lf, rf) in enumerate(zip(la, ra)):
            stage = left_stages[index]
            absent_left, absent_right = sorted(set(rf)-set(lf)), sorted(set(lf)-set(rf))
            row = dict(index=index, stage=stage, missing_left=absent_left, missing_right=absent_right, fields=[])
            if absent_left or absent_right:
                exact = False
                if first_difference is None:
                    first_difference = dict(index=index, stage=stage, kind="field_coverage_difference")
            for name in sorted(set(lf) & set(rf)):
                a, b = lf[name], rf[name]
                same_layout = a.shape == b.shape and a.dtype == b.dtype
                same_bits = same_layout and a.tobytes(order="C") == b.tobytes(order="C")
                field = dict(name=name, same_layout=same_layout, bit_exact=same_bits,
                    left_finite=bool(np.isfinite(a).all()), right_finite=bool(np.isfinite(b).all()))
                if not same_bits:
                    exact = False
                    if first_difference is None:
                        first_difference = dict(index=index, stage=stage, field=name, kind="recorded_buffer_difference")
                    if same_layout:
                        finite = np.isfinite(a) & np.isfinite(b)
                        # Float64 subtraction avoids unsigned integer wraparound.
                        if np.iscomplexobj(a):
                            delta = np.abs(a[finite].astype(np.complex128)-b[finite].astype(np.complex128))
                        else:
                            delta = np.abs(a[finite].astype(np.float64)-b[finite].astype(np.float64))
                        field["maximum_finite_absolute_difference"] = float(delta.max()) if len(delta) else None
                        field["different_numeric_elements"] = int(np.count_nonzero(a != b))
                row["fields"].append(field)
            rows.append(row)
    return dict(schema="v2d.passive-startup-trace-comparison.v1", left=lh, right=rh,
        producer_sha256=digest(Path(__file__)), trace_producer_sha256_equal=producer_equal,
        evidence_integrity_verified=True, lifecycle_order_equal=compatible,
        left_stages=left_stages, right_stages=right_stages, coverage_failures=coverage,
        all_recorded_fields_bit_exact=exact,
        required_startup_buffers_finite_and_bit_exact=bool(exact and not coverage),
        first_recorded_difference=first_difference, comparisons=rows,
        source_inputs_modified=False, physics_executed=False,
        all_model_and_solver_state_recorded=False,
        replay_variability_cause_proven=False, physical_task_accepted=False,
        scope="Recorded state comparison only; missing buffers remain unknown and no causal or score claim is made")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "comparisons"}, indent=2))
