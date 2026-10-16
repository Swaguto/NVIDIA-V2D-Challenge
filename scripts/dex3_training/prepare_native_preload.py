"""Derive bounded normal-only preload for native grasp validation.

The original Isaac plan includes a model-based tangential load torque. Project
that plan onto the recorded compressive pad Jacobians, with a declared minimum
normal preload. This changes inferred motor feedforward only, not observations,
geometry, masses, friction, joint targets, source time or native motor limits.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import nnls


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compressive_projection(effort, jacobians, minimum_normal):
    jacobians = np.asarray(jacobians, float)
    effort = np.asarray(effort, float)
    if (
        jacobians.ndim != 2
        or jacobians.shape[1] != 7
        or not np.isfinite(jacobians).all()
        or not 0.5 <= minimum_normal <= 6
    ):
        raise ValueError("Invalid bounded pad preload")
    matrix = jacobians.T
    floor = np.full(len(jacobians), minimum_normal)
    extra, _ = nnls(matrix, effort - matrix @ floor)
    forces = floor + extra
    target = matrix @ forces
    if not np.isfinite(target).all() or abs(target).max() > 0.55:
        raise ValueError("Projected preload exceeds the planned 0.55 Nm motor bound")
    return target, forces


def smooth(value):
    u = np.clip(value, 0, 1)
    return u * u * (3 - 2 * u)


def prepare(inputs, out, minimum_normal=2.0, sides=("left",)):
    m = json.loads((inputs / "manifest.json").read_text())
    if (
        m["schema"] != "v2d.dex3-training-task.v3"
        or not sides
        or any(s not in ("left", "right") for s in sides)
    ):
        raise ValueError("Need a declared inferred full-task motor plan")
    for name, expected in m["files"].items():
        path = (inputs / name).resolve()
        if not path.is_relative_to(inputs.resolve()) or sha(path) != expected:
            raise ValueError("Input bytes/path changed")
    if out.exists():
        raise FileExistsError(out)
    spec = json.loads((inputs / "task-grips.json").read_text())
    with np.load(inputs / "targets.npz") as f:
        arrays = {k: f[k].copy() for k in f.files}
    ff = arrays["finger_feedforward_Nm"].copy()
    times = arrays["timestamps_s"]
    phase_receipts = []
    for side in sides:
        si = ("left", "right").index(side)
        ff[:, si] = 0
        for phase in spec["phases"][side]:
            names = phase["joint_names"]
            indices = [names.index(n) for n in m["joint_names"][side]]
            jac = np.array(
                [p["normal_jacobian_m_per_rad"] for p in phase["pad_normal_jacobians"]]
            )
            et, efforts = (
                np.asarray(phase["effort_times_s"]),
                np.asarray(phase["effort_Nm"]),
            )
            active = np.flatnonzero(
                (times >= phase["close_s"] - 0.125 - 1e-8)
                & (times <= phase["release_start_s"] + 0.25 + 1e-8)
            )
            maximum_force = 0.0
            for frame in active:
                old = np.array(
                    [np.interp(times[frame], et, efforts[:, j]) for j in range(7)]
                )
                torque, forces = compressive_projection(old, jac, minimum_normal)
                gain = smooth((times[frame] - phase["close_s"] + 0.125) / 0.125) * (
                    1 - smooth((times[frame] - phase["release_start_s"]) / 0.25)
                )
                ff[frame, si] = torque[indices] * gain
                maximum_force = max(maximum_force, float(forces.max()))
            phase_receipts.append(
                dict(sample=phase["sample"], maximum_planned_normal_N=maximum_force)
            )
    shutil.copytree(inputs, out)
    np.savez_compressed(
        out / "original-feedforward.npz",
        finger_feedforward_Nm=arrays["finger_feedforward_Nm"],
    )
    arrays["finger_feedforward_Nm"] = ff
    np.savez_compressed(out / "targets.npz", **arrays)
    receipt = dict(
        schema="v2d.native-compressive-preload.v1",
        source_manifest_sha256=sha(inputs / "manifest.json"),
        preparation_source_sha256=sha(Path(__file__)),
        sides=list(sides),
        minimum_planned_normal_N=minimum_normal,
        phases=phase_receipts,
        maximum_planned_torque_Nm=float(abs(ff).max()),
        native_combined_motor_limit_Nm=0.76,
        original_target_arrays_changed=["finger_feedforward_Nm"],
        source_observations_changed=False,
        geometry_changed=False,
        physical_task_accepted=False,
    )
    (out / "native-preload.json").write_text(json.dumps(receipt, indent=2) + "\n")
    m = copy.deepcopy(m)
    m["native_preload_receipt_sha256"] = sha(out / "native-preload.json")
    m["motor_feedforward"] = (
        "Bounded projection onto compressive pad Jacobians; explicit native-preload.json overrides original task-spec effort plan for declared hands"
    )
    m["files"] = {
        p.relative_to(out).as_posix(): sha(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p != out / "manifest.json"
    }
    (out / "manifest.json").write_text(json.dumps(m, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--minimum-normal", type=float, default=2.0)
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left"])
    a = p.parse_args()
    print(json.dumps(prepare(a.inputs, a.out, a.minimum_normal, a.sides), indent=2))
