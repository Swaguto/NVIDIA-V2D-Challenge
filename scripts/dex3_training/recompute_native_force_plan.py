"""Recompute inertial feedforward at native source speed, freezing other targets."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

sys.path[:0] = [
    str(Path(__file__).parents[1] / "dex3_physics"),
    str(Path(__file__).parents[1] / "reproduction"),
]
from task_grasp import build
from prepare_task_reference import prepare


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recompute(inputs, case, proposal, static_manifest, urdfs, out):
    if out.exists():
        raise FileExistsError(out)
    seed = json.loads(proposal.read_text())
    static = json.loads(static_manifest.read_text())
    spec = build(
        case,
        [r["source_sample"] for r in seed["results"]],
        seed,
        np.array(static["stage_rotation_from_inferred_support_normal"]),
        urdfs,
        physical_source_speed=1.0,
    )
    with tempfile.TemporaryDirectory(
        prefix="native-force-plan-", dir=out.parent
    ) as temporary:
        root = Path(temporary)
        spec_path = root / "task-grips.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        regenerated = root / "inputs"
        prepare(inputs, case, spec_path, regenerated)
        with np.load(inputs / "targets.npz") as f:
            old = {k: f[k].copy() for k in f.files}
        with np.load(regenerated / "targets.npz") as f:
            fresh = {k: f[k].copy() for k in f.files}
        differences = {}
        for key in old:
            if key == "finger_feedforward_Nm":
                continue
            difference = float(
                abs(old[key].astype(float) - fresh[key].astype(float)).max()
            )
            if old[key].shape != fresh[key].shape or difference > 1e-6:
                raise ValueError(
                    "Force regeneration changed the inferred geometry/time plan: " + key
                )
            differences[key] = difference
        delta = float(
            abs(old["finger_feedforward_Nm"] - fresh["finger_feedforward_Nm"]).max()
        )
        old["finger_feedforward_Nm"] = fresh["finger_feedforward_Nm"]
        np.savez_compressed(regenerated / "targets.npz", **old)
        receipt = dict(
            schema="v2d.native-force-clock-repair.v1",
            source_manifest_sha256=sha(inputs / "manifest.json"),
            source_proposal_sha256=sha(proposal),
            source_static_manifest_sha256=sha(static_manifest),
            source_task_builder_sha256=sha(
                Path(sys.modules[build.__module__].__file__)
            ),
            preparation_source_sha256=sha(Path(__file__)),
            previous_force_plan_speed=json.loads(
                (inputs / "task-grips.json").read_text()
            )["source_motion_speed"],
            force_plan_and_playback_speed=1.0,
            maximum_feedforward_change_Nm=delta,
            maximum_feedforward_Nm=float(abs(old["finger_feedforward_Nm"]).max()),
            regeneration_alignment_differences=differences,
            non_feedforward_target_arrays_preserved_exactly=True,
            source_observations_changed=False,
            physical_task_accepted=False,
        )
        (regenerated / "force-clock-repair.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        manifest = json.loads((regenerated / "manifest.json").read_text())
        manifest["force_clock_repair_receipt_sha256"] = sha(
            regenerated / "force-clock-repair.json"
        )
        manifest["non_feedforward_target_arrays_preserved_exactly"] = True
        manifest["files"] = {
            p.relative_to(regenerated).as_posix(): sha(p)
            for p in sorted(regenerated.rglob("*"))
            if p.is_file() and p != regenerated / "manifest.json"
        }
        (regenerated / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        shutil.move(regenerated, out)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "case", "proposal", "static-manifest", "urdfs", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(
        json.dumps(
            recompute(a.inputs, a.case, a.proposal, a.static_manifest, a.urdfs, a.out),
            indent=2,
        )
    )
