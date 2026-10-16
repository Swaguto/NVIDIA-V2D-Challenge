"""Port a frozen, inferred grasp plan to full-episode PPO targets.

Source observations and confidence remain separate from robot commands. No
simulator readback or successful trajectory is used as a target. Physics and
checkpoint acceptance must be evaluated in the training runtime independently.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from v2d_dex3_flashchord.kinematics import URDFKinematics

sys.path.insert(0, str(Path(__file__).parents[1] / "reproduction"))
from _dex3_task_grip import TaskGripReference


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(base, case, task_spec, out):
    original = json.loads((base / "manifest.json").read_text())
    spec = json.loads(task_spec.read_text())
    if spec["case_manifest_sha256"] != sha(case / "file-manifest.json"):
        raise ValueError("Task reference is bound to a different reconstruction")
    declared = json.loads((case / "file-manifest.json").read_text())["files_sha256"]
    bundle = case / "bundle/episode.npz"
    if (
        sha(bundle) != declared["bundle/episode.npz"]
        or sha(bundle) != original["source_files_sha256"]["bundle/episode.npz"]
    ):
        raise ValueError("Training and grasp source bundles differ")
    for name, expected in original["files"].items():
        f = (base / name).resolve()
        if not f.is_relative_to(base.resolve()) or sha(f) != expected:
            raise ValueError(f"Base input member differs: {name}")
    with np.load(bundle, allow_pickle=False) as f:
        source = {k: f[k].copy() for k in f.files}
    times = source["timestamps_s"]
    if not np.array_equal(times, spec["source_times_s"]):
        raise ValueError("Task clock differs from original source")
    if "smooth_robot_object_reference" not in spec:
        raise ValueError("Need an explicit stage-aligned object reference")
    objects = np.stack(
        [
            spec["smooth_robot_object_reference"][str(i)]
            for i in range(len(original["objects"]))
        ],
        axis=1,
    )
    count = len(times)
    if (
        objects.shape != (count, len(original["objects"]), 7)
        or not np.isfinite(objects).all()
    ):
        raise ValueError("Invalid full-episode inferred object reference")
    q = np.zeros((count, 2, 14))
    effort = np.zeros((count, 2, 7))
    stiffness = np.zeros((count, 2, 7))
    frames = {}
    for si, side in enumerate(("left", "right")):
        phases = spec["phases"][side]
        object_ids = {p["object"] for p in phases}
        if len(object_ids) != 1:
            raise ValueError("Current task adapter requires one object per hand")
        oi = next(iter(object_ids))
        names = original["joint_names"][side]
        kin = URDFKinematics(base / f"robot/urdfs/g1/dex3_{side}.urdf")
        target_frames = []
        ref = TaskGripReference(spec)
        for i, time in enumerate(times):
            command = ref.command(side, float(time), objects[i, oi], commit=False)
            phase = next(p for p in phases if p["sample"] == command["phase"])
            order = [phase["joint_names"].index(n) for n in names]
            q[i, si] = np.r_[command["pose"], np.asarray(command["fingers"])[order]]
            effort[i, si] = np.asarray(command["effort"])[order]
            stiffness[i, si] = phase["stiffness_Nm_per_rad"]
            fk = kin.poses(command["pose"], dict(zip(names, q[i, si, 7:], strict=True)))
            target_frames.append(
                [fk[name][:3, 3] for name in original["frame_names"][side]]
            )
        frames[side] = np.asarray(target_frames)
    if (
        not all(np.isfinite(x).all() for x in (q, effort, stiffness))
        or abs(effort).max() > 0.55
        or not np.all((2 <= stiffness) & (stiffness <= 8))
    ):
        raise ValueError("Task motor commands exceed the preserved bounds")
    # The mask denotes source observation confidence, not validity of inferred commands.
    mask_key = "hand_metric_valid" if "hand_metric_valid" in source else "hand_valid"
    confidence = source[mask_key].all(axis=1) & source["hand_directly_observed"].all(
        axis=1
    )
    metadata = json.loads((case / "bundle/manifest.json").read_text())
    for obj in metadata["objects"]:
        key = f"object_{obj['index']}_valid"
        if key in source:
            confidence &= source[key]
        else:
            prov = metadata["object_observation_provenance"][obj["name"]]
            confidence &= np.array(
                [prov.get(str(int(i))) == "observed" for i in source["frame_indices"]]
            )
    if out.exists():
        raise FileExistsError(out)
    shutil.copytree(base, out)
    np.savez_compressed(
        out / "targets.npz",
        robot_qpos=q,
        object_poses=objects,
        left_frames=frames["left"],
        right_frames=frames["right"],
        confidence_valid=confidence,
        frame_indices=source["frame_indices"],
        timestamps_s=times,
        finger_feedforward_Nm=effort,
        finger_stiffness_Nm_per_rad=stiffness,
    )
    shutil.copyfile(task_spec, out / "task-grips.json")
    manifest = copy.deepcopy(original)
    manifest.update(
        schema="v2d.dex3-training-task.v3",
        target_contract="inferred_robot_grasp_commands.v1",
        source_rows_half_open=[0, count],
        source_frame_indices=source["frame_indices"].tolist(),
        selected_frame_count=count,
        source_frame_count=count,
        all_selected_targets_confident=bool(confidence.all()),
        observed_source_frame_count=int(confidence.sum()),
        allows_inferred_robot_targets=True,
        selection="Every original source frame; source observation masks preserved separately",
        target_preparation_source_sha256=sha(Path(__file__)),
        reference_planner_source_sha256=sha(
            Path(sys.modules[TaskGripReference.__module__].__file__)
        ),
        kinematics_source_sha256=sha(
            Path(sys.modules[URDFKinematics.__module__].__file__)
        ),
        base_manifest_sha256=sha(base / "manifest.json"),
        task_grip_spec_sha256=sha(task_spec),
        reactive_isaac_wrist_controller_transferred=False,
        motor_feedforward="Separate inferred motor targets; added inside the native capped finger drive",
        source_observations_changed=False,
        contact_labels_fabricated=False,
        physical_task_accepted=False,
        full_episode_covered=True,
        training_runtime_retention_verified=False,
    )
    manifest["files"] = {
        p.relative_to(out).as_posix(): sha(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p != out / "manifest.json"
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "case", "task-spec", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    m = prepare(a.base, a.case, a.task_spec, a.out)
    print(
        json.dumps(
            {k: v for k, v in m.items() if k not in {"files", "source_frame_indices"}},
            indent=2,
        )
    )
