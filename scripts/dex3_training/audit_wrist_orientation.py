"""Separate source-retarget, fitted-reference and measured wrist orientation.

Human and robot raw quaternion axes are not compared directly. The baseline is
an already calibrated Dex3 retarget, rotated into the simulation stage frame.
Unknown source frames remain excluded rather than filled.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def audit(inputs, case, stage_manifest, run):
    source_path = case / "bundle/episode.npz"
    retarget_path = case / "retarget/retarget.npz"
    target_path = inputs / "targets.npz"
    state_path = run / "native-state-readback.npz"
    manifest = json.loads((inputs / "manifest.json").read_text())
    original_manifest = json.loads((case / "retarget/manifest.json").read_text())
    stage_metadata = json.loads(stage_manifest.read_text())
    report = json.loads((run / "report.json").read_text())
    source, original, target, state = map(load, [source_path, retarget_path, target_path, state_path])
    if not (sha(source_path) == stage_metadata["source_bundle_payload_sha256"]
            == manifest["source_files_sha256"]["bundle/episode.npz"]
            == original_manifest["episode_bundle_payload_sha256"]):
        raise ValueError("Source, retarget and stage registration differ")
    if (sha(retarget_path) != original_manifest["payload"]["sha256"]
            or sha(target_path) != manifest["files"]["targets.npz"]
            or sha(inputs / "manifest.json") != report["source_manifest_sha256"]):
        raise ValueError("Payload or evaluation input hash differs")
    for data in [original, target]:
        if not (np.array_equal(data["frame_indices"], source["frame_indices"])
                and np.array_equal(data["timestamps_s"], source["timestamps_s"])):
            raise ValueError("Orientation comparison requires the same source clock")
    stage = Rotation.from_matrix(stage_metadata["stage_rotation_from_inferred_support_normal"])
    times = source["timestamps_s"]
    poses = state["body_poses"]
    labels = json.loads((run / "body-labels.json").read_text())
    worlds = report["metrics"]["world_count"]
    if len(poses) != len(times) or len(labels) % worlds:
        raise ValueError("Readback frame or body mapping differs")
    stride = len(labels) // worlds
    checks = []
    for index in range(len(manifest["objects"])):
        matrices = source[f"object_{index}_world"]
        mask = np.isfinite(matrices).all(axis=(1, 2))
        error = np.degrees(((stage * Rotation.from_matrix(matrices[mask, :3, :3])).inv()
                            * Rotation.from_quat(target["object_poses"][mask, index, 3:])).magnitude())
        checks.append(dict(object=index, median_deg=float(np.median(error)), maximum_deg=float(error.max())))
        if error.max() > 2.0001:
            raise ValueError("Stage correspondence exceeds the bounded object-reference rotation repair")
    carry = (times >= 2.4) & (times <= 11.5)
    rows = []
    for side_index, side in enumerate(["left", "right"]):
        mask = (original["robot_valid"][:, side_index]
                & source["hand_valid"][:, side_index]
                & source["hand_directly_observed"][:, side_index]
                & np.isfinite(original["robot_qpos"][:, side_index]).all(axis=1))
        baseline = stage * Rotation.from_quat(original["robot_qpos"][mask, side_index, 3:7])
        commanded = Rotation.from_quat(target["robot_qpos"][:, side_index, 3:7])
        reference_error = np.degrees((baseline.inv() * commanded[mask]).magnitude())
        body = labels[:stride].index(f"dex3_{side}/{side}_hand_palm_link")
        tracking = []
        for world in range(worlds):
            actual = Rotation.from_quat(poses[:, body + world * stride, 3:])
            error = np.degrees((commanded.inv() * actual).magnitude())
            tracking.append(dict(world=world, all_frames_median_deg=float(np.median(error)),
                                 all_frames_max_deg=float(error.max()),
                                 first_pot_carry_median_deg=float(np.median(error[carry])),
                                 first_pot_carry_max_deg=float(error[carry].max())))
        rows.append(dict(side=side, observed_frames=int(mask.sum()),
                         target_vs_source_aligned_robot_wrist_median_deg=float(np.median(reference_error)),
                         target_vs_source_aligned_robot_wrist_max_deg=float(reference_error.max()),
                         first_pot_carry_target_vs_source_aligned_robot_wrist_median_deg=float(np.median(reference_error[carry[mask]])),
                         baseline_retarget_wrist_orientation_error_median_deg=float(np.median(original["wrist_orientation_error_deg"][mask, side_index])),
                         actual_vs_training_target=tracking))
    paths = dict(source_bundle=source_path, source_retarget=retarget_path,
                 retarget_manifest=case / "retarget/manifest.json", targets=target_path,
                 input_manifest=inputs / "manifest.json", stage_manifest=stage_manifest,
                 state=state_path, report=run / "report.json", body_labels=run / "body-labels.json")
    return dict(schema="v2d.wrist-orientation-decomposition.v2", hands=rows,
                scope="e002; observed source frames; calibrated Dex3 retarget vs fitted commands vs measured palm; no raw human/robot quaternion comparison",
                source_frame_count=len(times), world_count=worlds,
                stage_rotation_from_inferred_support_normal=stage.as_matrix().tolist(),
                source_palm_calibration=original_manifest["palm_frame_calibration"],
                first_pot_carry_window_s=[2.4, 11.5],
                source_object_stage_correspondence=checks,
                input_hashes={name: sha(path) for name, path in paths.items()},
                source_sha256=sha(Path(__file__)), physical_task_accepted=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["inputs", "case", "stage-manifest", "run", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.inputs, args.case, args.stage_manifest, args.run)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["hands"], indent=2))
