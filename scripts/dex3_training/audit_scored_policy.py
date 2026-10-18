"""Audit full scored readbacks without prescribing a grasp or inventing contacts.

Errors are relative to our inferred reconstruction, not hidden ground truth.
Native contact samples cover only the last substep of each control frame.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from v2d_dex3_flashchord.sampling import integer_rate_factor, resample_targets


def tracking_errors(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if (
        actual.ndim != 4
        or actual.shape[-1] != 7
        or reference.shape != (actual.shape[0], actual.shape[2], 7)
    ):
        raise ValueError("Expected complete frame/world/object poses")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite physical rollout or reference")
    for poses in [actual, reference]:
        if not np.allclose(np.linalg.norm(poses[..., 3:], axis=-1), 1, atol=1e-4):
            raise ValueError("Invalid pose quaternion")
    position = np.linalg.norm(actual[..., :3] - reference[:, None, :, :3], axis=-1)
    goal = np.broadcast_to(reference[:, None, :, 3:], actual[..., 3:].shape)
    angle = np.rad2deg(
        (
            Rotation.from_quat(goal.reshape(-1, 4)).inv()
            * Rotation.from_quat(actual[..., 3:].reshape(-1, 4))
        ).magnitude()
    ).reshape(position.shape)
    return [
        dict(
            mean_position_error_m=float(position[..., i].mean()),
            p95_position_error_m=float(np.percentile(position[..., i], 95)),
            maximum_position_error_m=float(position[..., i].max()),
            mean_full_orientation_error_deg=float(angle[..., i].mean()),
            fraction_samples_over_10cm=float((position[..., i] > 0.1).mean()),
            per_world_mean_position_error_m=position[..., i].mean(axis=0).tolist(),
        )
        for i in range(actual.shape[2])
    ]


def audit(run, inputs):
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report = json.loads((run / "report.json").read_text())
    if (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != report["source_manifest_sha256"]
    ):
        raise ValueError("Readback input manifest differs")
    for relative, digest in manifest["files"].items():
        p = (inputs / relative).resolve()
        if (
            not p.is_relative_to(inputs.resolve())
            or hashlib.sha256(p.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Readback input bytes differ")
    if (
        report["virtual_object_assistance"] != 0
        or report["virtual_object_actuator_caps_zero"] is not True
        or report["evaluation_reset_stabilization_steps"] != 0
        or report["zero_assistance_control_steps_verified"] < report["frame_count"] - 1
    ):
        raise ValueError("Need completely unassisted evaluation")
    with np.load(run / "native-state-readback.npz", allow_pickle=False) as f:
        body = f["body_poses"]
    labels = json.loads((run / "body-labels.json").read_text())
    worlds = report["metrics"]["world_count"]
    if len(labels) % worlds or body.shape != (report["frame_count"], len(labels), 7):
        raise ValueError("Native world/body/frame axes differ")
    stride = len(labels) // worlds
    names = [o["name"] for o in manifest["objects"]]
    ids = []
    for name in names:
        matches = [
            i for i, label in enumerate(labels[:stride]) if label == name + "/object"
        ]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous native object: " + name)
        ids.append(matches[0])
    actual = body.reshape(len(body), worlds, stride, 7)[:, :, ids]
    with np.load(inputs / "targets.npz", allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    ratio = integer_rate_factor(report["source_fps"], report["control_fps"])
    if len(body) != (len(data["frame_indices"]) - 1) * ratio + 1:
        raise ValueError("Full original episode is not covered")
    reference = resample_targets(data, len(body))["object_poses"]
    rows = tracking_errors(actual, reference)
    for name, row in zip(names, rows):
        row["object"] = name
    with np.load(run / "native-contacts-readback.npz", allow_pickle=False) as f:
        gaps = f["signed_gap_m"]
        if not np.isfinite(gaps).all():
            raise ValueError("Nonfinite native contact gaps")
        object_ids = [i + world * stride for world in range(worlds) for i in ids]
        selected = np.isin(f["body0_ids"], object_ids) | np.isin(
            f["body1_ids"], object_ids
        )
        object_gaps = gaps[selected]
    return dict(
        schema="v2d.scored-policy-native-audit.v1",
        objects=rows,
        full_source_frame_axis_covered=True,
        unassisted_steps_verified=report["zero_assistance_control_steps_verified"],
        native_object_contact_samples=int(selected.sum()),
        maximum_sampled_native_object_penetration_m=float(max(0, -object_gaps.min()))
        if len(object_gaps)
        else None,
        contact_scope="Native collision-shape gaps at the final substep only; not continuous contact or a grip certificate",
        orientation_scope="Full quaternion diagnostic; official evaluator applies object symmetries separately",
        native_motor_verification=report["native_motor_verification"],
        reference="Our inferred reconstruction; not hidden ground truth",
        source_manifest_sha256=report["source_manifest_sha256"],
        physical_task_accepted=False,
        official_leaderboard_score_claimed=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["run", "inputs", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = audit(a.run, a.inputs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
