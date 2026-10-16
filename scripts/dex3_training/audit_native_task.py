"""Audit measured native grasp retention; a stationary fallen object cannot pass.

Contacts are sampled once per control step, at its last physics substep. This
reports sampled support and relative pose, not continuous collision certification.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def phase_metrics(hand, objects, reference, pad_force):
    if (
        pad_force.ndim != 2
        or pad_force.shape != (len(hand), pad_force.shape[1])
        or pad_force.shape[1] == 0
    ):
        raise ValueError("Need measured required pad forces")
    if len(hand) < 3 or not all(
        np.isfinite(a).all() for a in (hand, objects, reference, pad_force)
    ):
        raise ValueError("Need finite complete phase readback")
    h = Rotation.from_quat(hand[:, 3:])
    o = Rotation.from_quat(objects[:, 3:])
    relative_position = h.inv().apply(objects[:, :3] - hand[:, :3])
    relative_rotation = h.inv() * o
    position = float(
        np.linalg.norm(relative_position - relative_position[0], axis=1).max()
    )
    rotation = float(
        np.rad2deg((relative_rotation[0].inv() * relative_rotation).magnitude()).max()
    )
    support = float(np.all(pad_force >= 0.5, axis=1).mean())
    lift = float(objects[:, 2].max() - objects[0, 2])
    reference_lift = float(reference[:, 2].max() - reference[0, 2])
    lift_pass = lift >= max(0, reference_lift - 0.01)
    passed = position <= 0.01 and rotation <= 5 and support >= 0.95 and lift_pass
    return dict(
        maximum_relative_position_drift_m=position,
        maximum_relative_rotation_drift_deg=rotation,
        all_required_pad_sampled_support_fraction=support,
        maximum_actual_lift_m=lift,
        maximum_reference_lift_m=reference_lift,
        reference_lift_within_10mm=lift_pass,
        relative_retention_10mm_5deg_with_sampled_support=passed,
        maximum_source_position_error_m=float(
            np.linalg.norm(objects[:, :3] - reference[:, :3], axis=1).max()
        ),
    )


def audit(run, inputs):
    report = json.loads((run / "report.json").read_text())
    manifest_path = inputs / "manifest.json"
    if (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != report["source_manifest_sha256"]
    ):
        raise ValueError("Input manifest differs from the simulated inputs")
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        path = (inputs / name).resolve()
        if (
            not path.is_relative_to(inputs.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Input bytes/path differ")
    spec = json.loads((inputs / "task-grips.json").read_text())
    with np.load(run / "native-state-readback.npz") as f:
        body = f["body_poses"]
    with np.load(run / "native-contacts-readback.npz") as f:
        contacts = {k: f[k].copy() for k in f.files}
    with np.load(inputs / "targets.npz") as f:
        target = f["object_poses"]
        source_times = f["timestamps_s"]
    labels = json.loads((run / "body-labels.json").read_text())
    worlds = report["metrics"]["world_count"]
    stride = len(labels) // worlds
    local_labels = labels[:stride]

    def body_id(name):
        choices = [
            i
            for i, label in enumerate(local_labels)
            if label == name or label.split("/")[-1] == name
        ]
        if len(choices) != 1:
            raise ValueError("Ambiguous/missing native body: " + name)
        return choices[0]

    fps = report["control_fps"]
    times = np.arange(len(body)) / fps + source_times[0]
    rows = []
    for side in ("left", "right"):
        hand_id = body_id(side + "_hand_palm_link")
        for phase in spec["phases"][side]:
            oi = phase["object"]
            object_id = body_id(manifest["objects"][oi]["name"] + "/object")
            names = [
                p["link"]
                for p in phase["pad_normal_jacobians"]
                if any(d in p["link"] for d in phase["required_digits"])
            ]
            required = [body_id(n) for n in names]
            if len(required) != len(phase["required_digits"]):
                raise ValueError("Required digit/pad mapping is incomplete")
            indices = np.flatnonzero(
                (times >= phase["close_s"] + 0.05 - 1e-8)
                & (times <= phase["release_start_s"] - 0.05 + 1e-8)
            )
            reference = np.stack(
                [
                    np.interp(times[indices], source_times, target[:, oi, axis])
                    for axis in range(3)
                ],
                axis=1,
            )
            # phase_metrics only needs reference translation; unit identity rotation.
            reference = np.column_stack(
                [reference, np.tile([0, 0, 0, 1], (len(reference), 1))]
            )
            for world in range(worlds):
                offset = world * stride
                forces = np.zeros((len(indices), len(required)))
                for fi, frame in enumerate(indices):
                    start, end = contacts["frame_offsets"][frame : frame + 2]
                    a = contacts["body0_ids"][start:end]
                    b = contacts["body1_ids"][start:end]
                    force = contacts["contact_force_w"][start:end]
                    for pi, pad in enumerate(required):
                        pair = ((a == pad + offset) & (b == object_id + offset)) | (
                            (b == pad + offset) & (a == object_id + offset)
                        )
                        forces[fi, pi] = np.linalg.norm(force[pair], axis=1).sum()
                rows.append(
                    dict(
                        side=side,
                        sample=phase["sample"],
                        world=world,
                        required_pad_names=names,
                        sampled_frames=len(indices),
                        start_source_time_s=float(times[indices[0]]),
                        end_source_time_s=float(times[indices[-1]]),
                        **phase_metrics(
                            body[indices, hand_id + offset],
                            body[indices, object_id + offset],
                            reference,
                            forces,
                        ),
                    )
                )
    return dict(
        schema="v2d.native-grasp-audit.v1",
        grasps=rows,
        all_grasps_pass=bool(rows)
        and all(r["relative_retention_10mm_5deg_with_sampled_support"] for r in rows),
        native_motor_caps_verified=report["native_motor_verification"][
            "native_capped_drive_verified"
        ],
        contact_scope="Last physics substep per control frame; no continuous contact certificate",
        physical_task_accepted=False,
        official_score_claimed=False,
        source_manifest_sha256=report["source_manifest_sha256"],
        source_report_sha256=hashlib.sha256(
            (run / "report.json").read_bytes()
        ).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("run", "inputs", "out"):
        p.add_argument("--" + arg, type=Path, required=True)
    a = p.parse_args()
    result = audit(a.run, a.inputs)
    with a.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2))
