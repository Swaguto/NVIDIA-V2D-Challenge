"""Decide the narrow native reference-penetration gate from complete audits.

Passing does not establish grasp stability, contact forces, velocity feasibility
or physical task success. No references or evaluator criteria are changed.
"""

import hashlib
import math


def assess(manifest_path, manifest, object_report, hand_report):
    axis = manifest["source_frame_indices"]
    if not axis or axis != list(range(len(axis))):
        raise ValueError("Contact gate needs the complete ordered source clock")
    expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    names = [o["name"] for o in manifest["objects"]]
    if len(set(names)) != len(names) or not names:
        raise ValueError("Contact gate needs distinct ordered object slots")
    if object_report.get("objects") != names:
        raise ValueError("Object contact audit uses different object slots")
    for report, schema in (
        (object_report, "v2d.native-reference-object-contact-audit.v1"),
        (hand_report, "v2d.native-reference-hand-contact-audit.v1"),
    ):
        if (
            report.get("schema") != schema
            or report.get("input_manifest_sha256") != expected
            or report.get("reference_knots") != len(axis)
            or report.get("physics_stepped") is not False
            or report.get("source_targets_modified") is not False
            or report.get("native_mesh_checks_relaxed") is not False
            or report.get("reporting_tolerance_m") != 0.001
            or [f.get("reference_frame") for f in report.get("frames", [])] != axis
        ):
            raise ValueError("Contact audit lost its exact full-reference contract")
    if (
        object_report.get("reference_object_poses_verified") is not True
        or hand_report.get("reference_robot_and_object_poses_verified") is not True
        or hand_report.get("all_native_hand_body_fk_verified") is not True
        or hand_report.get("frozen_source_closure_verified") is not True
        or not object_report.get("runtime_commit")
        or object_report["runtime_commit"] != hand_report.get("runtime_commit")
    ):
        raise ValueError(
            "Need matching native runtimes, exact poses and verified hand FK"
        )

    def number(value, nonnegative=False):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (nonnegative and value < 0)
        ):
            raise ValueError("Native contact distance must be finite and valid")
        return value

    object_depths, hand_depths = [], []
    object_knots, hand_slots = set(), set()
    for frame in object_report["frames"]:
        for contact in frame["contact_pairs"]:
            pair = contact["bodies"]
            if len(pair) != 2 or any(n not in [*names, "support"] for n in pair):
                raise ValueError("Unknown native object contact pair")
            depth = number(contact["maximum_penetration_m"], nonnegative=True)
            object_depths.append(depth)
            if depth > 0.001:
                object_knots.add(frame["reference_frame"])
    for frame in hand_report["frames"]:
        for contact in frame["contacts"]:
            if contact["hand"] not in ("left", "right") or contact["target"] not in [
                *names,
                "support",
            ]:
                raise ValueError("Unknown native hand contact pair")
            depth = max(0, -number(contact["signed_gap_m"]))
            hand_depths.append(depth)
            if depth > 0.001:
                hand_slots.add((frame["reference_frame"], contact["hand"]))
    return dict(
        schema="v2d.reference-native-penetration-gate.v1",
        input_manifest_sha256=expected,
        reference_knots=len(axis),
        tolerance_m=0.001,
        object_knots_over_tolerance=len(object_knots),
        hand_slots_over_tolerance=len(hand_slots),
        maximum_object_penetration_m=max(object_depths, default=0),
        maximum_hand_penetration_m=max(hand_depths, default=0),
        native_penetration_gate_passed=not object_knots and not hand_slots,
        contact_distances_recomputed_from_every_audit_frame=True,
        source_targets_modified=False,
        contact_forces_or_grasp_stability_assessed=False,
        swept_contacts_assessed=False,
        physical_task_accepted=False,
    )
