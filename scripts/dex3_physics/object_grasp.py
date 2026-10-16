"""Reuse a geometric robot grasp on the same rigid object across source phases.

Only robot commands change. Each destination is independently checked against
its recorded cooked object and hand geometry before a new physics test.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cooked_grasp_audit import convex_escape, minimum_distance, world_convexes
from grasp_preload import URDFKinematics
from antipodal_seed import select_grasp


def transfer_pose(hand_pose, from_object_pose, to_object_pose):
    hand, before, after = (
        np.asarray(x, dtype=float)
        for x in (hand_pose, from_object_pose, to_object_pose)
    )
    if any(
        x.shape != (7,)
        or not np.isfinite(x).all()
        or abs(np.linalg.norm(x[3:]) - 1) > 1e-6
        for x in (hand, before, after)
    ):
        raise ValueError("Rigid grasp transfer needs finite unit xyzw poses")
    delta = Rotation.from_quat(after[3:]) * Rotation.from_quat(before[3:]).inv()
    return np.r_[
        delta.apply(hand[:3] - before[:3]) + after[:3],
        (delta * Rotation.from_quat(hand[3:])).as_quat(),
    ]


def seat_transferred_pads(kin, root, q, colliders, objects, links):
    """Seat submillimetre cooking differences with <=0.03 rad finger changes."""
    poses = kin.poses(root, q)
    anchors = []
    for name in links:
        c = colliders[name]
        pose = poses[name]
        world = world_convexes(
            c, np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
        )
        gap, witness = minimum_distance(world, objects)
        if gap > 0.0005:
            raise ValueError("Transferred pad gap exceeds the small seating correction")
        points = np.array(witness["points_world_m"])
        depth, escape = max(
            (convex_escape(a, b) for a in world for b in objects), key=lambda x: x[0]
        )
        inward = (points[1] - points[0]) / gap if depth < 0 and gap > 1e-8 else -escape
        anchor = pose[:3, :3].T @ (points[0] - pose[:3, 3])
        anchors.append((name, anchor, inward, points[0] + 0.00035 * inward))
    start = np.array([q[n] for n in kin.names])
    lower = [
        max(kin.limits[n][0] + 0.005, v - 0.03)
        for n, v in zip(kin.names, start, strict=True)
    ]
    upper = [
        min(kin.limits[n][1] - 0.005, v + 0.03)
        for n, v in zip(kin.names, start, strict=True)
    ]

    def residual(values):
        poses = kin.poses(root, dict(zip(kin.names, values, strict=True)))
        errors = []
        for name, anchor, inward, target in anchors:
            pose = poses[name]
            delta = pose[:3, :3] @ anchor + pose[:3, 3] - target
            errors.append(delta @ inward / 0.00035)
            errors.extend(0.1 * (delta - (delta @ inward) * inward) / 0.003)
        return np.r_[errors, 0.01 * (values - start) / 0.03]

    fit = least_squares(residual, start, bounds=(lower, upper), max_nfev=150)
    if not fit.success or np.max(abs(residual(fit.x)[np.arange(len(links)) * 4])) > 0.5:
        raise ValueError("Bounded destination pad seating failed")
    return dict(zip(kin.names, fit.x, strict=True))


def reuse(run, urdfs, proposal_path, prefer_low_lever=False, translation_cap_m=0.15):
    if not 0 < translation_cap_m <= 0.2:
        raise ValueError("Object-relative robot wrist correction must be in (0, 0.2] m")
    report = json.loads((run / "report.json").read_text())
    if report.get("controlled_acquisition", False):
        raise ValueError("Grasp transfer requires unpaired source phase fixtures")
    states_file = run / "states.json"
    if hashlib.sha256(states_file.read_bytes()).hexdigest() != report["states_sha256"]:
        raise ValueError("Recorded static states changed")
    proposal = copy.deepcopy(json.loads(proposal_path.read_text()))
    known = {r["sample"] for r in proposal["results"]}
    groups = {
        (r["source_sample"]["side"], r["source_sample"]["object"])
        for r in proposal["results"]
    }
    order = {}
    for i, provenance in enumerate(report["provenance"]):
        source = provenance["sample"]
        order[source["prefix"]] = i
        if (source["side"], source["object"]) in groups and source[
            "prefix"
        ] not in known:
            proposal["results"].append(
                dict(
                    sample=source["prefix"],
                    source_sample=copy.deepcopy(source),
                    candidate=None,
                )
            )
    proposal["results"].sort(key=lambda r: order[r["sample"]])
    if prefer_low_lever:
        for record in proposal["results"]:
            if record.get("candidates"):
                record["candidate"] = select_grasp(record["candidates"], "load")
                record["selection"] = (
                    "minimum gravity lever among candidates with normal dot <= -0.8"
                )
    if (
        proposal["physical_task_accepted"]
        or proposal["source_states_sha256"] != report["states_sha256"]
    ):
        raise ValueError("Need unaccepted proposals bound to this source run")
    colliders = json.loads((run / "cooked-colliders.json").read_text())["colliders"]
    states = json.loads(states_file.read_text())["records"]
    originals = copy.deepcopy(proposal["results"])
    for record in proposal["results"]:
        source = record["source_sample"]
        if record["candidate"] is not None:
            c = record["candidate"]
            c["root_translation_delta_m"] = (
                np.array(c["hand_position_stage_m"]) - source["hand_position_stage_m"]
            ).tolist()
            c["root_rotation_delta_rad"] = (
                (
                    Rotation.from_quat(c["hand_quaternion_stage_xyzw"])
                    * Rotation.from_quat(source["hand_quaternion_stage_xyzw"]).inv()
                )
                .as_rotvec()
                .tolist()
            )
            continue
        donors, seen = [], set()
        for r in originals:
            if (
                r["source_sample"]["side"] != source["side"]
                or r["source_sample"]["object"] != source["object"]
            ):
                continue
            for command in [r["candidate"], *r.get("candidates", [])]:
                if command is None:
                    continue
                signature = json.dumps(
                    [
                        command["hand_position_stage_m"],
                        command["hand_quaternion_stage_xyzw"],
                        command["finger_target_rad"],
                    ]
                )
                if signature in seen:
                    continue
                seen.add(signature)
                donors.append(
                    dict(
                        sample=r["sample"],
                        source_sample=r["source_sample"],
                        candidate=command,
                    )
                )
        if not donors:
            continue
        kin = URDFKinematics(urdfs / f"dex3_{source['side']}.urdf")
        index = next(
            i
            for i, initial in enumerate(report["initial_pose_readback"])
            if initial["sample"] == record["sample"]
        )
        row = next(r for r in states if r["step"] == 0 and r["sample_index"] == index)
        obj = [
            v
            for c in colliders
            if c["asset"] == f"object-{index}"
            for v in world_convexes(c, row["object_root_state_stage_xyzw"][:7])
        ]
        candidates = []
        for donor in donors:
            before = donor["source_sample"]
            if (
                abs(
                    before["object_scale_at_selected_frame"]
                    / source["object_scale_at_selected_frame"]
                    - 1
                )
                > 1e-6
            ):
                continue
            candidate = copy.deepcopy(donor["candidate"])
            pose = transfer_pose(
                np.r_[
                    candidate["hand_position_stage_m"],
                    candidate["hand_quaternion_stage_xyzw"],
                ],
                np.r_[
                    before["object_position_stage_m"],
                    before["object_quaternion_stage_xyzw"],
                ],
                np.r_[
                    source["object_position_stage_m"],
                    source["object_quaternion_stage_xyzw"],
                ],
            )
            q = dict(
                zip(
                    candidate["joint_names"],
                    candidate["finger_target_rad"],
                    strict=True,
                )
            )
            original_q = dict(
                zip(source["joint_names"], source["finger_target_rad"], strict=True)
            )
            translation = pose[:3] - source["hand_position_stage_m"]
            rotation = (
                Rotation.from_quat(pose[3:])
                * Rotation.from_quat(source["hand_quaternion_stage_xyzw"]).inv()
            ).as_rotvec()
            finger_delta = max(abs(q[n] - original_q[n]) for n in q)
            if (
                np.max(abs(translation)) > translation_cap_m
                or finger_delta > 1.2
                or any(
                    not kin.limits[n][0] + 0.005 <= q[n] <= kin.limits[n][1] - 0.005
                    for n in q
                )
            ):
                continue
            poses = kin.poses(pose, q)
            links = [
                f"{source['side']}_hand_{digit}_{distal}_link"
                for digit, distal in [("thumb", 2), ("index", 1)]
            ]
            if candidate.get("three_pad_geometric_contacts"):
                links.append(f"{source['side']}_hand_middle_1_link")
            local_colliders = {
                c["body_name"]: c for c in colliders if c["asset"] == f"hand-{index}"
            }
            if candidate.get("three_pad_geometric_contacts"):
                try:
                    q = seat_transferred_pads(kin, pose, q, local_colliders, obj, links)
                except ValueError:
                    continue
                poses = kin.poses(pose, q)
                candidate["finger_target_rad"] = [
                    float(q[n]) for n in candidate["joint_names"]
                ]
                candidate["destination_seating_joint_cap_rad"] = 0.03
            depths, escapes = {}, {}
            for c in colliders:
                if c["asset"] != f"hand-{index}":
                    continue
                name = c["body_name"]
                body = poses[name]
                pieces = world_convexes(
                    c, np.r_[body[:3, 3], Rotation.from_matrix(body[:3, :3]).as_quat()]
                )
                depth, escape = max(
                    (convex_escape(a, b) for a in pieces for b in obj),
                    key=lambda pair: pair[0],
                )
                depths[name], escapes[name] = depth, escape
            links = [
                f"{source['side']}_hand_{digit}_{distal}_link"
                for digit, distal in [("thumb", 2), ("index", 1)]
            ]
            if candidate.get("three_pad_geometric_contacts"):
                links.append(f"{source['side']}_hand_middle_1_link")
            dot = float(escapes[links[0]] @ escapes[links[1]])
            if (
                min(depths[n] for n in links) < 0
                or max(depths.values()) > 0.0015
                or dot > -0.5
            ):
                continue
            candidate.update(
                hand_position_stage_m=pose[:3].tolist(),
                hand_quaternion_stage_xyzw=pose[3:].tolist(),
                root_translation_delta_m=translation.tolist(),
                root_rotation_delta_rad=rotation.tolist(),
                maximum_finger_delta_rad=max(abs(q[n] - original_q[n]) for n in q),
                pad_escape_direction_dot=dot,
                maximum_piece_penetration_lower_bound_m=max(depths.values()),
                donor_sample=donor["sample"],
                transferred_object_relative_grasp=True,
                destination_piece_depths_m=depths,
                donor_candidate_sha256=hashlib.sha256(
                    json.dumps(donor["candidate"], sort_keys=True).encode()
                ).hexdigest(),
            )
            if "nominal_load_plan" in candidate:
                candidate["donor_nominal_load_plan"] = candidate.pop(
                    "nominal_load_plan"
                )
            if "object_contact_points_stage_m" in candidate:
                delta = (
                    Rotation.from_quat(source["object_quaternion_stage_xyzw"])
                    * Rotation.from_quat(before["object_quaternion_stage_xyzw"]).inv()
                )
                candidate["object_contact_points_stage_m"] = (
                    delta.apply(
                        np.asarray(candidate["object_contact_points_stage_m"])
                        - before["object_position_stage_m"]
                    )
                    + source["object_position_stage_m"]
                ).tolist()
            candidates.append(candidate)
        if candidates:
            record["candidate"] = min(
                candidates,
                key=lambda c: (
                    c.get("donor_nominal_load_plan", {}).get(
                        "maximum_motor_feedforward_Nm", 0
                    ),
                    c.get("gravity_lever_m", float("inf")),
                    np.linalg.norm(c["root_translation_delta_m"]),
                ),
            )
            record["geometrically_valid_transferred_candidates"] = len(candidates)
    proposal.update(
        wrist_translation_cap_m=translation_cap_m,
        wrist_rotation_cap_rad=float(np.pi),
        finger_delta_cap_rad=1.2,
        object_relative_reuse=dict(
            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_proposal_sha256=hashlib.sha256(
                proposal_path.read_bytes()
            ).hexdigest(),
            destination_cooked_geometry_independently_checked=True,
        ),
        physical_task_accepted=False,
    )
    return proposal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["run", "urdfs", "proposal", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prefer-low-lever", action="store_true")
    parser.add_argument("--wrist-translation-cap-m", type=float, default=0.15)
    args = parser.parse_args()
    result = reuse(
        args.run,
        args.urdfs,
        args.proposal,
        args.prefer_low_lever,
        args.wrist_translation_cap_m,
    )
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            [
                {
                    k: r.get(k)
                    for k in [
                        "sample",
                        "geometrically_valid_transferred_candidates",
                        "candidate",
                    ]
                }
                for r in result["results"]
            ],
            indent=2,
        )
    )
