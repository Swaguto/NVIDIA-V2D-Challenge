"""Plan bounded finger preload from recorded cooked geometry; preserve source targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

from cooked_grasp_audit import convex_escape, minimum_distance, world_convexes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dex3_training"))
from v2d_dex3_flashchord.kinematics import URDFKinematics


def plan(
    run,
    urdfs,
    *,
    preload_m=0.00075,
    max_joint_delta_rad=0.15,
    side="right",
    proposal=None,
    include_middle=False,
):
    if not 0 < preload_m <= 0.002 or not 0 < max_joint_delta_rad <= 0.25:
        raise ValueError("Preload bounds exceed the declared small-motion experiment")
    report = json.loads((run / "report.json").read_text())
    states_path = run / "states.json"
    if hashlib.sha256(states_path.read_bytes()).hexdigest() != report["states_sha256"]:
        raise ValueError("Recorded source states changed")
    states = json.loads(states_path.read_text())
    cooked = json.loads((run / "cooked-colliders.json").read_text())["colliders"]
    kin = URDFKinematics(urdfs / f"dex3_{side}.urdf")
    plans = []
    for i, initial in enumerate(report["initial_pose_readback"]):
        if not initial["sample"].startswith(side + "-"):
            continue
        row = next(
            r for r in states["records"] if r["step"] == 0 and r["sample_index"] == i
        )
        original = dict(
            zip(
                row["finger_names_sim_order"],
                row["finger_position_rad_sim_order"],
                strict=True,
            )
        )
        root_pose = initial["hand_body_link_pose_stage_xyzw"][kin.root_name]
        fk = kin.poses(root_pose, original)
        frame_error = max(
            np.max(np.abs(fk[name][:3, 3] - pose[:3]))
            for name, pose in initial["hand_body_link_pose_stage_xyzw"].items()
        )
        if frame_error > 2e-6:
            raise ValueError("URDF kinematics differ from recorded SDK link frames")
        if proposal is not None:
            candidate = next(
                r["candidate"]
                for r in proposal["results"]
                if r["sample"] == initial["sample"]
            )
            if candidate is None:
                raise ValueError(
                    "A predicted closed pose requires a complete geometric proposal"
                )
            original = dict(
                zip(
                    candidate["joint_names"],
                    candidate["finger_target_rad"],
                    strict=True,
                )
            )
            root_pose = np.r_[
                candidate["hand_position_stage_m"],
                candidate["hand_quaternion_stage_xyzw"],
            ]
            fk = kin.poses(root_pose, original)
        object_shapes = [
            v
            for c in cooked
            if c["asset"] == f"object-{i}"
            for v in world_convexes(c, row["object_root_state_stage_xyzw"][:7])
        ]
        anchors = []
        links = [f"{side}_hand_thumb_2_link", f"{side}_hand_index_1_link"]
        if include_middle:
            links.append(f"{side}_hand_middle_1_link")
        for link in links:
            collider = next(
                c
                for c in cooked
                if c["asset"] == f"hand-{i}" and c["body_name"] == link
            )
            from scipy.spatial.transform import Rotation

            pose = (
                initial["hand_body_link_pose_stage_xyzw"][link]
                if proposal is None
                else np.r_[
                    fk[link][:3, 3], Rotation.from_matrix(fk[link][:3, :3]).as_quat()
                ]
            )
            hand_shapes = world_convexes(collider, pose)
            gap, witness = minimum_distance(hand_shapes, object_shapes)
            if gap > 0.005:
                raise ValueError(
                    "Preload cannot repair a grasp with more than 5 mm initial pad gap"
                )
            points = np.asarray(witness["points_world_m"])
            shape = hand_shapes[witness["first_convex"]]
            # An overlap QP witness can lie anywhere inside both solids. Its
            # centroid direction is not a contact normal. Use the SAT escape
            # axis of the deepest overlapping cooked piece instead.
            overlaps = [convex_escape(shape, obj) for obj in object_shapes]
            depth, escape = max(overlaps, key=lambda pair: pair[0])
            if depth < 0:
                direction = points[1] - points[0]
                direction /= np.linalg.norm(direction)
            else:
                direction = -escape
            local_point = fk[link][:3, :3].T @ (points[0] - fk[link][:3, 3])
            anchors.append(
                {
                    "link": link,
                    "point_local_m": local_point.tolist(),
                    "direction_world": direction.tolist(),
                    "initial_gap_m": gap,
                    "target_world_m": (points[0] + preload_m * direction).tolist(),
                }
            )
        opposition = float(
            np.dot(anchors[0]["direction_world"], anchors[1]["direction_world"])
        )
        if opposition > -0.5:
            raise ValueError(
                f"{initial['sample']}: pad directions are not opposed "
                f"(dot={opposition:.3f}); replan acquisition before adding preload"
            )
        names = kin.names
        start = np.array([original[n] for n in names])
        lower = np.array(
            [
                max(kin.limits[n][0], start[j] - max_joint_delta_rad)
                for j, n in enumerate(names)
            ]
        )
        upper = np.array(
            [
                min(kin.limits[n][1], start[j] + max_joint_delta_rad)
                for j, n in enumerate(names)
            ]
        )

        def residual(q):
            poses = kin.poses(root_pose, dict(zip(names, q, strict=True)))
            errors = []
            for a in anchors:
                error = (
                    poses[a["link"]][:3, :3] @ a["point_local_m"]
                    + poses[a["link"]][:3, 3]
                    - a["target_world_m"]
                )
                normal = np.asarray(a["direction_world"])
                errors.append(float(error @ normal) / preload_m)
                errors.extend(0.05 * (error - (error @ normal) * normal) / preload_m)
            return np.r_[errors, 0.02 * (q - start) / max_joint_delta_rad]

        fit = least_squares(
            residual, start, bounds=(lower, upper), ftol=1e-11, xtol=1e-11, gtol=1e-11
        )
        error = np.abs(residual(fit.x)[np.arange(len(anchors)) * 4]) * preload_m
        if not fit.success or error.max() > preload_m * 0.4:
            raise ValueError(
                "Bounded finger motion could not reach the proposed pad preload"
            )
        plans.append(
            {
                "sample": initial["sample"],
                "joint_names": names,
                "source_finger_targets_rad": start.tolist(),
                "closed_root_pose_stage_xyzw": np.asarray(root_pose).tolist(),
                "closed_pose_is_fk_prediction": proposal is not None,
                "preload_finger_targets_rad": fit.x.tolist(),
                "maximum_joint_delta_rad": float(np.max(np.abs(fit.x - start))),
                "maximum_anchor_normal_fit_error_m": float(error.max()),
                "initial_pad_direction_dot": opposition,
                "sdk_fk_max_position_error_m": float(frame_error),
                "anchors": anchors,
            }
        )
    return {
        "schema": "v2d.dex3-preload-plan.v1",
        "source_static_report_sha256": hashlib.sha256(
            (run / "report.json").read_bytes()
        ).hexdigest(),
        "source_states_sha256": report["states_sha256"],
        "cooked_sha256": hashlib.sha256(
            (run / "cooked-colliders.json").read_bytes()
        ).hexdigest(),
        "urdf_sha256": hashlib.sha256(
            (urdfs / f"dex3_{side}.urdf").read_bytes()
        ).hexdigest(),
        "preload_m": preload_m,
        "joint_delta_cap_rad": max_joint_delta_rad,
        "side": side,
        "plans": plans,
        "source_observations_changed": False,
        "physical_task_accepted": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--urdfs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = plan(args.run, args.urdfs)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k != "anchors"} for r in result["plans"]],
            indent=2,
        )
    )
