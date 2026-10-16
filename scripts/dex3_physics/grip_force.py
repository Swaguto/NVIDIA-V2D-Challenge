"""Map bounded opposing pad preload to real finger motor torques.

The Jacobian transpose supplies squeeze effort in addition to position tracking.
This is an experimental motor command, not an external object wrench or a force
closure certificate. Observations and nominal motor limits stay unchanged.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial import ConvexHull

from cooked_grasp_audit import minimum_distance, world_convexes

from grasp_preload import URDFKinematics, plan


def point_jacobian(kin, root, positions, link, point, epsilon=1e-5):
    q = np.asarray([positions[n] for n in kin.names], dtype=float)
    columns = []
    for index in range(len(q)):
        step = np.zeros_like(q)
        step[index] = epsilon
        points = []
        for perturbed in (q + step, q - step):
            pose = kin.poses(root, dict(zip(kin.names, perturbed, strict=True)))[link]
            points.append(pose[:3, :3] @ point + pose[:3, 3])
        columns.append((points[0] - points[1]) / (2 * epsilon))
    return np.column_stack(columns)


def open_pregrasp(kin, root, closed, anchors, distance_m, clearance_terms=()):
    start = np.array([closed[n] for n in kin.names])
    initial = kin.poses(root, closed)
    targets = []
    for a in anchors:
        pose = initial[a["link"]]
        point = np.asarray(a["point_local_m"])
        targets.append(
            pose[:3, :3] @ point
            + pose[:3, 3]
            - distance_m * np.asarray(a["direction_world"])
        )

    def residual(q):
        poses = kin.poses(root, dict(zip(kin.names, q, strict=True)))
        errors = []
        for a, target in zip(anchors, targets, strict=True):
            pose = poses[a["link"]]
            delta = pose[:3, :3] @ a["point_local_m"] + pose[:3, 3] - target
            normal = np.asarray(a["direction_world"])
            errors.extend((delta @ normal / distance_m,))
            errors.extend(0.1 * (delta - (delta @ normal) * normal) / distance_m)
        return np.r_[errors, 0.01 * (q - start) / 0.4]

    def clearances(q):
        poses = kin.poses(root, dict(zip(kin.names, q, strict=True)))
        gaps = []
        for link, local, axes, obj_lower, obj_upper in clearance_terms:
            pose = poses[link]
            projected = np.einsum(
                "ij,kj->ik", local @ pose[:3, :3].T + pose[:3, 3], axes
            )
            separation = np.maximum(
                projected.min(0) - obj_upper, obj_lower - projected.max(0)
            ).max()
            gaps.append(separation)
        return np.asarray(gaps)

    lower = np.maximum(start - 0.4, [kin.limits[n][0] + 0.005 for n in kin.names])
    upper = np.minimum(start + 0.4, [kin.limits[n][1] - 0.005 for n in kin.names])
    fit = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=500,
    )
    if not fit.success:
        raise ValueError(f"Could not construct a bounded open pregrasp: {fit.message}")
    opened = fit.x
    if clearance_terms and np.min(clearances(opened)) < 0.002:
        corrected = minimize(
            lambda q: np.sum(residual(q) ** 2),
            opened,
            method="SLSQP",
            bounds=list(zip(lower, upper, strict=True)),
            constraints={
                "type": "ineq",
                "fun": lambda q: (clearances(q) - 0.002) / 0.001,
            },
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if np.min(clearances(corrected.x)) < 0.0018:
            raise ValueError(
                f"No collision-free bounded pregrasp: {corrected.message}, minimum axis clearance {np.min(clearances(corrected.x))}"
            )
        opened = corrected.x
    return dict(zip(kin.names, opened, strict=True))


def add_grip_effort(run, urdfs, seed_path, force_N=6.0, opening_m=0.0):
    if not 0 < force_N <= 8:
        raise ValueError("Pad force experiment must be in (0, 8] N")
    if not 0 <= opening_m <= 0.006:
        raise ValueError("Pregrasp opening must be in [0, 6] mm")
    result = copy.deepcopy(json.loads(seed_path.read_text()))
    report = json.loads((run / "report.json").read_text())
    planned = plan(run, urdfs)
    kin = URDFKinematics(urdfs / "dex3_right.urdf")
    controls = []
    states = json.loads((run / "states.json").read_text())["records"]
    colliders = json.loads((run / "cooked-colliders.json").read_text())["colliders"]
    for row in planned["plans"]:
        initial = next(
            x for x in report["initial_pose_readback"] if x["sample"] == row["sample"]
        )
        root = initial["hand_body_link_pose_stage_xyzw"][kin.root_name]
        q = dict(zip(row["joint_names"], row["source_finger_targets_rad"], strict=True))
        candidate = next(
            x["candidate"] for x in result["results"] if x["sample"] == row["sample"]
        )
        declared = dict(
            zip(candidate["joint_names"], candidate["finger_target_rad"], strict=True)
        )
        if any(abs(q[n] - declared[n]) > 2e-6 for n in kin.names):
            raise ValueError("Closed-state readback differs from the geometric seed")
        torque = np.zeros(len(kin.names))
        pads = []
        for anchor in row["anchors"]:
            jac = point_jacobian(
                kin, root, q, anchor["link"], np.asarray(anchor["point_local_m"])
            )
            direction = np.asarray(anchor["direction_world"])
            force = direction * force_N
            contribution = jac.T @ force
            torque += contribution
            pads.append(
                dict(
                    link=anchor["link"],
                    inward_direction_stage=direction.tolist(),
                    contact_point_link_m=anchor["point_local_m"],
                    normal_jacobian_m_per_rad=(jac.T @ direction).tolist(),
                )
            )
        if not np.isfinite(torque).all() or np.max(abs(torque)) > 0.55:
            raise ValueError("Squeeze feedforward exceeds the 0.55 Nm experiment cap")
        candidate["finger_feedforward_Nm"] = [
            float(torque[kin.names.index(n)]) for n in candidate["joint_names"]
        ]
        gaps = None
        if opening_m:
            index = report["initial_pose_readback"].index(initial)
            obj = next(
                x for x in states if x["sample_index"] == index and x["step"] == 0
            )
            object_shapes = [
                v
                for c in colliders
                if c["asset"] == f"object-{index}"
                for v in world_convexes(c, obj["object_root_state_stage_xyzw"][:7])
            ]
            terms = []
            closed_poses = kin.poses(root, q)
            for anchor in row["anchors"]:
                link = anchor["link"]
                collider = next(
                    c
                    for c in colliders
                    if c["asset"] == f"hand-{index}" and c["body_name"] == link
                )
                for local in world_convexes(collider, [0, 0, 0, 0, 0, 0, 1]):
                    pose = closed_poses[link]
                    world = local @ pose[:3, :3].T + pose[:3, 3]
                    for obj_shape in object_shapes:
                        aabb_gap = np.maximum(
                            np.maximum(
                                world.min(0) - obj_shape.max(0),
                                obj_shape.min(0) - world.max(0),
                            ),
                            0,
                        )
                        if np.linalg.norm(aabb_gap) > 0.03:
                            continue
                        axes = np.concatenate(
                            [
                                ConvexHull(obj_shape).equations[:, :3],
                                ConvexHull(world).equations[:, :3],
                            ]
                        )
                        axes = np.unique(np.round(axes, 8), axis=0)
                        projections = np.einsum("ij,kj->ik", obj_shape, axes)
                        terms.append(
                            (link, local, axes, projections.min(0), projections.max(0))
                        )
            opened = open_pregrasp(kin, root, q, row["anchors"], opening_m, terms)
            poses = kin.poses(root, opened)
            gaps = []
            for anchor in row["anchors"]:
                c = next(
                    c
                    for c in colliders
                    if c["asset"] == f"hand-{index}"
                    and c["body_name"] == anchor["link"]
                )
                pose = poses[anchor["link"]]
                from scipy.spatial.transform import Rotation

                moved = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
                gap, _ = minimum_distance(world_convexes(c, moved), object_shapes)
                gaps.append(gap)
            if min(gaps) < 0.0015:
                raise ValueError(
                    f"{row['sample']}: proposed pregrasp pad gaps {gaps} are below 1.5 mm"
                )
            candidate["finger_pregrasp_rad"] = [
                float(opened[n]) for n in candidate["joint_names"]
            ]
        controls.append(
            dict(
                sample=row["sample"],
                pads=pads,
                max_feedforward_Nm=float(np.max(abs(torque))),
                pregrasp_pad_solid_gaps_m=gaps,
                estimated_pad_force_from_original_0_75mm_overlap_N=[
                    float(
                        2.0
                        * 0.00075
                        / np.sum(np.square(p["normal_jacobian_m_per_rad"]))
                    )
                    for p in pads
                ],
                scope="linearized unsaturated spring estimate; not a measured contact force",
            )
        )
    result["grip_effort"] = dict(
        schema="v2d.dex3-jacobian-grip-effort.v1",
        pad_force_N=force_N,
        pregrasp_opening_m=opening_m,
        feedforward_cap_Nm=0.55,
        preserve_motor_effort_cap_Nm=0.76,
        planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_closed_report_sha256=hashlib.sha256(
            (run / "report.json").read_bytes()
        ).hexdigest(),
        source_seed_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        controls=controls,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--urdfs", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--pad-force-N", type=float, default=6.0)
    parser.add_argument("--pregrasp-open-mm", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = add_grip_effort(
        args.run, args.urdfs, args.seed, args.pad_force_N, args.pregrasp_open_mm / 1000
    )
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["grip_effort"], indent=2))
