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
from scipy.optimize import least_squares, minimize, linprog
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


def solve_gravity_contacts(
    points,
    inward,
    jacobians,
    com,
    mass_kg,
    normal_cap_N=10.0,
    friction=1.0,
    minimum_normal_N=2.0,
):
    """Plan motor forces under a point-contact Coulomb model; not a physical certificate."""
    points, inward, jacobians, com = [
        np.asarray(x, float) for x in (points, inward, jacobians, com)
    ]
    count = len(points)
    if (
        not 2 <= count <= 3
        or points.shape != (count, 3)
        or inward.shape != (count, 3)
        or jacobians.ndim != 3
        or jacobians.shape[:2] != (count, 3)
        or com.shape != (3,)
        or not all(np.isfinite(x).all() for x in (points, inward, jacobians, com))
    ):
        raise ValueError("Contact geometry and Jacobians must be finite and aligned")
    if (
        not 0 < mass_kg <= 1
        or not 0 < friction <= 1.25
        or not 2 <= normal_cap_N <= 10
        or not 2 <= minimum_normal_N <= normal_cap_N
        or not np.allclose(np.linalg.norm(inward, axis=1), 1, atol=1e-6)
    ):
        raise ValueError("Invalid nominal load or contact force bounds")
    wrench = np.zeros((6, 3 * count))
    inequalities, bounds = [], []
    for i, (point, axis) in enumerate(zip(points, inward, strict=True)):
        sl = slice(3 * i, 3 * i + 3)
        wrench[:3, sl] = np.eye(3)
        x, y, z = point - com
        wrench[3:, sl] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        for sign, bound in [(1, normal_cap_N), (-1, -minimum_normal_N)]:
            row = np.zeros(3 * count)
            row[sl] = sign * axis
            inequalities.append(row)
            bounds.append(bound)
        basis = np.eye(3)[np.argmin(abs(axis))]
        tangent = np.cross(axis, basis)
        tangent /= np.linalg.norm(tangent)
        other = np.cross(axis, tangent)
        # Inscribed friction octagon: the resultant tangential force remains
        # inside the circular cone, including halfway between facet normals.
        for angle in np.arange(8) * np.pi / 4:
            row = np.zeros(3 * count)
            row[sl] = (
                np.cos(angle) * tangent
                + np.sin(angle) * other
                - friction * np.cos(np.pi / 8) * axis
            )
            inequalities.append(row)
            bounds.append(0.0)
    torque_map = np.concatenate([jac.T for jac in jacobians], axis=1)
    inequalities.extend(torque_map)
    bounds.extend([0.55] * len(torque_map))
    inequalities.extend(-torque_map)
    bounds.extend([0.55] * len(torque_map))
    desired = np.array([0, 0, mass_kg * 9.81, 0, 0, 0])
    result = linprog(
        inward.ravel(),
        A_ub=np.asarray(inequalities),
        b_ub=np.asarray(bounds),
        A_eq=wrench,
        b_eq=desired,
        bounds=[(None, None)] * (3 * count),
        method="highs",
    )
    if not result.success:
        raise ValueError(
            "No contact load plan within the unchanged force/motor bounds: "
            + result.message
        )
    forces = result.x.reshape(count, 3)
    torque = torque_map @ result.x
    return (
        forces,
        torque,
        dict(
            nominal_mass_kg=mass_kg,
            friction_lower_bound=friction,
            minimum_normal_N=minimum_normal_N,
            normal_cap_N=normal_cap_N,
            normal_forces_N=np.einsum("ij,ij->i", forces, inward).tolist(),
            forces_stage_N=forces.tolist(),
            equilibrium_residual=(wrench.dot(result.x) - desired).tolist(),
            desired_object_wrench_stage=desired.tolist(),
            maximum_motor_feedforward_Nm=float(np.max(abs(torque))),
            scope="FK/cooked-geometry force proposal under a nominal point-contact friction model; not measured forces or force-closure proof",
        ),
    )


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


def add_grip_effort(
    run,
    urdfs,
    seed_path,
    force_N=6.0,
    opening_m=0.0,
    side="right",
    predict_closed_pose=False,
    finger_stiffness=None,
    load_balance=False,
):
    if not 0 < force_N <= 10:
        raise ValueError(
            "Pad force experiment must be in (0, 10] N; the 0.55 Nm feedforward cap still applies"
        )
    if not 0 <= opening_m <= 0.006:
        raise ValueError("Pregrasp opening must be in [0, 6] mm")
    if finger_stiffness is not None and not 2 <= finger_stiffness <= 8:
        raise ValueError("Experimental finger stiffness must be in [2, 8] Nm/rad")
    result = copy.deepcopy(json.loads(seed_path.read_text()))
    report = json.loads((run / "report.json").read_text())
    if predict_closed_pose:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduction"))
        from _dex3_static_load import apply_grasp_seeds

        apply_grasp_seeds([p["sample"] for p in report["provenance"]], result)
    planned = plan(
        run,
        urdfs,
        side=side,
        proposal=result if predict_closed_pose else None,
        include_middle=load_balance,
    )
    kin = URDFKinematics(urdfs / f"dex3_{side}.urdf")
    controls = []
    states = json.loads((run / "states.json").read_text())["records"]
    colliders = json.loads((run / "cooked-colliders.json").read_text())["colliders"]
    for row in planned["plans"]:
        initial = next(
            x for x in report["initial_pose_readback"] if x["sample"] == row["sample"]
        )
        root = row["closed_root_pose_stage_xyzw"]
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
        jacobians, contact_points, directions = [], [], []
        poses = kin.poses(root, q)
        for anchor in row["anchors"]:
            jac = point_jacobian(
                kin, root, q, anchor["link"], np.asarray(anchor["point_local_m"])
            )
            direction = np.asarray(anchor["direction_world"])
            force = direction * force_N
            contribution = jac.T @ force
            torque += contribution
            jacobians.append(jac)
            contact_points.append(
                poses[anchor["link"]][:3, :3] @ anchor["point_local_m"]
                + poses[anchor["link"]][:3, 3]
            )
            directions.append(direction)
            pads.append(
                dict(
                    link=anchor["link"],
                    inward_direction_stage=direction.tolist(),
                    contact_point_link_m=anchor["point_local_m"],
                    normal_jacobian_m_per_rad=(jac.T @ direction).tolist(),
                )
            )
        load_plan = None
        if load_balance:
            index = report["initial_pose_readback"].index(initial)
            obj = next(
                s for s in states if s["sample_index"] == index and s["step"] == 0
            )
            forces, torque, load_plan = solve_gravity_contacts(
                contact_points,
                directions,
                jacobians,
                obj["object_root_state_stage_xyzw"][:3],
                0.3,
                force_N,
            )
            for pad, force in zip(pads, forces, strict=True):
                pad["desired_force_stage_N"] = force.tolist()
            candidate["required_contact_digits"] = ["thumb", "index", "middle"]
        if not np.isfinite(torque).all() or np.max(abs(torque)) > 0.55:
            raise ValueError("Squeeze feedforward exceeds the 0.55 Nm experiment cap")
        candidate["finger_feedforward_Nm"] = [
            float(torque[kin.names.index(n)]) for n in candidate["joint_names"]
        ]
        if finger_stiffness is not None:
            candidate["finger_stiffness_Nm_per_rad"] = float(finger_stiffness)
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
                load_plan=load_plan,
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
        closed_pose_is_urdf_fk_prediction=predict_closed_pose,
        source_geometry_is_recorded_cooked_mesh=True,
        new_sdk_closed_pose_readback=not predict_closed_pose,
        finger_stiffness_override_Nm_per_rad=finger_stiffness,
        gravity_load_balanced=load_balance,
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
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--finger-stiffness", type=float)
    parser.add_argument(
        "--load-balance",
        action="store_true",
        help="Use three contacts to balance nominal gravity and its moment; --pad-force-N is the per-contact normal cap",
    )
    parser.add_argument(
        "--predict-closed-pose",
        action="store_true",
        help="Use explicit FK-predicted robot commands on recorded cooked geometry; no new SDK readback is claimed",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = add_grip_effort(
        args.run,
        args.urdfs,
        args.seed,
        args.pad_force_N,
        args.pregrasp_open_mm / 1000,
        args.side,
        args.predict_closed_pose,
        args.finger_stiffness,
        args.load_balance,
    )
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["grip_effort"], indent=2))
