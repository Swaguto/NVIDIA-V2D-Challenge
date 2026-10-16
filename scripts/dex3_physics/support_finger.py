"""Add a load-bearing middle pad to a verified two-pad geometric proposal.

This searches robot commands on the saved cooked meshes. Neither fitted poses
nor the Coulomb load plan are simulator measurements or acceptance evidence.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from antipodal_seed import exposed_faces, collision_residual
from cooked_grasp_audit import convex_escape, minimum_distance, world_convexes
from grasp_preload import URDFKinematics
from grip_force import point_jacobian, solve_gravity_contacts


def refine_tripod(
    kin,
    root,
    q,
    local,
    shapes,
    links,
    middle_point,
    middle_normal,
    middle_anchor,
    initial_middle,
):
    """Keep the first two contacts while moving the wrist slightly for the third."""
    original = kin.poses(root, q)
    fixed = []
    for name in links[:2]:
        world = [
            v @ original[name][:3, :3].T + original[name][:3, 3] for v in local[name]
        ]
        _, witness = minimum_distance(world, shapes)
        point = np.array(witness["points_world_m"][0])
        _, normal, piece = max(
            ((*convex_escape(a, b), b) for a in world for b in shapes),
            key=lambda x: x[0],
        )
        anchor = original[name][:3, :3].T @ (point - original[name][:3, 3])
        fixed.append((name, anchor, point, normal, np.max(piece @ normal)))
    hand_collision = {
        name: [(v, ConvexHull(v).equations[:, :3]) for v in pieces]
        for name, pieces in local.items()
    }
    object_collision = [(v, ConvexHull(v).equations[:, :3]) for v in shapes]
    start = np.array([q[n] for n in kin.names])
    lower = np.r_[
        [-0.02] * 3,
        [-0.3] * 3,
        [
            max(kin.limits[n][0] + 0.08 - v, -1.2 if "_middle_" in n else -0.25)
            for n, v in zip(kin.names, start, strict=True)
        ],
    ]
    upper = np.r_[
        [0.02] * 3,
        [0.3] * 3,
        [
            min(kin.limits[n][1] - 0.08 - v, 1.2 if "_middle_" in n else 0.25)
            for n, v in zip(kin.names, start, strict=True)
        ],
    ]

    def decode(x):
        moved = np.r_[
            root[:3] + x[:3],
            (Rotation.from_rotvec(x[3:6]) * Rotation.from_quat(root[3:])).as_quat(),
        ]
        current = dict(zip(kin.names, start + x[6:], strict=True))
        return moved, current, kin.poses(moved, current)

    def residual(x):
        _, _, poses = decode(x)
        errors = []
        for name, anchor, point, normal, plane in fixed:
            pose = poses[name]
            actual = pose[:3, :3] @ anchor + pose[:3, 3]
            world = np.concatenate(local[name]) @ pose[:3, :3].T + pose[:3, 3]
            errors.extend((actual - point) / 0.003)
            errors.append(2 * (np.min(world @ normal) - plane + 0.00075) / 0.001)
        pose = poses[links[2]]
        world = np.concatenate(local[links[2]]) @ pose[:3, :3].T + pose[:3, 3]
        actual = pose[:3, :3] @ middle_anchor + pose[:3, 3]
        delta = actual - middle_point
        errors.append(
            2 * (np.min((world - middle_point) @ middle_normal) + 0.00075) / 0.001
        )
        errors.extend(0.2 * (delta - (delta @ middle_normal) * middle_normal) / 0.01)
        return np.r_[
            errors,
            2 * collision_residual(poses, hand_collision, object_collision, links),
            0.1 * x[:3] / 0.02,
            0.1 * x[3:6] / 0.3,
            0.02 * x[6:],
        ]

    initial = np.zeros(13)
    for name, value in initial_middle.items():
        initial[6 + kin.names.index(name)] = value - q[name]
    fit = least_squares(
        residual,
        np.clip(initial, lower + 1e-9, upper - 1e-9),
        bounds=(lower, upper),
        max_nfev=100,
    )
    return (*decode(fit.x), fit)


def add_middle(run, urdfs, proposal_path, sample_name):
    report = json.loads((run / "report.json").read_text())
    if (
        hashlib.sha256((run / "states.json").read_bytes()).hexdigest()
        != report["states_sha256"]
    ):
        raise ValueError("Source states changed")
    proposal = copy.deepcopy(json.loads(proposal_path.read_text()))
    states = json.loads((run / "states.json").read_text())["records"]
    colliders = json.loads((run / "cooked-colliders.json").read_text())["colliders"]
    record = next(r for r in proposal["results"] if r["sample"] == sample_name)
    source, candidate = record["source_sample"], record["candidate"]
    side = source["side"]
    kin = URDFKinematics(urdfs / f"dex3_{side}.urdf")
    index = next(
        i
        for i, r in enumerate(report["initial_pose_readback"])
        if r["sample"] == sample_name
    )
    row = next(r for r in states if r["step"] == 0 and r["sample_index"] == index)
    shapes = [
        v
        for c in colliders
        if c["asset"] == f"object-{index}"
        for v in world_convexes(c, row["object_root_state_stage_xyzw"][:7])
    ]
    local = {
        c["body_name"]: world_convexes(c, [0, 0, 0, 0, 0, 0, 1])
        for c in colliders
        if c["asset"] == f"hand-{index}"
    }
    root = np.r_[
        candidate["hand_position_stage_m"], candidate["hand_quaternion_stage_xyzw"]
    ]
    q = dict(zip(candidate["joint_names"], candidate["finger_target_rad"], strict=True))
    fk = kin.poses(root, q)
    links = [
        f"{side}_hand_{digit}_{distal}_link"
        for digit, distal in [("thumb", 2), ("index", 1), ("middle", 1)]
    ]
    middle = links[-1]
    vertices = np.concatenate(local[middle])
    center = vertices.mean(0) @ fk[middle][:3, :3].T + fk[middle][:3, 3]
    points, normals = exposed_faces(shapes, 0.01)
    distance = np.linalg.norm(points - center, axis=1)
    pool = np.flatnonzero(distance < 0.07)
    chosen = []
    top = pool[normals[pool, 2] > 0.5]
    other = pool[normals[pool, 2] <= 0.5]
    order = np.r_[
        top[np.argsort(distance[top], kind="stable")],
        other[np.argsort(distance[other], kind="stable")],
    ]
    for j in order:
        if all(np.linalg.norm(points[j] - points[k]) >= 0.004 for k in chosen):
            chosen.append(j)
        if len(chosen) == 48:
            break
    names = [n for n in kin.names if "_middle_" in n]
    start = np.array([q[n] for n in names])
    lower = [
        max(kin.limits[n][0] + 0.08, v - 1.2) for n, v in zip(names, start, strict=True)
    ]
    upper = [
        min(kin.limits[n][1] - 0.08, v + 1.2) for n, v in zip(names, start, strict=True)
    ]
    valid, rejected = [], {"fit": 0, "geometry": 0, "load": 0}
    closest_rejection = None
    trials = 0
    for j in chosen:
        trials += 1
        direction = fk[middle][:3, :3].T @ -normals[j]
        projection = vertices @ direction
        anchor = vertices[projection > projection.max() - 0.001].mean(0)
        target = points[j] - 0.00075 * normals[j]

        def positions(values):
            current = dict(q)
            current.update(dict(zip(names, values, strict=True)))
            return current, kin.poses(root, current)

        def residual(values):
            _, poses = positions(values)
            pose = poses[middle]
            world = vertices @ pose[:3, :3].T + pose[:3, 3]
            actual = pose[:3, :3] @ anchor + pose[:3, 3]
            clearance = np.min((world - points[j]) @ normals[j])
            delta = actual - target
            tangent = delta - (delta @ normals[j]) * normals[j]
            return np.r_[
                2 * (clearance + 0.00075) / 0.001,
                0.5 * tangent / 0.01,
                0.02 * (values - start),
            ]

        fit = least_squares(residual, start, bounds=(lower, upper), max_nfev=100)
        moved_root, current, poses, fit_all = refine_tripod(
            kin,
            root,
            q,
            local,
            shapes,
            links,
            points[j],
            normals[j],
            anchor,
            dict(zip(names, fit.x, strict=True)),
        )
        if not fit_all.success:
            rejected["fit"] += 1
            continue
        depths, escapes, hand = {}, {}, {}
        for name, pieces in local.items():
            hand[name] = [
                v @ poses[name][:3, :3].T + poses[name][:3, 3] for v in pieces
            ]
            depths[name], escapes[name] = max(
                (convex_escape(a, b) for a in hand[name] for b in shapes),
                key=lambda x: x[0],
            )
        if (
            min(depths[n] for n in links) < 0
            or max(depths.values()) > 0.0015
            or escapes[links[0]] @ escapes[links[1]] > -0.5
        ):
            rejected["geometry"] += 1
            score = max(0, max(depths.values()) - 0.0015) + max(0, -depths[middle])
            if closest_rejection is None or score < closest_rejection["score_m"]:
                closest_rejection = dict(
                    score_m=score,
                    piece_depths_m=depths,
                    middle_fingers_rad=fit.x.tolist(),
                )
            continue
        contact, inward, jac = [], [], []
        for name in links:
            _, witness = minimum_distance(hand[name], shapes)
            point = np.array(witness["points_world_m"][0])
            local_point = poses[name][:3, :3].T @ (point - poses[name][:3, 3])
            contact.append(point)
            inward.append(-escapes[name])
            jac.append(point_jacobian(kin, moved_root, current, name, local_point))
        try:
            _, torque, load_plan = solve_gravity_contacts(
                contact, inward, jac, row["object_root_state_stage_xyzw"][:3], 0.3
            )
        except ValueError:
            rejected["load"] += 1
            continue
        planned = copy.deepcopy(candidate)
        original = dict(
            zip(source["joint_names"], source["finger_target_rad"], strict=True)
        )
        planned.update(
            hand_position_stage_m=moved_root[:3].tolist(),
            hand_quaternion_stage_xyzw=moved_root[3:].tolist(),
            finger_target_rad=[float(current[n]) for n in candidate["joint_names"]],
            maximum_finger_delta_rad=max(
                abs(current[n] - original[n]) for n in current
            ),
            maximum_piece_penetration_lower_bound_m=max(depths.values()),
            three_pad_geometric_contacts=True,
            middle_contact_point_stage_m=points[j].tolist(),
            nominal_load_plan=load_plan,
            objective=float(np.linalg.norm(fit_all.fun)),
        )
        if planned["maximum_finger_delta_rad"] <= 1.2:
            valid.append(planned)
        if trials % 8 == 0:
            print(
                json.dumps(dict(trials=trials, valid=len(valid), rejected=rejected)),
                flush=True,
            )
        if len(valid) >= 4:
            break
    for other in proposal["results"]:
        other["candidate"] = None
    record["candidate"] = (
        min(
            valid,
            key=lambda c: (
                c["nominal_load_plan"]["maximum_motor_feedforward_Nm"],
                sum(c["nominal_load_plan"]["normal_forces_N"]),
            ),
        )
        if valid
        else None
    )
    record["candidates"] = valid
    record["middle_search"] = dict(
        surface_candidates=len(pool),
        fitted_candidates=trials,
        rejection_counts=rejected,
        geometrically_and_nominally_load_feasible=len(valid),
        closest_geometry_rejection=closest_rejection,
    )
    proposal["support_finger_plan"] = dict(
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_proposal_sha256=hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        physical_task_accepted=False,
    )
    return proposal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["run", "urdfs", "proposal", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sample", required=True)
    args = parser.parse_args()
    result = add_middle(args.run, args.urdfs, args.proposal, args.sample)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            [
                {k: r.get(k) for k in ["sample", "middle_search", "candidate"]}
                for r in result["results"]
            ],
            indent=2,
        )
    )
