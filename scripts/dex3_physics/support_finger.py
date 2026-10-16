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
from grasp_roll import roll_about_contacts


def support_patch(vertices, world_rotation, object_normal):
    """Choose a real pad face, rather than a single extreme mesh vertex.

    Return its centroid, outward normal and area. Coplanar triangles are joined
    so triangulation does not change the face ranking. The fitter must rotate
    this face toward the object; a point touching the requested plane alone
    does not establish a supporting contact normal.
    """
    vertices = np.asarray(vertices, float)
    hull = ConvexHull(vertices)
    faces = {}
    for tri, equation in zip(hull.simplices, hull.equations, strict=True):
        key = tuple(np.round(equation, 7))
        normal = equation[:3]
        points = vertices[tri]
        area = (
            np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) / 2
        )
        if key not in faces:
            faces[key] = [np.zeros(3), normal, 0.0]
        faces[key][0] += area * points.mean(0)
        faces[key][2] += area
    candidates = [
        (center / area, normal, area) for center, normal, area in faces.values()
    ]
    desired = -np.asarray(object_normal)
    # Prefer an already reachable, broad face; area breaks near-alignment ties.
    return max(
        candidates,
        key=lambda f: float((world_rotation @ f[1]) @ desired) + 0.05 * np.log(f[2]),
    )


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
    joint_reference=None,
    support_height_stage_m=None,
    middle_patch_normal=None,
    pinch_targets=None,
):
    """Keep the first two contacts while moving the wrist slightly for the third."""
    original = kin.poses(root, q)
    fixed = []
    for index, name in enumerate(links[:2]):
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
        patch_normal = None
        plane = np.max(piece @ normal)
        if pinch_targets is not None:
            point, normal, plane = pinch_targets[index]
            anchor, patch_normal, _area = support_patch(
                np.concatenate(local[name]), original[name][:3, :3], normal
            )
        fixed.append((name, anchor, point, normal, plane, patch_normal))
    hand_collision = {
        name: [(v, ConvexHull(v).equations[:, :3]) for v in pieces]
        for name, pieces in local.items()
    }
    object_collision = [(v, ConvexHull(v).equations[:, :3]) for v in shapes]
    start = np.array([q[n] for n in kin.names])
    translation_limit = 0.04 if joint_reference is not None else 0.02
    lower = np.r_[
        [-translation_limit] * 3,
        [-0.3] * 3,
        [
            max(kin.limits[n][0] + 0.08 - v, -1.2 if "_middle_" in n else -0.25)
            for n, v in zip(kin.names, start, strict=True)
        ],
    ]
    upper = np.r_[
        [translation_limit] * 3,
        [0.3] * 3,
        [
            min(kin.limits[n][1] - 0.08 - v, 1.2 if "_middle_" in n else 0.25)
            for n, v in zip(kin.names, start, strict=True)
        ],
    ]
    if joint_reference is not None:
        lower[6:] = [
            max(kin.limits[n][0] + 0.08, joint_reference[n] - 1.2) - q[n]
            for n in kin.names
        ]
        upper[6:] = [
            min(kin.limits[n][1] - 0.08, joint_reference[n] + 1.2) - q[n]
            for n in kin.names
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
        for name, anchor, point, normal, plane, patch_normal in fixed:
            pose = poses[name]
            actual = pose[:3, :3] @ anchor + pose[:3, 3]
            world = np.concatenate(local[name]) @ pose[:3, :3].T + pose[:3, 3]
            errors.extend((actual - point) / 0.003)
            errors.append(2 * (np.min(world @ normal) - plane + 0.00075) / 0.001)
            if patch_normal is not None:
                errors.extend((pose[:3, :3] @ patch_normal + normal) / 0.2)
        pose = poses[links[2]]
        world = np.concatenate(local[links[2]]) @ pose[:3, :3].T + pose[:3, 3]
        actual = pose[:3, :3] @ middle_anchor + pose[:3, 3]
        delta = actual - middle_point
        errors.append(
            2 * (np.min((world - middle_point) @ middle_normal) + 0.00075) / 0.001
        )
        tangent_scale = 1 / 0.003 if joint_reference is not None else 0.2 / 0.01
        errors.extend(tangent_scale * (delta - (delta @ middle_normal) * middle_normal))
        if middle_patch_normal is not None:
            # An aligned pad face is needed for a stable area contact. Without
            # this term the optimizer can touch an underside target using a
            # side edge, while the actual SAT/contact normal stays horizontal.
            errors.extend((pose[:3, :3] @ middle_patch_normal + middle_normal) / 0.15)
        if support_height_stage_m is not None:
            bottom = min(
                np.min(np.einsum("ij,j->i", v, poses[name][:3, :3][2]))
                + poses[name][2, 3]
                for name, pieces in local.items()
                for v in pieces
            )
            errors.append(2 * max(0, support_height_stage_m + 0.0015 - bottom) / 0.001)
        return np.r_[
            errors,
            2 * collision_residual(poses, hand_collision, object_collision, links),
            0.1 * x[:3] / translation_limit,
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
        max_nfev=200 if joint_reference is not None else 100,
    )
    return (*decode(fit.x), fit)


def add_middle(
    run,
    urdfs,
    proposal_path,
    sample_name,
    *,
    wrist_roll_deg=0.0,
    support_from_below=False,
    support_height_stage_m=None,
    maximum_candidates=48,
    reconfigure_fingers=False,
    orient_support_patch=False,
    side_support=False,
    orient_pinch_patches=False,
):
    if (
        not np.isfinite(wrist_roll_deg)
        or not -180 <= wrist_roll_deg <= 180
        or not isinstance(support_from_below, bool)
        or not isinstance(reconfigure_fingers, bool)
        or not isinstance(orient_support_patch, bool)
        or not isinstance(side_support, bool)
        or not isinstance(orient_pinch_patches, bool)
        or (orient_pinch_patches and not orient_support_patch)
        or (side_support and support_from_below)
        or not 1 <= maximum_candidates <= 48
        or (
            support_height_stage_m is not None
            and not np.isfinite(support_height_stage_m)
        )
    ):
        raise ValueError("Invalid support-grasp search configuration")
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
    pinch_targets = []
    if wrist_roll_deg or support_from_below or orient_support_patch:
        contacts = []
        for name in links[:2]:
            world = [v @ fk[name][:3, :3].T + fk[name][:3, 3] for v in local[name]]
            _, witness = minimum_distance(world, shapes)
            contacts.append(witness["points_world_m"][0])
            _, normal, piece = max(
                ((*convex_escape(a, b), b) for a in world for b in shapes),
                key=lambda x: x[0],
            )
            pinch_targets.append(
                (np.array(witness["points_world_m"][0]), normal, np.max(piece @ normal))
            )
        if wrist_roll_deg:
            root = roll_about_contacts(root, contacts, wrist_roll_deg)
            fk = kin.poses(root, q)
    middle = links[-1]
    vertices = np.concatenate(local[middle])
    center = vertices.mean(0) @ fk[middle][:3, :3].T + fk[middle][:3, 3]
    points, normals = exposed_faces(shapes, 0.01)
    distance = np.linalg.norm(points - center, axis=1)
    pool = np.flatnonzero(distance < 0.07)
    if support_from_below:
        # Outward-down object faces admit an upward force from the middle pad.
        pool = pool[normals[pool, 2] < -0.5]
        distance = distance + 0.5 * np.linalg.norm(
            points - np.mean(contacts, axis=0), axis=1
        )
    if side_support:
        pool = pool[np.abs(normals[pool, 2]) <= 0.35]
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
        if len(chosen) == maximum_candidates:
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
        patch_normal = None
        if orient_support_patch:
            anchor, patch_normal, _area = support_patch(
                vertices, fk[middle][:3, :3], normals[j]
            )
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
            joint_reference=dict(
                zip(source["joint_names"], source["finger_target_rad"], strict=True)
            )
            if reconfigure_fingers
            else None,
            support_height_stage_m=support_height_stage_m,
            middle_patch_normal=patch_normal,
            pinch_targets=pinch_targets if orient_pinch_patches else None,
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
            or (support_from_below and -escapes[links[2]][2] < 0.5)
            or (side_support and abs(escapes[links[2]][2]) > 0.35)
            or (
                support_height_stage_m is not None
                and min(v[:, 2].min() for pieces in hand.values() for v in pieces)
                < support_height_stage_m + 0.001
            )
        ):
            rejected["geometry"] += 1
            score = max(0, max(depths.values()) - 0.0015) + max(0, -depths[middle])
            if closest_rejection is None or score < closest_rejection["score_m"]:
                closest_rejection = dict(
                    score_m=score,
                    piece_depths_m=depths,
                    middle_fingers_rad=fit.x.tolist(),
                    middle_inward_up_component=float(-escapes[middle][2]),
                    minimum_hand_z_stage_m=float(
                        min(v[:, 2].min() for pieces in hand.values() for v in pieces)
                    ),
                    fitted_root_pose_stage_xyzw=moved_root.tolist(),
                    fitted_joints_rad={
                        name: float(value) for name, value in current.items()
                    },
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
        # Changing the fitted pose invalidates commands derived from its old
        # contact geometry. Recompute effort and opening after pose transfer.
        for key in (
            "finger_feedforward_Nm",
            "finger_pregrasp_rad",
            "required_contact_digits",
            "finger_stiffness_Nm_per_rad",
        ):
            planned.pop(key, None)
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
            pad_escape_direction_dot=float(escapes[links[0]] @ escapes[links[1]]),
            object_contact_points_stage_m=np.asarray(contact).tolist(),
            three_pad_geometric_contacts=True,
            middle_contact_point_stage_m=points[j].tolist(),
            middle_inward_direction_stage=(-escapes[links[2]]).tolist(),
            nominal_load_plan=load_plan,
            objective=float(np.linalg.norm(fit_all.fun)),
        )
        translation = np.linalg.norm(moved_root[:3] - source["hand_position_stage_m"])
        angle = (
            Rotation.from_quat(moved_root[3:])
            * Rotation.from_quat(source["hand_quaternion_stage_xyzw"]).inv()
        ).magnitude()
        if (
            planned["maximum_finger_delta_rad"] <= 1.2
            and translation <= proposal.get("wrist_translation_cap_m", 0.2)
            and angle <= proposal.get("wrist_rotation_cap_rad", np.pi)
        ):
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
        wrist_roll_deg=float(wrist_roll_deg),
        support_from_below=support_from_below,
        support_height_stage_m=support_height_stage_m,
        reconfigure_fingers=reconfigure_fingers,
        orient_support_patch=orient_support_patch,
        side_support=side_support,
        orient_pinch_patches=orient_pinch_patches,
        physical_task_accepted=False,
    )
    return proposal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["run", "urdfs", "proposal", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--wrist-roll-deg", type=float, default=0)
    parser.add_argument("--support-from-below", action="store_true")
    parser.add_argument("--support-height-stage-m", type=float)
    parser.add_argument("--maximum-candidates", type=int, default=48)
    parser.add_argument("--reconfigure-fingers", action="store_true")
    parser.add_argument("--orient-support-patch", action="store_true")
    parser.add_argument("--side-support", action="store_true")
    parser.add_argument("--orient-pinch-patches", action="store_true")
    args = parser.parse_args()
    result = add_middle(
        args.run,
        args.urdfs,
        args.proposal,
        args.sample,
        wrist_roll_deg=args.wrist_roll_deg,
        support_from_below=args.support_from_below,
        support_height_stage_m=args.support_height_stage_m,
        maximum_candidates=args.maximum_candidates,
        reconfigure_fingers=args.reconfigure_fingers,
        orient_support_patch=args.orient_support_patch,
        side_support=args.side_support,
        orient_pinch_patches=args.orient_pinch_patches,
    )
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
