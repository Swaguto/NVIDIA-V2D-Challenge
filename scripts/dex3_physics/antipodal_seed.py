"""Search bounded robot-only grasp seeds on exposed cooked convex surfaces.

This generates geometric proposals for a subsequent gravity/load test. It does
not alter observations and cannot certify friction, acquisition, or retention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from cooked_grasp_audit import convex_escape, world_convexes
from grasp_preload import URDFKinematics


def exposed_faces(shapes, spacing_m=None):
    """Sample exposed triangle interiors, including partially exposed large faces.

    A face centroid alone can miss the usable part of a long handle. A fixed
    barycentric grid keeps proposals deterministic and checks exposure at each
    sampled point against the union of cooked pieces.
    """
    hulls = [ConvexHull(v) for v in shapes]
    points, normals = [], []
    for shape, hull in zip(shapes, hulls, strict=True):
        if spacing_m is None:
            centers = shape[hull.simplices].mean(1)
            directions = hull.equations[:, :3]
        else:
            if not np.isfinite(spacing_m) or spacing_m <= 0:
                raise ValueError("Surface spacing must be finite and positive")
            centers, directions = [], []
            for triangle, normal in zip(
                shape[hull.simplices], hull.equations[:, :3], strict=True
            ):
                count = max(
                    1,
                    int(
                        np.ceil(
                            np.max(
                                np.linalg.norm(
                                    triangle - np.roll(triangle, 1, axis=0), axis=1
                                )
                            )
                            / spacing_m
                        )
                    ),
                )
                for a in range(count):
                    for b in range(count - a):
                        weights = (
                            np.array([a + 1 / 3, b + 1 / 3, count - a - b - 2 / 3])
                            / count
                        )
                        centers.append(np.einsum("i,ij->j", weights, triangle))
                        directions.append(normal)
            centers, directions = np.asarray(centers), np.asarray(directions)
        outward = centers + 0.0001 * directions
        exposed = np.ones(len(centers), dtype=bool)
        for other in hulls:
            if other is hull:
                continue
            inside = (
                np.einsum("ij,kj->ik", outward, other.equations[:, :3])
                + other.equations[:, 3]
                < -1e-7
            ).all(1)
            exposed &= ~inside
        points.extend(centers[exposed])
        normals.extend(directions[exposed])
    return np.asarray(points), np.asarray(normals)


def antipodal_pairs(points, normals, seed_points, radius=0.06):
    """Antipodal candidates inside a conservative friction cone (mu=0.5)."""
    pools = [
        np.flatnonzero(np.linalg.norm(points - seed, axis=1) < radius)
        for seed in seed_points
    ]
    candidates = []
    for i in pools[0]:
        j = pools[1]
        delta = points[j] - points[i]
        distance = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(distance[:, None], 1e-9)
        valid = (
            (distance >= 0.008)
            & (distance <= 0.07)
            & (np.einsum("ij,j->i", normals[j], normals[i]) < -0.8)
            & (np.einsum("ij,j->i", direction, normals[i]) < -0.89)
            & (np.einsum("ij,ij->i", direction, normals[j]) > 0.89)
        )
        for index in j[valid]:
            cost = sum(
                np.linalg.norm(points[k] - seed)
                for k, seed in zip((i, index), seed_points, strict=True)
            )
            candidates.append((float(cost), int(i), int(index)))
    return sorted(candidates)


def collision_residual(
    poses, hand_shapes, object_shapes, contact_links, minimum_other_clearance_m=0.0005
):
    """Cheap conservative face-axis overlap penalty for IK refinement.

    Omitting edge cross axes can overestimate overlap. Final acceptance always
    uses the complete independent SAT check, including edge/edge axes.
    """
    penalties = []
    for name, shapes in hand_shapes.items():
        pose = poses[name]
        maximum = -np.inf
        for local, local_axes in shapes:
            world = local @ pose[:3, :3].T + pose[:3, 3]
            axes_hand = local_axes @ pose[:3, :3].T
            for vertices, axes_object in object_shapes:
                aabb_gap = np.maximum(
                    world.min(0) - vertices.max(0), vertices.min(0) - world.max(0)
                )
                if aabb_gap.max() > max(0.003, minimum_other_clearance_m):
                    continue
                axes = np.r_[axes_hand, axes_object]
                origin = vertices.mean(0)
                a = np.einsum("ij,kj->ik", world - origin, axes)
                b = np.einsum("ij,kj->ik", vertices - origin, axes)
                gap = np.maximum(a.min(0) - b.max(0), b.min(0) - a.max(0)).max()
                maximum = max(maximum, -gap)
        allowed = 0.001 if name in contact_links else -minimum_other_clearance_m
        penalties.append(max(0.0, maximum - allowed) / 0.001)
    return np.asarray(penalties)


def select_grasp(candidates, mode="source", gravity_lever_weight=0.0):
    """Rank valid proposals; gravity lever is a heuristic, not wrench closure."""
    if mode == "load":
        opposed = [c for c in candidates if c["pad_escape_direction_dot"] <= -0.8]
        return (
            min(opposed, key=lambda c: (c["gravity_lever_m"], c["objective"]))
            if opposed
            else None
        )
    if mode != "source":
        raise ValueError("Unknown grasp ranking mode")
    return (
        min(
            candidates,
            key=lambda c: (
                c["objective"] + gravity_lever_weight * c["gravity_lever_m"] / 0.05
            ),
        )
        if candidates
        else None
    )


def search(
    run,
    urdfs,
    *,
    side="right",
    max_candidates=32,
    radius_m=0.06,
    surface_spacing_m=None,
    max_joint_delta_rad=0.6,
    max_rotation_rad=0.8,
    translation_cap_m=0.05,
    gravity_lever_weight=0.0,
    sample_name=None,
    selection_mode="source",
):
    if not 0 < max_joint_delta_rad <= 1.2:
        raise ValueError("Robot posture correction must be in (0, 1.2] rad")
    if not 0 < radius_m <= 0.15 or not 1 <= max_candidates <= 512:
        raise ValueError("Contact search exceeds the declared planning budget")
    if not 0 < max_rotation_rad <= 1.2:
        raise ValueError("Robot wrist rotation correction must be in (0, 1.2] rad")
    if not 0 < translation_cap_m <= 0.15 or not 0 <= gravity_lever_weight <= 5:
        raise ValueError(
            "Robot translation or gravity lever cost exceeds planning bounds"
        )
    report = json.loads((run / "report.json").read_text())
    states_bytes = (run / "states.json").read_bytes()
    if hashlib.sha256(states_bytes).hexdigest() != report["states_sha256"]:
        raise ValueError("Source static states changed")
    states = json.loads(states_bytes)
    colliders = json.loads((run / "cooked-colliders.json").read_text())["colliders"]
    kin = URDFKinematics(urdfs / f"dex3_{side}.urdf")
    results = []
    for si, initial in enumerate(report["initial_pose_readback"]):
        if not initial["sample"].startswith(side + "-"):
            continue
        if sample_name is not None and initial["sample"] != sample_name:
            continue
        row = next(
            r for r in states["records"] if r["step"] == 0 and r["sample_index"] == si
        )
        original = dict(
            zip(
                row["finger_names_sim_order"],
                row["finger_position_rad_sim_order"],
                strict=True,
            )
        )
        root = np.array(initial["hand_body_link_pose_stage_xyzw"][kin.root_name])
        source_q = np.array([original[n] for n in kin.names])
        fk = kin.poses(root, original)
        position_error = max(
            np.max(np.abs(fk[n][:3, 3] - pose[:3]))
            for n, pose in initial["hand_body_link_pose_stage_xyzw"].items()
        )
        orientation_error = max(
            (
                Rotation.from_matrix(fk[n][:3, :3]) * Rotation.from_quat(pose[3:]).inv()
            ).magnitude()
            for n, pose in initial["hand_body_link_pose_stage_xyzw"].items()
        )
        if position_error > 2e-6 or orientation_error > 2e-6:
            raise ValueError("URDF FK and recorded SDK link frames differ")
        object_shapes = [
            v
            for c in colliders
            if c["asset"] == f"object-{si}"
            for v in world_convexes(c, row["object_root_state_stage_xyzw"][:7])
        ]
        hand_shapes = {}
        for c in colliders:
            if c["asset"] != f"hand-{si}":
                continue
            # Convert recorded SDK cooked geometry into each URDF link frame.
            world = world_convexes(
                c, initial["hand_body_link_pose_stage_xyzw"][c["body_name"]]
            )
            hand_shapes[c["body_name"]] = [
                (v - fk[c["body_name"]][:3, 3]) @ fk[c["body_name"]][:3, :3]
                for v in world
            ]
        links = [f"{side}_hand_thumb_2_link", f"{side}_hand_index_1_link"]
        hand_collision = {
            n: [(v, ConvexHull(v).equations[:, :3]) for v in pieces]
            for n, pieces in hand_shapes.items()
        }
        object_collision = [(v, ConvexHull(v).equations[:, :3]) for v in object_shapes]
        centers = [np.concatenate(hand_shapes[n]).mean(0) for n in links]
        world_centers = [
            fk[n][:3, :3] @ p + fk[n][:3, 3]
            for n, p in zip(links, centers, strict=True)
        ]
        points, normals = exposed_faces(object_shapes, surface_spacing_m)
        pairs = antipodal_pairs(points, normals, world_centers, radius_m)
        if gravity_lever_weight:
            com = np.asarray(row["object_root_state_stage_xyzw"][:3])
            pairs.sort(
                key=lambda item: (
                    item[0]
                    + gravity_lever_weight
                    * np.linalg.norm((points[list(item[1:])].mean(0) - com)[:2])
                )
            )
        # Avoid spending the whole fit budget on adjacent samples of one face.
        diverse = []
        for item in pairs:
            pair = points[list(item[1:])]
            if all(
                np.linalg.norm(pair - points[list(old[1:])], axis=1).max() >= 0.008
                for old in diverse
            ):
                diverse.append(item)
            if len(diverse) == max_candidates:
                break
        rejected = {"contact_fit": 0, "rotation_bound": 0, "collision_or_opposition": 0}
        closest_fit = None
        candidates = []
        lower = np.r_[
            [-translation_cap_m] * 3,
            [-min(0.7, max_rotation_rad)] * 3,
            [
                max(kin.limits[n][0] + 0.08 - q, -max_joint_delta_rad)
                for n, q in zip(kin.names, source_q, strict=True)
            ],
        ]
        upper = np.r_[
            [translation_cap_m] * 3,
            [min(0.7, max_rotation_rad)] * 3,
            [
                min(kin.limits[n][1] - 0.08 - q, max_joint_delta_rad)
                for n, q in zip(kin.names, source_q, strict=True)
            ],
        ]

        def decode(x):
            pose = np.r_[
                root[:3] + x[:3],
                (Rotation.from_rotvec(x[3:6]) * Rotation.from_quat(root[3:])).as_quat(),
            ]
            return pose, source_q + x[6:]

        for _, a, b in diverse:
            ids = [a, b]
            anchors = []
            for link, index in zip(links, ids, strict=True):
                v = np.concatenate(hand_shapes[link])
                direction = fk[link][:3, :3].T @ -normals[index]
                score = v @ direction
                anchors.append(v[score > score.max() - 0.001].mean(0))
            target = points[ids] - 0.00075 * normals[ids]

            def residual(x):
                pose, q = decode(x)
                poses = kin.poses(pose, dict(zip(kin.names, q, strict=True)))
                actual = np.array(
                    [
                        poses[n][:3, :3] @ p + poses[n][:3, 3]
                        for n, p in zip(links, anchors, strict=True)
                    ]
                )
                # Bound the whole pad against each chosen contact tangent plane,
                # rather than fitting a vertex while another pad edge penetrates.
                plane_errors = []
                for link, index in zip(links, ids, strict=True):
                    vertices = np.concatenate(hand_shapes[link])
                    world = vertices @ poses[link][:3, :3].T + poses[link][:3, 3]
                    clearance = np.min((world - points[index]) @ normals[index])
                    plane_errors.append(2.0 * (clearance + 0.00075) / 0.001)
                return np.r_[
                    (actual - target).ravel() / 0.002,
                    plane_errors,
                    0.1 * x[:3] / translation_cap_m,
                    0.1 * x[3:6] / 0.7,
                    0.04 * x[6:] / max_joint_delta_rad,
                ]

            fit = least_squares(
                residual,
                np.clip(np.zeros(13), lower + 1e-9, upper - 1e-9),
                bounds=(lower, upper),
                max_nfev=100,
            )
            if not fit.success or np.max(np.abs(residual(fit.x)[:6])) > 0.5:
                rejected["contact_fit"] += 1
                error = float(np.max(np.abs(residual(fit.x)[:6])) * 0.002)
                closest_fit = (
                    min(closest_fit, error) if closest_fit is not None else error
                )
                continue
            pose, q = decode(fit.x)
            if np.linalg.norm(fit.x[3:6]) > max_rotation_rad:
                rejected["rotation_bound"] += 1
                continue
            poses = kin.poses(pose, dict(zip(kin.names, q, strict=True)))
            if (
                collision_residual(poses, hand_collision, object_collision, links).max()
                > 0.2
            ):
                # Keep the same contact/pose objective while allowing the other
                # fingers to move out of the object. A pad-only objective leaves
                # their joints at the source pose even when the wrist rotates.
                def refine_residual(x):
                    moved, fingers = decode(x)
                    body = kin.poses(moved, dict(zip(kin.names, fingers, strict=True)))
                    return np.r_[
                        residual(x),
                        3.0
                        * collision_residual(
                            body, hand_collision, object_collision, links
                        ),
                    ]

                refined = least_squares(
                    refine_residual, fit.x, bounds=(lower, upper), max_nfev=100
                )
                if (
                    np.max(np.abs(residual(refined.x)[:6])) <= 0.5
                    and np.linalg.norm(refined.x[3:6]) <= max_rotation_rad
                ):
                    fit = refined
                    pose, q = decode(fit.x)
                    poses = kin.poses(pose, dict(zip(kin.names, q, strict=True)))
            depths, escapes = {}, {}
            for name, shapes in hand_shapes.items():
                overlaps = [
                    convex_escape(v @ poses[name][:3, :3].T + poses[name][:3, 3], obj)
                    for v in shapes
                    for obj in object_shapes
                ]
                depth, escape = max(overlaps, key=lambda item: item[0])
                depths[name], escapes[name] = depth, escape
            dot = float(escapes[links[0]] @ escapes[links[1]])
            if (
                dot > -0.5
                or min(depths[n] for n in links) < 0
                or max(depths.values()) > 0.0015
            ):
                rejected["collision_or_opposition"] += 1
                continue
            candidates.append(
                dict(
                    hand_position_stage_m=pose[:3].tolist(),
                    hand_quaternion_stage_xyzw=pose[3:].tolist(),
                    joint_names=kin.names,
                    finger_target_rad=q.tolist(),
                    pad_escape_direction_dot=dot,
                    maximum_piece_penetration_lower_bound_m=max(depths.values()),
                    root_translation_delta_m=fit.x[:3].tolist(),
                    root_rotation_delta_rad=fit.x[3:6].tolist(),
                    maximum_finger_delta_rad=float(np.max(np.abs(fit.x[6:]))),
                    objective=float(np.linalg.norm(fit.fun)),
                    object_contact_points_stage_m=points[ids].tolist(),
                    gravity_lever_m=float(
                        np.linalg.norm(
                            (
                                points[ids].mean(0)
                                - row["object_root_state_stage_xyzw"][:3]
                            )[:2]
                        )
                    ),
                )
            )
        results.append(
            dict(
                sample=initial["sample"],
                source_sample=report["provenance"][si]["sample"],
                sdk_fk_position_error_m=float(position_error),
                sdk_fk_orientation_error_rad=float(orientation_error),
                candidate_pairs=len(pairs),
                searched_pairs=len(diverse),
                rejection_counts=rejected,
                closest_rejected_anchor_max_component_error_m=closest_fit,
                geometrically_valid_candidates=len(candidates),
                candidate=select_grasp(
                    candidates, selection_mode, gravity_lever_weight
                ),
                candidates=candidates,
            )
        )
    return dict(
        schema="v2d.dex3-antipodal-seed.v1",
        source_states_sha256=report["states_sha256"],
        planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_report_sha256=hashlib.sha256(
            (run / "report.json").read_bytes()
        ).hexdigest(),
        cooked_sha256=hashlib.sha256(
            (run / "cooked-colliders.json").read_bytes()
        ).hexdigest(),
        urdf_sha256=hashlib.sha256(
            (urdfs / f"dex3_{side}.urdf").read_bytes()
        ).hexdigest(),
        joint_limit_margin_rad=0.08,
        finger_delta_cap_rad=max_joint_delta_rad,
        surface_spacing_m=surface_spacing_m,
        contact_search_radius_m=radius_m,
        wrist_translation_cap_m=translation_cap_m,
        gravity_lever_cost_weight=gravity_lever_weight,
        selection_mode=selection_mode,
        wrist_rotation_cap_rad=max_rotation_rad,
        target_pad_preload_m=0.00075,
        maximum_allowed_piece_penetration_lower_bound_m=0.0015,
        results=results,
        source_observations_changed=False,
        physical_task_accepted=False,
        scope="Geometric robot-motion proposals only; require a new gravity/load test and collision review",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--urdfs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--surface-spacing-mm", type=float)
    parser.add_argument("--contact-radius-m", type=float, default=0.06)
    parser.add_argument("--finger-delta-cap-rad", type=float, default=0.6)
    parser.add_argument("--wrist-rotation-cap-rad", type=float, default=0.8)
    parser.add_argument("--wrist-translation-cap-m", type=float, default=0.05)
    parser.add_argument("--gravity-lever-weight", type=float, default=0.0)
    parser.add_argument("--sample")
    parser.add_argument("--selection", choices=["source", "load"], default="source")
    args = parser.parse_args()
    result = search(
        args.run,
        args.urdfs,
        max_candidates=args.max_candidates,
        side=args.side,
        radius_m=args.contact_radius_m,
        surface_spacing_m=args.surface_spacing_mm / 1000
        if args.surface_spacing_mm is not None
        else None,
        max_joint_delta_rad=args.finger_delta_cap_rad,
        max_rotation_rad=args.wrist_rotation_cap_rad,
        translation_cap_m=args.wrist_translation_cap_m,
        gravity_lever_weight=args.gravity_lever_weight,
        sample_name=args.sample,
        selection_mode=args.selection,
    )
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
