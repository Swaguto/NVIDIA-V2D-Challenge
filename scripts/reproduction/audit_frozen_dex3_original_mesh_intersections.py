"""Check original hand/CAD triangle intersections at every source frame.

Original nonwatertight CAD is never assigned a trustworthy signed solid depth.
Surface contacts are reported separately from inferred own-object grasp intent.
This finite source-frame check is not continuous certification or source truth.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import coal
import numpy as np
import pinocchio as pin
import trimesh


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "methods", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case-manifest-sha256", required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    started = time.monotonic()
    assert sha(args.case / "file-manifest.json") == args.case_manifest_sha256
    files = json.loads((args.case / "file-manifest.json").read_text())["files_sha256"]
    for name, expected in files.items():
        path = args.case / name
        assert (
            path.resolve().is_relative_to(args.case.resolve()) and sha(path) == expected
        )
    sys.path.insert(0, str(args.methods))
    from audit_static_hold_scene_vertices_v1 import controls, model

    tested = controls()
    # This negative control demonstrates why a surface test is not a solid test.
    inner, outer = (
        trimesh.creation.box(extents=[0.01] * 3),
        trimesh.creation.box(extents=[0.1] * 3),
    )
    identity = coal.Transform3s(np.eye(3), np.zeros(3))
    contained = coal.CollisionResult()
    coal.collide(
        model(inner.vertices, inner.faces),
        identity,
        model(outer.vertices, outer.faces),
        identity,
        coal.CollisionRequest(),
        contained,
    )
    assert not contained.isCollision()
    tested[
        "full_containment_has_no_surface_intersection_and_must_not_be_accepted_as_solid_clearance"
    ] = True
    source = args.case / "source/robotic_grounding"
    sys.path.insert(0, str(source / "source/robotic_grounding"))
    from robotic_grounding.retarget.retarget_utils import setup_dex3_kinematics
    from robotic_grounding.retarget.track3_model_lock import verify_robot_assets

    proposal = json.loads((args.case / "proposal/report.json").read_text())
    verify_robot_assets(
        proposal, source / "source/robotic_grounding/robotic_grounding/assets"
    )
    with np.load(args.case / "bundle/episode.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    with np.load(
        args.case / "proposal/robot-motion-proposal.npz", allow_pickle=False
    ) as z:
        motion = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(d["frame_indices"], motion["frame_indices"])
    np.testing.assert_array_equal(d["hand_valid"], motion["source_hand_valid"])
    assert (
        motion["robot_pose_is_inferred"].all()
        and not motion["robot_motion_target_validated"].any()
    )
    objects, topology = [], []
    for oi in range(
        len(json.loads((args.case / "bundle/manifest.json").read_text())["objects"])
    ):
        raw, faces = d[f"object_{oi}_vertices"], d[f"object_{oi}_faces"]
        vertices, inverse = np.unique(raw, axis=0, return_inverse=True)
        np.testing.assert_array_equal(vertices[inverse], raw)
        faces = inverse[faces]
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        topology.append(
            {
                "object": oi,
                "original_vertices": len(raw),
                "exact_unique_vertices": len(vertices),
                "original_triangles": len(faces),
                "exact_original_triangle_points_preserved": True,
                "watertight_after_exact_welding": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
            }
        )
        objects.append((vertices, faces))
    hands, geometry = [], []
    for side in ("left", "right"):
        kin = setup_dex3_kinematics(side=side, max_iter=1)
        geom = kin.robot.collision_model
        meshes = []
        for obj in geom.geometryObjects:
            mesh = trimesh.load(Path(obj.meshPath), force="mesh", process=False)
            original = np.asarray(mesh.vertices, dtype=float) * np.asarray(
                obj.meshScale
            )
            vertices, inverse = np.unique(original, axis=0, return_inverse=True)
            np.testing.assert_array_equal(vertices[inverse], original)
            faces = inverse[mesh.faces]
            name = kin.robot.model.frames[obj.parentFrame].name
            meshes.append((vertices, faces, model(vertices, faces), name))
            geometry.append(
                {
                    "side": side,
                    "link": name,
                    "mesh_sha256": sha(Path(obj.meshPath)),
                    "all_original_triangle_points_preserved": True,
                    "original_triangles": len(faces),
                }
            )
        hands.append((kin, geom, geom.createData(), meshes))
    args.out.mkdir(parents=True)
    request = coal.CollisionRequest()
    request.enable_contact = True
    request.num_max_contacts = 16
    hits = []
    checks, aabb_separated, exact_bvh_checks = 0, 0, 0
    for row, frame in enumerate(d["frame_indices"]):
        world_hands = []
        for si, (kin, geom, gd, meshes) in enumerate(hands):
            q = motion["robot_qpos_inferred"][row, si]
            kin.configuration.update(q)
            pin.updateGeometryPlacements(
                kin.robot.model, kin.configuration.data, geom, gd, q
            )
            world_hands.append(
                [
                    (
                        vertices @ tf.rotation.T + tf.translation,
                        faces,
                        shape,
                        name,
                        coal.Transform3s(tf.rotation, tf.translation),
                    )
                    for (vertices, faces, shape, name), tf in zip(meshes, gd.oMg)
                ]
            )
        for oi, (vertices, faces) in enumerate(objects):
            pose = d[f"object_{oi}_world"][row]
            matrix = pose[:3, :3] * float(d[f"object_{oi}_scale"][row])
            world = vertices @ matrix.T + pose[:3, 3]
            object_low, object_high = world.min(axis=0), world.max(axis=0)
            shape = None
            for si, links in enumerate(world_hands):
                for hand_world, hand_faces, hand_shape, name, transform in links:
                    checks += 1
                    if (
                        (hand_world.max(axis=0) < object_low)
                        | (object_high < hand_world.min(axis=0))
                    ).any():
                        aabb_separated += 1
                        continue
                    if shape is None:
                        # Preserve the full source affine and scale; do not rigidize it.
                        shape = model(world, faces)
                    result = coal.CollisionResult()
                    coal.collide(
                        hand_shape, transform, shape, identity, request, result
                    )
                    exact_bvh_checks += 1
                    if not result.isCollision():
                        continue
                    contacts = result.getContacts()
                    witnesses = []
                    for contact in contacts:
                        if not (
                            0 <= contact.b1 < len(hand_faces)
                            and 0 <= contact.b2 < len(faces)
                        ):
                            continue
                        point = np.asarray(contact.pos)
                        triangles = np.array(
                            [
                                hand_world[hand_faces[contact.b1]],
                                world[faces[contact.b2]],
                            ]
                        )
                        projected = trimesh.triangles.closest_point(
                            triangles, np.array([point, point])
                        )
                        residual = np.linalg.norm(projected - point, axis=1)
                        witnesses.append(
                            {
                                "hand_triangle": int(contact.b1),
                                "object_triangle": int(contact.b2),
                                "point_world_m": point.tolist(),
                                "point_to_original_surfaces_residual_m": residual.tolist(),
                                "on_both_surfaces_within1e_8m": bool(
                                    (residual <= 1e-8).all()
                                ),
                                "solver_contact_depth_m_not_a_solid_depth_bound": float(
                                    contact.penetration_depth
                                ),
                            }
                        )
                    hits.append(
                        {
                            "source_frame": int(frame),
                            "timestamp_s": float(d["timestamps_s"][row]),
                            "side": ("left", "right")[si],
                            "link": name,
                            "against_object": oi,
                            "inferred_own_object_grasp_intent": bool(
                                motion["robot_contact_intent_inferred"][row, si, oi]
                            ),
                            "source_hand_valid": bool(d["hand_valid"][row, si]),
                            "source_hand_directly_observed": bool(
                                d["hand_directly_observed"][row, si]
                            ),
                            "returned_contacts": len(contacts),
                            "contact_limit_reached": len(contacts) == 16,
                            "independently_checked_contact_positions": witnesses,
                        }
                    )
        if (row + 1) % 20 == 0 or row + 1 == len(d["frame_indices"]):
            progress = {
                "source_frames_complete": row + 1,
                "total_source_frames": len(d["frame_indices"]),
                "original_surface_pairs_checked": checks,
                "intersecting_pairs": len(hits),
                "elapsed_seconds": time.monotonic() - started,
                "terminal": False,
            }
            write(args.out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    expected_pairs = (
        len(d["frame_indices"]) * sum(len(h[3]) for h in hands) * len(objects)
    )
    assert checks == expected_pairs
    report = {
        "schema": "v2d.original-rendered-cad-source-intersections.v1",
        "source_sha256": sha(Path(__file__)),
        "reused_original_bvh_builder_sha256": sha(
            args.methods / "audit_static_hold_scene_vertices_v1.py"
        ),
        "case_manifest_sha256": args.case_manifest_sha256,
        "motion_payload_sha256": sha(args.case / "proposal/robot-motion-proposal.npz"),
        "controls": tested,
        "source_frames_complete": len(d["frame_indices"]),
        "original_surface_pairs_checked": checks,
        "AABB_separated_pairs": aabb_separated,
        "exact_original_triangle_BVH_checks": exact_bvh_checks,
        "all_source_frame_surface_pairs_checked": True,
        "all_original_hand_and_object_triangles_preserved": True,
        "source_object_full_affine_and_scale_preserved": True,
        "object_topology": topology,
        "hand_geometry": geometry,
        "intersecting_surface_pairs": hits,
        "intersecting_pairs_with_no_inferred_own_grasp_intent": sum(
            not v["inferred_own_object_grasp_intent"] for v in hits
        ),
        "original_poses_masks_and_targets_modified": False,
        "continuous_original_CAD_intersection_certified": False,
        "signed_solid_depth_trusted": False,
        "source_accuracy_accepted": False,
        "physical_task_accepted": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write(args.out / "report.json", report)
    write(
        args.out / "progress.json",
        {
            "terminal": True,
            "source_frames_complete": len(d["frame_indices"]),
            "intersecting_pairs": len(hits),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in (
                    "intersecting_surface_pairs",
                    "object_topology",
                    "hand_geometry",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Do not disable invariant checks")
    main()
