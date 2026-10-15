"""Measure declared Dex3 pad gaps against original rendered CAD at source times.

This unsigned surface diagnostic does not infer a solid from nonwatertight CAD,
rerun a fit, change source observations, or approve a physical motion target.
Source-only annotations enter validation only after geometric measurements.
Use --cad-candidates to bind the selected pad faces to the final CAD grasp fit;
without it the preserved legacy mode measures the earlier template faces.
Use the pinned Linux geometry environment documented for the corrected replay.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import coal
import numpy as np
import trimesh


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def surface_distance(first, second, model):
    request = coal.DistanceRequest()
    request.enable_signed_distance = False
    request.gjk_tolerance = 1e-12
    result = coal.DistanceResult()
    identity = coal.Transform3s(np.eye(3), np.zeros(3))
    coal.distance(first, identity, second, identity, request, result)
    points = np.asarray([result.getNearestPoint1(), result.getNearestPoint2()])
    distance = float(result.min_distance)
    assert np.isfinite(points).all() and np.isfinite(distance) and distance >= -1e-12
    # Coal's unsigned zero-distance result need not supply a common contact
    # witness. Never interpret its fallback nearest points as contact evidence.
    if distance > 1e-12:
        assert abs(np.linalg.norm(points[0] - points[1]) - distance) <= 1e-8
    return max(0.0, distance), points, int(result.b1), int(result.b2)


def controls(model):
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    triangle = np.asarray([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]])
    distance, _, _, _ = surface_distance(
        model(triangle, faces), model(triangle + [0, 0, 0.003], faces), model
    )
    assert abs(distance - 0.003) < 1e-12
    crossing = np.asarray([[0.1, 0.1, -0.001], [0.1, 0.1, 0.001], [0.2, 0.1, 0]])
    distance, _, _, _ = surface_distance(
        model(triangle, faces), model(crossing, faces), model
    )
    assert distance <= 1e-12
    return {
        "3mm_original_triangle_gap_recovered": True,
        "crossing_original_triangles_have_zero_unsigned_gap": True,
        "unsigned_gap_alone_is_not_a_solid_penetration_test": True,
        "zero_distance_fallback_points_not_used_as_contact_witnesses": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "methods", "fitted-inputs", "annotation", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case-manifest-sha256", required=True)
    parser.add_argument(
        "--cad-candidates",
        type=Path,
        help="Final CAD candidate report bound by the case; audits its selected faces",
    )
    args = parser.parse_args()
    assert not args.out.exists()
    started = time.monotonic()
    assert sha(args.case / "file-manifest.json") == args.case_manifest_sha256
    case_files = read(args.case / "file-manifest.json")["files_sha256"]
    root = args.case.resolve()
    for name, expected in case_files.items():
        path = args.case / name
        assert path.resolve().is_relative_to(root) and sha(path) == expected
    sys.path.insert(0, str(args.methods))
    from audit_static_hold_scene_vertices_v1 import model
    from closest_mesh_surface_v2 import analytic_controls, closest_triangle_points

    tested = controls(model)
    tested.update(analytic_controls())
    tiny = np.asarray([[[0.0, 0, 0], [0.0001, 0, 0], [0, 0.0001, 0]]])
    np.testing.assert_allclose(
        closest_triangle_points(tiny, tiny[:, 2]), tiny[:, 2], atol=1e-15, rtol=0
    )
    tested["submillimetre_triangle_vertex_projection_verified"] = True
    source = args.case / "source/robotic_grounding"
    sys.path.insert(0, str(source / "source/robotic_grounding"))
    from robotic_grounding.retarget.retarget_utils import setup_dex3_kinematics
    from robotic_grounding.retarget.track3_model_lock import verify_robot_assets

    report = read(args.case / "proposal/report.json")
    verify_robot_assets(
        report, source / "source/robotic_grounding/robotic_grounding/assets"
    )
    bundle = read(args.case / "bundle/manifest.json")
    data = load(args.case / "bundle/episode.npz")
    motion = load(args.case / "proposal/robot-motion-proposal.npz")
    cad_trials = {}
    if args.cad_candidates is not None:
        if sha(args.cad_candidates) != report.get("CAD_candidate_report_sha256"):
            raise ValueError("CAD candidate report does not belong to this motion")
        cad_report = read(args.cad_candidates)
        if (
            not cad_report.get("terminal")
            or cad_report.get("atlas_sha256") != report["atlas_sha256"]
        ):
            raise ValueError("Incomplete or mismatched final CAD candidate report")
        for trial in cad_report["trials"]:
            prefix = trial["prefix"]
            if prefix in cad_trials or not trial.get(
                "candidate_static_geometry_passed"
            ):
                raise ValueError("Ambiguous or refused final CAD grasp")
            faces = trial["rounds"][-1]["selected_original_hand_faces"]
            targets = {target["finger"]: target for target in trial["targets"]}
            if set(faces) != {"thumb", "index"} or set(targets) != set(faces):
                raise ValueError(
                    "Final CAD grasp needs both declared pad faces and targets"
                )
            cad_trials[prefix] = trial
    assert (
        sha(args.case / "proposal/robot-motion-proposal.npz")
        == report["payload_sha256"]
    )
    np.testing.assert_array_equal(data["frame_indices"], motion["frame_indices"])
    np.testing.assert_array_equal(data["hand_valid"], motion["source_hand_valid"])
    assert motion["robot_pose_is_inferred"].all()
    assert not motion["robot_motion_target_validated"].any()
    atlas_path = args.fitted_inputs / "per-grasp-motion-atlas-reserve-v2.json"
    assert sha(atlas_path) == report["atlas_sha256"]
    atlas = read(atlas_path)
    bindings, phases = [], {}
    for spec in atlas["grasps"]:
        template_path = args.fitted_inputs / spec["templates"] / "report.json"
        scene_path = args.fitted_inputs / spec["scenes"] / "report.json"
        assert template_path.resolve().is_relative_to(args.fitted_inputs.resolve())
        assert scene_path.resolve().is_relative_to(args.fitted_inputs.resolve())
        assert sha(template_path) == spec["templates_report_sha256"]
        assert sha(scene_path) == spec["scenes_report_sha256"]
        side, object_token, phase_token = spec["prefix"].split("-")
        oi, pi = (
            int(object_token.removeprefix("object")),
            int(phase_token.removeprefix("grasp")),
        )
        scenes = [
            s
            for s in read(scene_path)["scenes"]
            if s["side"] == side and s["object"] == oi
        ]
        assert len(scenes) == 1
        candidates = [
            c
            for c in read(template_path)["templates"]
            if c["side"] == side
            and c["object"] == oi
            and c.get("seed") == scenes[0]["seed"]
        ]
        assert len(candidates) == 1 and candidates[0]["success"]
        candidate = candidates[0]
        assert {p["finger"] for p in candidate["pads"]} == {"thumb", "index"}
        if args.cad_candidates is not None:
            trial = cad_trials[spec["prefix"]]
            if (
                trial["side"] != side
                or trial["object"] != oi
                or trial["parent_seed"] != scenes[0]["seed"]
                or trial["parent_template_report_sha256"] != sha(template_path)
            ):
                raise ValueError(
                    "Final CAD grasp is not bound to this template and phase"
                )
            candidate = {
                **candidate,
                "pads": [
                    {
                        **pad,
                        "original_mesh_triangle_index": trial["rounds"][-1][
                            "selected_original_hand_faces"
                        ][pad["finger"]],
                        "current_CAD_target": next(
                            t for t in trial["targets"] if t["finger"] == pad["finger"]
                        ),
                    }
                    for pad in candidate["pads"]
                ],
            }
        key = (("left", "right").index(side), pi)
        assert key not in phases
        phases[key] = (oi, spec["prefix"], candidate)
        bindings.append(
            {
                "prefix": spec["prefix"],
                "selected_seed": scenes[0]["seed"],
                "templates_report_sha256": sha(template_path),
                "scenes_report_sha256": sha(scene_path),
                "original_pad_triangles": {
                    p["finger"]: p["original_mesh_triangle_index"]
                    for p in candidate["pads"]
                },
                "pad_selection_basis": "final_CAD_fit"
                if args.cad_candidates is not None
                else "original_template",
            }
        )
    if args.cad_candidates is not None and set(cad_trials) != {
        b["prefix"] for b in bindings
    }:
        raise ValueError("Final CAD report and motion grasp roster differ")
    hands = []
    for side in ("left", "right"):
        kin = setup_dex3_kinematics(side=side, max_iter=1)
        pads = {}
        for finger, suffix in (("thumb", "2"), ("index", "1")):
            name = f"{side}_hand_{finger}_{suffix}_link"
            objects = [
                g
                for g in kin.robot.collision_model.geometryObjects
                if kin.robot.model.frames[g.parentFrame].name == name
            ]
            assert len(objects) == 1
            obj = objects[0]
            mesh = trimesh.load(Path(obj.meshPath), force="mesh", process=False)
            vertices = np.asarray(mesh.vertices, dtype=float) * np.asarray(
                obj.meshScale
            )
            # Collision placement may contain a local URDF origin.
            frame = kin.robot.model.frames[obj.parentFrame]
            relative = frame.placement.inverse() * obj.placement
            vertices = vertices @ relative.rotation.T + relative.translation
            pads[finger] = (
                name,
                vertices,
                np.asarray(mesh.faces),
                sha(Path(obj.meshPath)),
            )
        hands.append((kin, pads))
    objects = {}
    for entry in bundle["objects"]:
        oi = entry["index"]
        raw = data[f"object_{oi}_vertices"]
        vertices, inverse = np.unique(raw, axis=0, return_inverse=True)
        np.testing.assert_array_equal(vertices[inverse], raw)
        objects[oi] = (vertices, inverse[data[f"object_{oi}_faces"]])
    args.out.mkdir(parents=True)
    records = []
    single_face = np.asarray([[0, 1, 2]], dtype=np.int64)
    for row, frame in enumerate(data["frame_indices"]):
        world_objects = {}
        for si, (kin, pads) in enumerate(hands):
            pi = int(motion["robot_grasp_phase_index_inferred"][row, si])
            if pi < 0:
                continue
            oi, prefix, candidate = phases[(si, pi)]
            if oi not in world_objects:
                vertices, faces = objects[oi]
                pose = data[f"object_{oi}_world"][row]
                matrix = pose[:3, :3] * float(data[f"object_{oi}_scale"][row])
                world = vertices @ matrix.T + pose[:3, 3]
                world_objects[oi] = (world, faces, model(world, faces))
            world, object_faces, object_model = world_objects[oi]
            kin.configuration.update(motion["robot_qpos_inferred"][row, si])
            measurements = []
            for intended in candidate["pads"]:
                finger = intended["finger"]
                name, vertices, faces, mesh_sha = pads[finger]
                fixed_id = int(intended["original_mesh_triangle_index"])
                if not 0 <= fixed_id < len(faces):
                    raise ValueError(
                        "Selected pad triangle lies outside its locked mesh"
                    )
                tf = kin.configuration.get_transform_frame_to_world(name)
                triangle = vertices[faces[fixed_id]] @ tf.rotation.T + tf.translation
                distance, points, face1, face2 = surface_distance(
                    model(triangle, single_face), object_model, model
                )
                assert face1 == 0 and 0 <= face2 < len(object_faces)
                cad_triangle = world[object_faces[face2]]
                projected = closest_triangle_points(
                    np.asarray([triangle, cad_triangle]), points
                )
                residuals = np.linalg.norm(projected - points, axis=1)
                witness_verified = distance > 1e-12 and bool((residuals <= 1e-8).all())
                if distance > 1e-12:
                    assert witness_verified, {
                        "frame": int(frame),
                        "side": si,
                        "finger": finger,
                        "distance": distance,
                        "residuals": residuals.tolist(),
                        "points": points.tolist(),
                        "triangle": triangle.tolist(),
                        "cad_triangle": cad_triangle.tolist(),
                    }
                normals = np.cross(
                    np.asarray([triangle, cad_triangle])[:, 1]
                    - np.asarray([triangle, cad_triangle])[:, 0],
                    np.asarray([triangle, cad_triangle])[:, 2]
                    - np.asarray([triangle, cad_triangle])[:, 0],
                )
                norms = np.linalg.norm(normals, axis=1)
                assert (norms > 1e-15).all()
                normal_dot = float(np.dot(*(normals / norms[:, None])))
                measurements.append(
                    {
                        "finger": finger,
                        "link": name,
                        "mesh_sha256": mesh_sha,
                        "original_hand_triangle": fixed_id,
                        "original_CAD_triangle": face2,
                        "unsigned_triangle_surface_gap_m": distance,
                        "witness_points_world_m": points.tolist(),
                        "independent_witness_residual_m": residuals.tolist(),
                        "positive_gap_witness_independently_verified": witness_verified,
                        "triangle_normal_dot_diagnostic": normal_dot,
                        "within_unchanged_2mm_contact_distance": distance
                        <= 0.002 + 1e-12,
                    }
                )
                if args.cad_candidates is not None:
                    target_id = int(intended["current_CAD_target"]["CAD_triangle"])
                    if not 0 <= target_id < len(object_faces):
                        raise ValueError(
                            "Selected target triangle lies outside its locked CAD"
                        )
                    target_triangle = world[object_faces[target_id]]
                    target_distance, target_points, _, _ = surface_distance(
                        model(triangle, single_face),
                        model(target_triangle, single_face),
                        model,
                    )
                    target_projection = closest_triangle_points(
                        np.asarray([triangle, target_triangle]), target_points
                    )
                    target_residual = np.linalg.norm(
                        target_projection - target_points, axis=1
                    )
                    target_verified = target_distance > 1e-12 and bool(
                        (target_residual <= 1e-8).all()
                    )
                    if target_distance > 1e-12 and not target_verified:
                        raise ValueError(
                            "Selected CAD contact witness failed independent projection"
                        )
                    target_normal = np.cross(
                        target_triangle[1] - target_triangle[0],
                        target_triangle[2] - target_triangle[0],
                    )
                    target_normal /= np.linalg.norm(target_normal)
                    hand_normal = normals[0] / norms[0]
                    target_dot = float(hand_normal @ target_normal)
                    measurements[-1]["current_CAD_target_witness"] = {
                        "original_CAD_triangle": target_id,
                        "unsigned_triangle_gap_m": target_distance,
                        "points_world_m": target_points.tolist(),
                        "independent_projection_residual_m": target_residual.tolist(),
                        "positive_gap_witness_verified": target_verified,
                        "target_normal_world": target_normal.tolist(),
                        "hand_target_normal_dot": target_dot,
                        "within2mm_and_opposed": target_verified
                        and target_distance <= 0.002 + 1e-12
                        and target_dot <= -0.5 + 1e-7,
                    }
            records.append(
                {
                    "source_frame": int(frame),
                    "timestamp_s": float(data["timestamps_s"][row]),
                    "side": ("left", "right")[si],
                    "object": oi,
                    "prefix": prefix,
                    "inferred_closed_grasp_intent": bool(
                        motion["robot_contact_intent_inferred"][row, si, oi]
                    ),
                    "source_hand_valid": bool(data["hand_valid"][row, si]),
                    "pads": measurements,
                    "both_declared_pad_faces_within2mm": all(
                        p["within_unchanged_2mm_contact_distance"] for p in measurements
                    ),
                }
            )
            if args.cad_candidates is not None:
                witnesses = [m["current_CAD_target_witness"] for m in measurements]
                pair_dot = float(np.dot(*(w["target_normal_world"] for w in witnesses)))
                records[-1]["current_CAD_target_normal_pair_dot"] = pair_dot
                records[-1]["both_current_CAD_target_faces_within2mm_and_opposed"] = (
                    all(w["within2mm_and_opposed"] for w in witnesses)
                    and pair_dot <= -0.5 + 1e-7
                )
        if (row + 1) % 40 == 0 or row + 1 == len(data["frame_indices"]):
            print(
                json.dumps(
                    {
                        "source_frames": row + 1,
                        "hand_frame_records": len(records),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    # Labels are deliberately read after all measurements; no fitting occurs.
    labels = read(args.annotation)
    validations = []
    by_key = {(r["source_frame"], r["side"], r["object"]): r for r in records}
    for label in labels["visible_grasp_intervals"]:
        oi = next(
            o["index"] for o in bundle["objects"] if o["name"] == label["object_id"]
        )
        frames = range(
            label["definite_core_start_frame"], label["definite_core_end_frame"] + 1
        )
        missing, failed = [], []
        for f in frames:
            record = by_key.get((f, label["hand"], oi))
            if record is None:
                missing.append(f)
            elif not record["both_declared_pad_faces_within2mm"]:
                failed.append(f)
        validations.append(
            {
                "grasp_id": label["grasp_id"],
                "core_frames": len(frames),
                "frames_without_grasp_phase": missing,
                "original_CAD_2mm_gap_failed_core_frames": failed,
            }
        )
        if args.cad_candidates is not None:
            validations[-1]["current_CAD_target_witness_failed_core_frames"] = [
                f
                for f in frames
                if (f, label["hand"], oi) not in by_key
                or not by_key[(f, label["hand"], oi)][
                    "both_current_CAD_target_faces_within2mm_and_opposed"
                ]
            ]
    summary = []
    for prefix in sorted({r["prefix"] for r in records}):
        selected = [
            r
            for r in records
            if r["prefix"] == prefix and r["inferred_closed_grasp_intent"]
        ]
        summary.append(
            {
                "prefix": prefix,
                "inferred_closed_hand_frames": len(selected),
                "both_pad_faces_within2mm_frames": sum(
                    r["both_declared_pad_faces_within2mm"] for r in selected
                ),
                "pads": {
                    finger: {
                        "minimum_gap_m": float(np.min(gaps)),
                        "median_gap_m": float(np.median(gaps)),
                        "maximum_gap_m": float(np.max(gaps)),
                    }
                    for finger in ("thumb", "index")
                    for gaps in [
                        [
                            p["unsigned_triangle_surface_gap_m"]
                            for r in selected
                            for p in r["pads"]
                            if p["finger"] == finger
                        ]
                    ]
                },
            }
        )
        if args.cad_candidates is not None:
            summary[-1][
                "both_current_CAD_target_faces_within2mm_and_opposed_frames"
            ] = sum(
                r["both_current_CAD_target_faces_within2mm_and_opposed"]
                for r in selected
            )
    output = {
        "schema": "v2d.original-CAD-selected-pad-gaps.v1",
        "source_sha256": sha(Path(__file__)),
        "case_manifest_sha256": args.case_manifest_sha256,
        "motion_payload_sha256": report["payload_sha256"],
        "hand_geometry_scope": "locked URDF collision meshes, not visual meshes",
        "pad_selection_basis": "final_CAD_fit"
        if args.cad_candidates is not None
        else "original_template",
        "CAD_candidate_report_sha256": sha(args.cad_candidates)
        if args.cad_candidates is not None
        else None,
        "original_BVH_builder_sha256": sha(
            args.methods / "audit_static_hold_scene_vertices_v1.py"
        ),
        "original_triangle_projection_module_sha256": sha(
            args.methods / "closest_mesh_surface_v2.py"
        ),
        "atlas_sha256": sha(atlas_path),
        "annotation_sha256": sha(args.annotation),
        "controls": tested,
        "phase_bindings": bindings,
        "summary": summary,
        "source_label_validation": validations,
        "measurements": records,
        "all_source_frames_processed": len(data["frame_indices"]),
        "original_selected_pad_triangle_distance_checks": 2 * len(records),
        "full_original_CAD_triangles_preserved": True,
        "full_source_object_affine_and_scale_preserved": True,
        "contact_distance_limit_m": 0.002,
        "continuous_contact_certified": False,
        "unsigned_gap_does_not_certify_solid_clearance_or_force": True,
        "source_labels_read_after_geometry": True,
        "original_poses_masks_and_targets_modified": False,
        "source_accuracy_accepted": False,
        "physical_task_accepted": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write(args.out / "report.json", output)
    print(
        json.dumps({"summary": summary, "source_label_validation": validations}),
        flush=True,
    )


if __name__ == "__main__":
    main()
