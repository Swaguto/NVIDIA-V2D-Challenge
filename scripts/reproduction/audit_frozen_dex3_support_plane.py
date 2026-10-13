"""Measure original Dex3 surfaces against a frozen estimated source plane.

All source poses and explicit interpolation knots are checked, without changing
the object, human or robot poses. This is a finite diagnostic for one uncertain
plane, not a continuous environment or real-table clearance certificate.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import trimesh
from audit_frozen_dex3_original_contacts import load, read, sha, write
from scipy.spatial import ConvexHull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "plane", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case-manifest-sha256", required=True)
    parser.add_argument("--plane-report-sha256", required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    started = time.monotonic()
    assert sha(args.case / "file-manifest.json") == args.case_manifest_sha256
    case_root = args.case.resolve()
    for name, expected in read(args.case / "file-manifest.json")[
        "files_sha256"
    ].items():
        path = args.case / name
        assert path.resolve().is_relative_to(case_root) and sha(path) == expected
    plane_report = read(args.plane / "report.json")
    assert sha(args.plane / "report.json") == args.plane_report_sha256
    assert (
        sha(args.plane / "support-plane-hypothesis.npz")
        == plane_report["payload_sha256"]
    )
    assert (
        sha(args.case / "bundle/manifest.json")
        == plane_report["bundle_manifest_sha256"]
    )
    hypothesis = load(args.plane / "support-plane-hypothesis.npz")
    plane = hypothesis["plane_world"]
    np.testing.assert_allclose(np.linalg.norm(plane[:3]), 1.0, atol=1e-12, rtol=0)
    seed = np.eye(3)[int(np.argmin(abs(plane[:3])))]
    u = np.cross(plane[:3], seed)
    u /= np.linalg.norm(u)
    basis = np.stack([u, np.cross(plane[:3], u)], axis=1)
    origin = -plane[3] * plane[:3]
    near = abs(hypothesis["background_signed_plane_distance_m"]) < 0.005
    xy = (hypothesis["background_points_world_m"][near] - origin) @ basis
    cells = np.floor(xy / 0.01).astype(np.int64)
    distinct = np.unique(
        np.c_[cells, hypothesis["background_source_rows"][near]], axis=0
    )
    supported, count = np.unique(distinct[:, :2], axis=0, return_counts=True)
    reliable = {tuple(cell) for cell in supported[count >= 3]}
    selected = np.asarray([tuple(cell) in reliable for cell in cells])
    assert selected.sum() >= 100
    hull = ConvexHull(xy[selected])
    data = load(args.case / "bundle/episode.npz")
    motion = load(args.case / "proposal/robot-motion-proposal.npz")
    knots = load(args.case / "proposal/replay-knots.npz")
    report = read(args.case / "proposal/report.json")
    assert (
        sha(args.case / "proposal/robot-motion-proposal.npz")
        == report["payload_sha256"]
    )
    assert sha(args.case / "proposal/replay-knots.npz") == report["replay_knots_sha256"]
    assert not motion["robot_motion_target_validated"].any()
    source = args.case / "source/robotic_grounding"
    sys.path.insert(0, str(source / "source/robotic_grounding"))
    from robotic_grounding.retarget.retarget_utils import setup_dex3_kinematics
    from robotic_grounding.retarget.track3_model_lock import verify_robot_assets

    verify_robot_assets(
        report, source / "source/robotic_grounding/robotic_grounding/assets"
    )
    hands, geometry = [], []
    for side in ("left", "right"):
        kin = setup_dex3_kinematics(side=side, max_iter=1)
        geom = kin.robot.collision_model
        meshes = []
        for obj in geom.geometryObjects:
            mesh = trimesh.load(Path(obj.meshPath), force="mesh", process=False)
            raw = np.asarray(mesh.vertices, dtype=float) * np.asarray(obj.meshScale)
            vertices, first_ids, inverse = np.unique(
                raw, axis=0, return_index=True, return_inverse=True
            )
            np.testing.assert_array_equal(vertices[inverse], raw)
            name = kin.robot.model.frames[obj.parentFrame].name
            meshes.append((vertices, first_ids, name))
            geometry.append(
                {
                    "side": side,
                    "link": name,
                    "mesh_sha256": sha(Path(obj.meshPath)),
                    "original_vertices": len(raw),
                    "exact_unique_vertices": len(vertices),
                    "original_triangles": len(mesh.faces),
                }
            )
        hands.append((kin, geom, geom.createData(), meshes))
    args.out.mkdir(parents=True)
    results, summaries = {}, []
    for kind, times, poses in (
        ("source", motion["timestamps_s"], motion["robot_qpos_inferred"]),
        ("knots", knots["timestamps_s"], knots["qpos_inferred"]),
    ):
        minima = np.empty((len(times), 2))
        below_footprint_counts = np.zeros((len(times), 2), dtype=np.int64)
        worst = [None, None]
        worst_inside = [None, None]
        for row, (timestamp, q) in enumerate(zip(times, poses)):
            for si, (kin, geom, gd, meshes) in enumerate(hands):
                kin.configuration.update(q[si])
                pin.updateGeometryPlacements(
                    kin.robot.model, kin.configuration.data, geom, gd, q[si]
                )
                minima[row, si] = np.inf
                for (vertices, ids, name), tf in zip(meshes, gd.oMg):
                    # Exact linear plane functional; avoid rounding a world mesh.
                    distance = (
                        vertices @ (tf.rotation.T @ plane[:3])
                        + tf.translation @ plane[:3]
                        + plane[3]
                    )
                    vi = int(distance.argmin())
                    value = float(distance[vi])
                    minima[row, si] = min(minima[row, si], value)
                    # Only vertices already below the plane can be a negative
                    # witness. This does not certify triangles crossing the
                    # polygon with all vertices outside its inferred footprint.
                    below = np.flatnonzero(distance < -0.002)
                    if len(below):
                        world_below = vertices[below] @ tf.rotation.T + tf.translation
                        projected = (world_below - origin) @ basis
                        inside = (
                            projected @ hull.equations[:, :2].T + hull.equations[:, 2]
                            <= 1e-10
                        ).all(axis=1)
                        below_footprint_counts[row, si] += int(inside.sum())
                        inside_ids = below[inside]
                        if len(inside_ids):
                            ii = int(inside_ids[distance[inside_ids].argmin()])
                            if (
                                worst_inside[si] is None
                                or distance[ii]
                                < worst_inside[si]["signed_plane_distance_m"]
                            ):
                                worst_inside[si] = {
                                    "side": ("left", "right")[si],
                                    "timestamp_s": float(timestamp),
                                    "source_frame": int(data["frame_indices"][row])
                                    if kind == "source"
                                    else None,
                                    "link": name,
                                    "original_vertex_index": int(ids[ii]),
                                    "point_world_m": (
                                        tf.rotation @ vertices[ii] + tf.translation
                                    ).tolist(),
                                    "signed_plane_distance_m": float(distance[ii]),
                                }
                    if (
                        worst[si] is None
                        or value < worst[si]["signed_plane_distance_m"]
                    ):
                        world = tf.rotation @ vertices[vi] + tf.translation
                        assert abs(world @ plane[:3] + plane[3] - value) < 1e-12
                        worst[si] = {
                            "side": ("left", "right")[si],
                            "timestamp_s": float(timestamp),
                            "source_frame": int(data["frame_indices"][row])
                            if kind == "source"
                            else None,
                            "link": name,
                            "original_vertex_index": int(ids[vi]),
                            "point_world_m": world.tolist(),
                            "signed_plane_distance_m": value,
                        }
            if (row + 1) % 500 == 0:
                print(
                    {
                        "kind": kind,
                        "states": row + 1,
                        "elapsed_s": time.monotonic() - started,
                    },
                    flush=True,
                )
        results[kind + "_timestamps_s"] = times
        results[kind + "_minimum_robot_signed_plane_distance_m"] = minima
        results[kind + "_vertices_below2mm_inside_inferred_footprint"] = (
            below_footprint_counts
        )
        summaries.append(
            {
                "kind": kind,
                "states": len(times),
                "per_hand": [
                    {
                        "side": side,
                        "minimum_m": float(minima[:, si].min()),
                        "states_below2mm": int((minima[:, si] < -0.002).sum()),
                        "states_below20mm": int((minima[:, si] < -0.020).sum()),
                        "worst_original_vertex": worst[si],
                        "states_with_below2mm_vertex_inside_inferred_footprint": int(
                            (below_footprint_counts[:, si] > 0).sum()
                        ),
                        "worst_below2mm_original_vertex_inside_inferred_footprint": worst_inside[
                            si
                        ],
                    }
                    for si, side in enumerate(("left", "right"))
                ],
            }
        )
    np.savez_compressed(args.out / "support-distances.npz", **results)
    output = {
        "schema": "v2d.frozen-Dex3-estimated-support-diagnostic.v1",
        "source_sha256": sha(Path(__file__)),
        "shared_utility_sha256": sha(
            Path(__file__).with_name("audit_frozen_dex3_original_contacts.py")
        ),
        "case_manifest_sha256": args.case_manifest_sha256,
        "motion_payload_sha256": report["payload_sha256"],
        "replay_knots_sha256": report["replay_knots_sha256"],
        "source_plane_report_sha256": args.plane_report_sha256,
        "plane_world": plane.tolist(),
        "inferred_footprint": {
            "basis_columns_world": basis.tolist(),
            "origin_world_m": origin.tolist(),
            "polygon_coordinates_m": xy[selected][hull.vertices].tolist(),
            "cell_pitch_m": 0.01,
            "minimum_distinct_sample_frames_per_cell": 3,
            "plane_neighborhood_m": 0.005,
            "convex_hull_can_fill_unobserved_gaps": True,
            "finite_footprint_vertex_diagnostics_are_not_complete_surface_clearance": True,
        },
        "original_geometry": geometry,
        "summaries": summaries,
        "payload_sha256": sha(args.out / "support-distances.npz"),
        "triangle_linear_plane_minimum_attained_at_original_vertex": True,
        "all_original_vertices_preserved_by_exact_deduplication": True,
        "continuous_plane_clearance_certified": False,
        "plane_semantic_identity_unconfirmed": True,
        "plane_is_estimated_stereo_not_ground_truth": True,
        "infinite_plane_is_not_complete_environment": True,
        "plane_uncertainty_not_bounded": True,
        "source_or_generated_poses_changed": False,
        "physical_task_accepted": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write(args.out / "report.json", output)
    print({"summaries": summaries, "physical_task_accepted": False}, flush=True)


if __name__ == "__main__":
    main()
