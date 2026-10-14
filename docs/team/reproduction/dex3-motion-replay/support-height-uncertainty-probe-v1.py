"""Probe frozen closest support witnesses with the existing CAD query machinery.

This is a finite sensitivity diagnostic, not a new reproducer or ground truth.
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

p = argparse.ArgumentParser(description=__doc__)
for name in ("case", "methods", "sheet", "proof", "out"):
    p.add_argument("--" + name, type=Path, required=True)
a = p.parse_args()
assert not a.out.exists()
started = time.monotonic()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert (
    sha(a.proof) == "a5390d2dfa7a9c3f7a3729aa80dec4100b77238ed65447a46b4bc1da715fde17"
)
files = json.loads((a.case / "file-manifest.json").read_text())["files_sha256"]
for name, expected in files.items():
    path = a.case / name
    assert path.resolve().is_relative_to(a.case.resolve()) and sha(path) == expected
sys.path.insert(0, str(a.methods))
from audit_static_hold_scene_vertices_v1 import controls, model
from CAD_contact_replay_motion_v1 import LockedReplayMotion, load
from original_CAD_scene_query_v1 import SceneQuery

tested = controls()
source = a.case / "source/robotic_grounding"
sys.path.insert(0, str(source / "source/robotic_grounding"))
from robotic_grounding.retarget.retarget_utils import setup_dex3_kinematics

motion = LockedReplayMotion(
    a.case / "bundle",
    a.case / "retarget",
    None,
    None,
    None,
    a.case / "proposal",
    source=source,
)
scene = SceneQuery(motion, setup_dex3_kinematics, model)
sheet = load(a.sheet / "inferred-support-sheet.npz")
assert (
    sha(a.sheet / "inferred-support-sheet.npz")
    == "347b1ce1446329e16d9fe3faa510689c2bf5d742f3956e4fa988c27f43c91af7"
)
normal = sheet["plane_world"][:3]
np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-12, rtol=0)
sheet_shape = model(sheet["vertices_world_m"], sheet["faces"])
proof = json.loads(a.proof.read_text())
assert proof["complete_original_CAD_triangle_surface_interpolation_certified"]
assert not proof["failed_or_unresolved_intervals"]
states = load(a.case / "proposal/robot-motion-proposal.npz")
groups = {}
not_robust_intervals = set()
sheet_records = 0
minimum = float("inf")
for interval in proof["certified_intervals"]:
    timestamp = (interval["time_low_s"] + interval["time_high_s"]) / 2
    row, fraction = motion.bracket(timestamp)
    for record in interval["original_surface_checks"]:
        if record["object"] != 2:
            continue
        sheet_records += 1
        minimum = min(minimum, record["margin_m"])
        if record["margin_m"] < 0.010:
            not_robust_intervals.add(
                (interval["segment"], interval["time_low_s"], interval["time_high_s"])
            )
        si = record["side"]
        held = next(
            (
                phase
                for phase in motion.sides[si]["phases"]
                if phase["interval"][0] <= timestamp <= phase["interval"][1]
            ),
            None,
        )
        observed = bool(states["source_hand_valid"][row : row + 2, si].all())
        group = (si, held is not None, observed)
        witness = {
            "side": si,
            "closed_phase": held is not None,
            "source_observations_at_both_brackets": observed,
            "timestamp_s": timestamp,
            "source_bracket_frames": states["frame_indices"][row : row + 2].tolist(),
            "witness_link": record["link"],
            "original_sampled_unsigned_distance_m": record["distance_m"],
            "original_interval_lower_bound_m": record["margin_m"],
        }
        if (
            group not in groups
            or record["distance_m"]
            < groups[group]["original_sampled_unsigned_distance_m"]
        ):
            groups[group] = witness
print(
    json.dumps(
        {
            "stage": "selected_closest_witnesses",
            "groups": len(groups),
            "sheet_records": sheet_records,
        }
    ),
    flush=True,
)
collision_request = coal.CollisionRequest()
collision_request.enable_contact = True
collision_request.num_max_contacts = 8
distance_request = coal.DistanceRequest()
distance_request.enable_signed_distance = False
distance_request.gjk_tolerance = 1e-12
results = []
for group, witness in sorted(groups.items()):
    poses = motion.evaluate(witness["timestamp_s"])[0]
    kin, geom, gd, meshes = scene.hands[witness["side"]]
    q = poses[witness["side"]]
    kin.configuration.update(q)
    pin.updateGeometryPlacements(kin.robot.model, kin.configuration.data, geom, gd, q)
    hypotheses = []
    for offset in (0.0, -0.010, -0.008, 0.008, 0.010):
        sheet_transform = coal.Transform3s(np.eye(3), normal * offset)
        links = []
        for (vertices, faces, shape, name), transform in zip(meshes, gd.oMg):
            placement = coal.Transform3s(transform.rotation, transform.translation)
            collision = coal.CollisionResult()
            coal.collide(
                shape,
                placement,
                sheet_shape,
                sheet_transform,
                collision_request,
                collision,
            )
            distance = coal.DistanceResult()
            coal.distance(
                shape,
                placement,
                sheet_shape,
                sheet_transform,
                distance_request,
                distance,
            )
            links.append(
                {
                    "link": name,
                    "surface_intersection": bool(collision.isCollision()),
                    "unsigned_distance_m": float(distance.min_distance),
                    "contact_positions_world_m": [
                        np.asarray(c.pos).tolist() for c in collision.getContacts()
                    ],
                }
            )
        if offset == 0:
            assert not any(link["surface_intersection"] for link in links)
            matching = next(
                link for link in links if link["link"] == witness["witness_link"]
            )
            np.testing.assert_allclose(
                matching["unsigned_distance_m"],
                witness["original_sampled_unsigned_distance_m"],
                atol=1e-9,
                rtol=0,
            )
        hypotheses.append(
            {
                "hypothesized_normal_shift_m": offset,
                "crossing_links": [
                    link for link in links if link["surface_intersection"]
                ],
                "minimum_unsigned_distance_m": min(
                    link["unsigned_distance_m"] for link in links
                ),
            }
        )
    results.append({**witness, "hypotheses": hypotheses})
    print(
        json.dumps(
            {
                "stage": "support_witness",
                "side": witness["side"],
                "closed_phase": witness["closed_phase"],
                "observed": witness["source_observations_at_both_brackets"],
                "source_bracket_frames": witness["source_bracket_frames"],
                "crossing_offset_hypotheses": [
                    h["hypothesized_normal_shift_m"]
                    for h in hypotheses
                    if h["crossing_links"]
                ],
            }
        ),
        flush=True,
    )
report = {
    "schema": "v2d.frozen-support-height-sensitivity-diagnostic.v1",
    "source_sha256": sha(Path(__file__)),
    "case_manifest_sha256": sha(a.case / "file-manifest.json"),
    "motion_payload_sha256": sha(a.case / "proposal/robot-motion-proposal.npz"),
    "original_continuous_proof_sha256": sha(a.proof),
    "sheet_payload_sha256": sha(a.sheet / "inferred-support-sheet.npz"),
    "existing_query_module_sha256": sha(a.methods / "original_CAD_scene_query_v1.py"),
    "existing_motion_module_sha256": sha(a.methods / "CAD_contact_replay_motion_v1.py"),
    "controls": tested,
    "witness_selection": "Minimum actual sampled sheet distance within each left/right, closed/free and source-observation bracket class of the complete stored CAD proof; no hand-picked frame.",
    "translation_ball_radius_m": 0.010,
    "translation_ball_check_scope": "A sufficient conservative bound subtracts10mm from stored unsigned sheet separation. Failing this bound is missing robustness proof, not an actual collision by itself.",
    "stored_sheet_records": sheet_records,
    "minimum_stored_sheet_bound_m": minimum,
    "stored_intervals_without_10mm_translation_ball_clearance_bound": len(
        not_robust_intervals
    ),
    "normal_offsets_are_hypotheses_not_measurements": True,
    "original_zero_offset_witnesses_exact_and_nonintersecting": True,
    "witness_results": results,
    "hypothesized_shift_crossing_witnesses": sum(
        any(h["crossing_links"] for h in r["hypotheses"]) for r in results
    ),
    "all_source_motion_masks_anchors_objects_cameras_and_original_sheet_unchanged": True,
    "continuous_shifted_surface_check_performed": False,
    "real_table_thickness_or_physical_collision_established": False,
    "physical_task_accepted": False,
    "source_accuracy_accepted": False,
    "elapsed_seconds": time.monotonic() - started,
}
a.out.mkdir(parents=True)
(a.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        {
            "hypothesized_shift_crossing_witnesses": report[
                "hypothesized_shift_crossing_witnesses"
            ],
            "stored_intervals_without_10mm_bound": len(not_robust_intervals),
            "elapsed_seconds": report["elapsed_seconds"],
        }
    ),
    flush=True,
)
