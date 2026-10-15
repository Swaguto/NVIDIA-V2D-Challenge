"""Verify stored full-path certificate coverage, bounds and frozen bindings.

This checks the recorded certificate. It does not rerun geometric certification
or establish source accuracy, true-object penetration or manipulation dynamics.
"""

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path

from reproduce_frozen_dex3_replay import digest


def verify(case):
    import numpy as np

    snapshot = (
        Path(__file__).resolve().parents[2]
        / "docs/team/reproduction/dex3-motion-replay"
    )
    expected = json.loads((snapshot / "corrected-inputs-manifest-v1.json").read_text())
    if digest(case / "file-manifest.json") != expected["case_file_manifest_sha256"]:
        raise ValueError("Use the verified corrected simulator case")
    manifest = json.loads((case / "file-manifest.json").read_text())
    for name, value in manifest["files_sha256"].items():
        if (
            not (case / name).resolve().is_relative_to(case)
            or digest(case / name) != value
        ):
            raise ValueError("Input path or hash mismatch: " + name)
    methods = json.loads(
        (snapshot / "continuous-methods-manifest-v1.json").read_text()
    )["files_sha256"]
    r = json.loads((case / "continuous/report.json").read_text())
    proposal = json.loads((case / "proposal/report.json").read_text())
    for field, value in (
        ("source_sha256", methods["certify_locked_replay_scene_v11.py"]),
        ("motion_module_sha256", methods["locked_replay_motion_v3.py"]),
        ("proposal_report_sha256", digest(case / "proposal/report.json")),
        ("bundle_manifest_sha256", digest(case / "bundle/manifest.json")),
    ):
        if r[field] != value:
            raise ValueError("Certificate binding differs: " + field)
    for field, filename in (
        ("payload_sha256", "robot-motion-proposal.npz"),
        ("replay_knots_sha256", "replay-knots.npz"),
    ):
        if proposal[field] != digest(case / "proposal" / filename):
            raise ValueError("Proposal binding differs: " + field)
    if (
        not r["complete_replay_object_self_and_cross_hand_surfaces_certified"]
        or r["failed_or_unresolved_intervals"]
        or r["object_material_world_depth_limit_m"] != 0.002
        or r["physical_task_accepted"]
        or r["physical_material_geometry_ground_truth"]
    ):
        raise ValueError(
            "Certificate completeness, limits or physical qualification differs"
        )
    with np.load(case / "proposal/replay-knots.npz", allow_pickle=False) as z:
        times = z["timestamps_s"]
    if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("Invalid interpolation knots")
    groups = {}
    bounded, separated, other_complete, maximum_upper = 0, 0, 0, 0.0
    for interval in r["certified_intervals"]:
        groups.setdefault(interval["segment"], []).append(interval)
        records = interval["material_checks"]
        if (
            len(records) != 32
            or len({(c["side"], c["object"], c["link"]) for c in records}) != 32
        ):
            raise ValueError("Incomplete original-surface/material pair coverage")
        for field in ("self_separation_bound_m", "cross_hand_separation_bound_m"):
            if not math.isfinite(interval[field]) or interval[field] <= 0:
                raise ValueError("Missing positive swept hand separation bound")
        for c in records:
            if (
                c["pass"] is not True
                or not math.isfinite(c["world_padding_m"])
                or c["world_padding_m"] < 0
            ):
                raise ValueError("Refused or invalid material certificate")
            if "maximum_depth_upper_m" in c:
                upper = c["maximum_depth_upper_m"]
                terms = (
                    c["reference_depth_upper_m"],
                    c["canonical_reference_displacement_m"],
                    c["world_padding_m"],
                )
                if not all(math.isfinite(v) and v >= 0 for v in (upper, *terms)):
                    raise ValueError("Nonfinite or negative material bound")
                if upper > 0.002 or upper + 1e-15 < sum(terms):
                    raise ValueError(
                        "Stored complete-surface bound exceeds the unchanged limit"
                    )
                maximum_upper = max(maximum_upper, upper)
                bounded += 1
            elif c["status"] == "complete swept AABB separated":
                separated += 1
            elif c["status"] in (
                "certified for declared material only",
                "cached complete rigid hold scale/affine/local-anchor family",
            ):
                other_complete += 1
            else:
                raise ValueError("Unknown certificate form")
    if set(groups) != set(range(len(times) - 1)):
        raise ValueError("Missing trajectory segments")
    for segment, rows in groups.items():
        rows.sort(key=lambda row: row["time_low_s"])
        if (
            rows[0]["time_low_s"] != times[segment]
            or rows[-1]["time_high_s"] != times[segment + 1]
        ):
            raise ValueError("Segment endpoint mismatch")
        if any(row["time_high_s"] <= row["time_low_s"] for row in rows):
            raise ValueError("Empty or reversed certified interval")
        if any(a["time_high_s"] != b["time_low_s"] for a, b in pairwise(rows)):
            raise ValueError("Gap or overlap in stored continuous coverage")
    return {
        "schema": "v2d.corrected-continuous-certificate-verification.v1",
        "verifier_sha256": digest(Path(__file__)),
        "continuous_report_sha256": digest(case / "continuous/report.json"),
        "all_segment_intervals_exactly_covered": True,
        "segments": len(times) - 1,
        "certified_subintervals": len(r["certified_intervals"]),
        "complete_surface_material_records": bounded + separated + other_complete,
        "complete_surface_margin_records": bounded,
        "complete_swept_AABB_separation_records": separated,
        "other_declared_complete_surface_records": other_complete,
        "other_complete_forms_verified_by_frozen_report_hash_not_new_geometry": True,
        "maximum_stored_depth_upper_m": maximum_upper,
        "material_limit_m": 0.002,
        "new_geometric_certification_run": False,
        "true_object_penetration_guaranteed": False,
        "source_accuracy_accepted": False,
        "physical_task_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.case.resolve()), indent=2))


if __name__ == "__main__":
    main()
