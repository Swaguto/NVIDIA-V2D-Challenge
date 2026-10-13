"""Verify stored original-CAD and material/self/cross proof coverage; no new geometry."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from reproduce_frozen_dex3_replay import digest


def coverage(report, times, field):
    groups = {}
    for interval in report["certified_intervals"]:
        groups.setdefault(interval["segment"], []).append(interval)
    if set(groups) != set(range(len(times) - 1)):
        raise ValueError("Missing or unexpected trajectory segments")
    minimum = float("inf")
    for segment, rows in groups.items():
        rows.sort(key=lambda row: row["time_low_s"])
        if (
            rows[0]["time_low_s"] != times[segment]
            or rows[-1]["time_high_s"] != times[segment + 1]
        ):
            raise ValueError("Segment boundaries are not exactly covered")
        if any(a["time_high_s"] != b["time_low_s"] for a, b in pairwise(rows)):
            raise ValueError("Gap or overlap in interval proof coverage")
        for row in rows:
            checks = row[field]
            if (
                row["time_low_s"] >= row["time_high_s"]
                or len(checks) != 32
                or len({(v["side"], v["object"], v["link"]) for v in checks}) != 32
            ):
                raise ValueError(
                    "Incomplete interval or original link/object accounting"
                )
            if field == "original_surface_checks":
                margins = [v["margin_m"] for v in checks]
                if not all(value > 0 for value in margins):
                    raise ValueError("Original CAD surface margin refused")
            else:
                margins = [
                    row["self_separation_bound_m"],
                    row["cross_hand_separation_bound_m"],
                ]
                if not all(value > 0 for value in margins) or not all(
                    v["pass"] for v in checks
                ):
                    raise ValueError(
                        "Model material/self/cross-hand certificate refused"
                    )
                if any(
                    v.get("maximum_depth_upper_m", 0) > 0.002 + 1e-12 for v in checks
                ):
                    raise ValueError(
                        "Recorded material bound exceeds unchanged 2 mm limit"
                    )
            minimum = min(minimum, *margins)
    return minimum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    case = args.case.resolve()
    snapshot = (
        Path(__file__).resolve().parents[2]
        / "docs/team/reproduction/dex3-motion-replay"
    )
    kit = json.loads((snapshot / "CAD-fitted-method-kit-manifest-v1.json").read_text())
    if digest(case / "file-manifest.json") != kit["published_case_manifest_sha256"]:
        raise ValueError("Use the verified frozen CAD case")
    manifest = json.loads((case / "file-manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = case / name
        if not path.resolve().is_relative_to(case) or digest(path) != expected:
            raise ValueError("Case byte/path mismatch: " + name)
    proposal = case / "proposal"
    pr = json.loads((proposal / "report.json").read_text())
    if pr["payload_sha256"] != digest(proposal / "robot-motion-proposal.npz") or pr[
        "replay_knots_sha256"
    ] != digest(proposal / "replay-knots.npz"):
        raise ValueError("Motion payload binding mismatch")
    with np.load(proposal / "replay-knots.npz", allow_pickle=False) as z:
        times, q = z["timestamps_s"], z["qpos_inferred"]
    model, cad = [
        json.loads((case / f"validation/{name}/report.json").read_text())
        for name in ("model-continuous", "original-CAD-continuous")
    ]
    if (
        not model["complete_replay_object_self_and_cross_hand_surfaces_certified"]
        or not cad["complete_original_CAD_triangle_surface_interpolation_certified"]
    ):
        raise ValueError("A complete certificate is unavailable")
    if model["object_material_world_depth_limit_m"] != 0.002:
        raise ValueError("Material acceptance limit changed")
    for report, producer in (
        (model, "certify_CAD_contact_replay_scene_v3.py"),
        (cad, "certify_original_CAD_replay_v1.py"),
    ):
        if (
            report["failed_or_unresolved_intervals"]
            or report["source_sha256"] != kit["files_sha256"]["methods/" + producer]
            or report["proposal_report_sha256"] != digest(proposal / "report.json")
            or report["bundle_manifest_sha256"] != digest(case / "bundle/manifest.json")
            or report["motion_module_sha256"]
            != kit["files_sha256"]["methods/CAD_contact_replay_motion_v1.py"]
        ):
            raise ValueError("Certificate input/method binding mismatch")
    cad_min = coverage(cad, times, "original_surface_checks")
    model_min = coverage(model, times, "material_checks")
    with (
        np.load(proposal / "robot-motion-proposal.npz", allow_pickle=False) as data,
        np.load(case / "bundle/episode.npz", allow_pickle=False) as source,
    ):
        np.testing.assert_array_equal(data["source_hand_valid"], source["hand_valid"])
        np.testing.assert_array_equal(
            data["source_hand_directly_observed"], source["hand_directly_observed"]
        )
        rows = np.searchsorted(times, data["timestamps_s"])
        np.testing.assert_array_equal(times[rows], data["timestamps_s"])
        np.testing.assert_array_equal(q[rows], data["robot_qpos_inferred"])
        if (
            not data["robot_pose_is_inferred"].all()
            or data["robot_motion_target_validated"].any()
        ):
            raise ValueError("Inferred poses were promoted to accepted targets")
        observed = source["hand_valid"].sum(axis=0).tolist()
        held = int(data["robot_contact_intent_inferred"].any(axis=2).sum())
    result = {
        "schema": "v2d.original-CAD-stored-certificate-verification.v1",
        "verifier_sha256": digest(Path(__file__)),
        "segments": len(times) - 1,
        "CAD_intervals": len(cad["certified_intervals"]),
        "model_intervals": len(model["certified_intervals"]),
        "CAD_original_link_object_records": 32 * len(cad["certified_intervals"]),
        "model_original_link_object_records": 32 * len(model["certified_intervals"]),
        "minimum_original_CAD_unsigned_margin_m": cad_min,
        "minimum_self_or_cross_margin_m": model_min,
        "observed_hand_counts": observed,
        "inferred_closed_hand_object_states": held,
        "all_segment_boundaries_and_saved_margins_verified": True,
        "geometry_recomputed": False,
        "source_accuracy_accepted": False,
        "physical_task_accepted": False,
    }
    if args.out:
        with args.out.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
