"""Measure saved contact coverage on the original axis; never grant visual acceptance."""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

EPISODES = (
    0,
    2,
    11,
    12,
    13,
    18,
    21,
    22,
    23,
    24,
    26,
    27,
    31,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
)
TIPS = {"thumb", "index", "middle", "ring", "little"}


def longest_gap(frames, supported):
    """Inclusive sample durations on the source timestamp axis, including the tail."""
    steps = [b["timestamp_s"] - a["timestamp_s"] for a, b in pairwise(frames)]
    if any(step <= 0 for step in steps):
        raise ValueError("Source timestamps must increase")
    dt = statistics.median(steps) if steps else 0.0
    best = None
    start = None
    for i in range(len(frames) + 1):
        unknown = i < len(frames) and not supported.get(frames[i]["frame_id"], False)
        if unknown and start is None:
            start = i
        if not unknown and start is not None:
            end = i - 1
            candidate = {
                "start_frame": frames[start]["frame_id"],
                "end_frame": frames[end]["frame_id"],
                "frames": end - start + 1,
                "seconds": frames[end]["timestamp_s"]
                - frames[start]["timestamp_s"]
                + dt,
            }
            if best is None or candidate["seconds"] > best["seconds"]:
                best = candidate
            start = None
    return best


def measure(rows, frames):
    grouped = defaultdict(list)
    objects = sorted({row["object_id"] for row in rows})
    frame_ids = {frame["frame_id"] for frame in frames}
    if len(frame_ids) != len(frames):
        raise ValueError("Duplicate source frame IDs")
    for row in rows:
        if row["frame_id"] not in frame_ids or row["hand"] not in ("left", "right"):
            raise ValueError("Unexpected frame or hand")
        if not row["valid"] and (
            row["raw_contact"] is not None
            or row["window_contact"] is not None
            or row["bridged"]
        ):
            raise ValueError("Unknown contact must remain null and unbridged")
        grouped[row["frame_id"], row["hand"], row["object_id"]].append(row)
    hand_object = []
    for hand in ("left", "right"):
        for obj in objects:
            support, raw, reasons = {}, {}, Counter()
            for frame in frames:
                values = grouped[frame["frame_id"], hand, obj]
                if len(values) != 5 or {row["fingertip"] for row in values} != TIPS:
                    raise ValueError(
                        "Every source frame/hand/object needs exactly five fingertip rows"
                    )
                valid_flags = {row["valid"] for row in values}
                if len(valid_flags) != 1:
                    raise ValueError(
                        "Conflicting geometry validity within a hand/object frame"
                    )
                support[frame["frame_id"]] = valid_flags == {True}
                raw[frame["frame_id"]] = any(
                    row["raw_contact"] is True for row in values
                )
                if not support[frame["frame_id"]]:
                    labels = {row["reason"] for row in values}
                    if len(labels) != 1:
                        raise ValueError("Conflicting dropout reasons")
                    reasons[next(iter(labels))] += 1
            hand_object.append(
                {
                    "hand": hand,
                    "object_id": obj,
                    "source_frames": len(frames),
                    "supported_frames": sum(support.values()),
                    "usable_fraction_all_frames": sum(support.values()) / len(frames),
                    "raw_contact_frames": sum(raw.values()),
                    "longest_unsupported_source_gap": longest_gap(frames, support),
                    "unsupported_frame_reasons": dict(reasons),
                }
            )
    return {
        "source_frames": len(frames),
        "contact_rows": len(rows),
        "valid_rows": sum(row["valid"] for row in rows),
        "unknown_rows": sum(not row["valid"] for row in rows),
        "hand_object": hand_object,
    }


def evaluate_visible_interval(rows, frames, interval):
    """Compute interval metrics only for separately source-annotated visible holds."""
    if interval.get("visibility") != "visible_hold" or not interval.get(
        "source_evidence"
    ):
        raise ValueError(
            "Visible holding intervals need independent source-video evidence"
        )
    ids = [f["frame_id"] for f in frames]
    start, end = ids.index(interval["start_frame"]), ids.index(interval["end_frame"])
    if start > end:
        raise ValueError("Holding interval is reversed")
    axis = frames[start : end + 1]
    eligible_ids = {f["frame_id"] for f in axis}
    selected = [
        r
        for r in rows
        if r["frame_id"] in eligible_ids
        and r["hand"] == interval["hand"]
        and r["object_id"] == interval["object_id"]
    ]
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["frame_id"]].append(row)
    support, contact = {}, {}
    for frame in axis:
        values = grouped[frame["frame_id"]]
        if len(values) != 5 or {r["fingertip"] for r in values} != TIPS:
            raise ValueError("Incomplete holding-interval evidence")
        support[frame["frame_id"]] = all(r["valid"] for r in values)
        contact[frame["frame_id"]] = any(
            r["valid"] and r["raw_contact"] is True for r in values
        )
    gap = longest_gap(axis, support)
    usable = sum(support.values()) / len(axis)
    raw = sum(contact.values()) / len(axis)
    metrics_pass = (
        usable >= 0.90 and raw >= 0.80 and (gap is None or gap["seconds"] <= 0.50)
    )
    return {
        "interval": interval,
        "eligible_frames": len(axis),
        "usable_fraction": usable,
        "raw_contact_fraction": raw,
        "longest_unsupported_gap": gap,
        "status": "metrics_passed_requires_geometry_and_event_review"
        if metrics_pass
        else "failed_metrics",
        "quality_acceptance": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for episode in EPISODES:
        name = f"e{episode:03d}"
        path = (
            args.evidence_root
            / "issue6-full-runs/all20-final-v3"
            / name
            / "contacts/contacts.json"
        )
        cad_path = (
            args.evidence_root
            / "pr32-evidence/all20-final-review-v3"
            / name
            / "cad-temporal-hands-receipt.json"
        )
        payload = path.read_bytes()
        rows = json.loads(payload)
        frames = json.loads(cad_path.read_text())["frames"]
        results.append(
            {
                "episode": episode,
                "contacts_sha256": hashlib.sha256(payload).hexdigest(),
                "source_axis_sha256": hashlib.sha256(cad_path.read_bytes()).hexdigest(),
                **measure(rows, frames),
                "quality_status": "pending_visible_grasp_annotation_and_review",
            }
        )
    result = {
        "status": "baseline_coverage_measured_quality_pending",
        "episodes": results,
        "quality_passes": 0,
        "interpretation": "All-source support includes out-of-view/occluded frames and is not visible-grasp recall.",
        "required": "Source-video holding intervals, current geometry/event review and all20 quality passes.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in sorted(results, key=lambda r: r["valid_rows"] / r["contact_rows"]):
        print(
            json.dumps(
                {
                    "episode": row["episode"],
                    "supported_rows_pct": round(
                        100 * row["valid_rows"] / row["contact_rows"], 1
                    ),
                    "reasons": {
                        hand: next(
                            x["unsupported_frame_reasons"]
                            for x in row["hand_object"]
                            if x["hand"] == hand
                        )
                        for hand in ("left", "right")
                    },
                    "longest_gap_s": round(
                        max(
                            (x["longest_unsupported_source_gap"] or {"seconds": 0})[
                                "seconds"
                            ]
                            for x in row["hand_object"]
                        ),
                        3,
                    ),
                }
            )
        )


if __name__ == "__main__":
    main()
