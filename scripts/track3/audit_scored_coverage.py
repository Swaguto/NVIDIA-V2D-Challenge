"""Audit scored video frames, object slots, assets and policy readiness offline."""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from acquire_scored_inputs import hashes


def official_frames(kit, roster):
    import pyarrow.parquet as pq

    frames = defaultdict(lambda: defaultdict(set))
    for row in (
        pq.read_table(
            kit / "data/track_3_sample_submission.parquet", columns=["row_id"]
        )
        .column(0)
        .to_pylist()
    ):
        m = re.fullmatch(r"track_3/e(\d+)/w000/f(\d+)/o(\d+)", row)
        if not m:
            raise ValueError("Unexpected official sample row ID")
        episode, frame, slot = map(int, m.groups())
        frames[episode][slot].add(frame)
    if set(frames) != set(map(int, roster)):
        raise ValueError("Official sample and object roster episodes differ")
    result = {}
    for episode, objects in roster.items():
        slots = frames[int(episode)]
        if set(slots) != set(range(len(objects))):
            raise ValueError("Official sample and object slots differ")
        axis = slots[0]
        if axis != set(range(len(axis))) or any(
            values != axis for values in slots.values()
        ):
            raise ValueError("Official sample does not have one complete frame axis")
        result[int(episode)] = len(axis)
    return result


def probe_video(path):
    if not path.is_file():
        return {"status": "missing"}
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "status": "decoded",
        "frames": int(stream["nb_read_frames"]),
        "width": stream["width"],
        "height": stream["height"],
        "nominal_rate": stream["r_frame_rate"],
        "sha256": hashes(path)[0],
    }


def audit(kit, data, work):
    roster_path = kit / "data/track_3_evaluation_objects.json"
    roster = json.loads(roster_path.read_text())["episodes"]
    expected = official_frames(kit, roster)
    rows = []
    for episode, names in sorted(roster.items(), key=lambda item: int(item[0])):
        episode = int(episode)
        cameras = {
            cam: probe_video(
                data
                / "videos/chunk-000"
                / f"observation.images.ego_cam_{cam}"
                / f"episode_{episode:06d}.mp4"
            )
            for cam in "abc"
        }
        complete = all(c.get("frames") == expected[episode] for c in cameras.values())
        geometry = []
        for slot, name in enumerate(names):
            choices = [
                data / "mesh" / name / f"{name}{suffix}.glb"
                for suffix in ("", "_visual", "_collision")
            ]
            provided = next((p for p in choices if p.is_file()), None)
            geometry.append(
                {
                    "slot": slot,
                    "name": name,
                    "released_asset_status": "available" if provided else "missing",
                    "released_asset_sha256": hashes(provided)[0] if provided else None,
                    "reconstructed_mesh_status": "not_verified",
                    "reconstruction_provenance_status": "not_verified",
                    "body_frame_units_status": "not_verified_for_rollout",
                    "work_owner": "current pipeline work",
                }
            )
        ep_root = work / f"e{episode:03d}"
        rows.append(
            {
                "episode": episode,
                "expected_frames": expected[episode],
                "object_slot_names": names,
                "videos": cameras,
                "complete_video_frame_axis": complete,
                "geometry": geometry,
                "bundle_status": "not_verified",
                "policy_status": "not_verified",
                "official_evaluation_status": "not_run",
                "work_owner": "current pipeline work",
                "stage_reports_present": {
                    stage: (ep_root / stage / "report.json").is_file()
                    for stage in ("camera", "masks")
                },
            }
        )
    return {
        "schema": "v2d.scored-coverage-audit.v1",
        "official_roster_sha256": hashes(roster_path)[0],
        "episodes": rows,
        "summary": {
            "scored_episodes": len(rows),
            "object_mesh_pairs": sum(len(r["geometry"]) for r in rows),
            "video_files": sum(len(r["videos"]) for r in rows),
            "episodes_with_complete_video_frames": sum(
                r["complete_video_frame_axis"] for r in rows
            ),
            "pairs_with_released_assets": sum(
                g["released_asset_status"] == "available"
                for r in rows
                for g in r["geometry"]
            ),
            "pairs_missing_released_assets": sum(
                g["released_asset_status"] == "missing"
                for r in rows
                for g in r["geometry"]
            ),
        },
        "hidden_ground_truth_used": False,
        "official_score_claimed": False,
        "submission_ready": False,
        "status_policy": "Not verified does not mean absent; acquired CAD is not labeled participant reconstruction",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kit", "data", "work", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.kit, args.data, args.work)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))
