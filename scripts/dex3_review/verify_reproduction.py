"""Verify a teammate's frozen-target replay using input hashes and existing tolerances."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(inputs: Path, outputs: Path, episodes: list[int]) -> dict:
    rows = []
    for ep in episodes:
        label = f"e{ep:03d}"
        source = inputs / "episodes" / label
        review = outputs / label / "review"
        receipt = json.loads((review / "isaac-receipt.json").read_text())
        bundle = json.loads((source / "bundle/manifest.json").read_text())
        hashes = json.loads((review / "delivery-hashes.json").read_text())
        for name, expected in hashes.items():
            path = review / name
            if (
                not path.resolve().is_relative_to(review.resolve())
                or digest(path) != expected
            ):
                raise ValueError(f"Changed or escaping delivery file: {label}/{name}")
        if receipt["episode"] != ep or receipt["source_frames"] != len(
            bundle["frames"]
        ):
            raise ValueError("Episode or source coverage differs")
        if receipt["bundle_manifest_sha256"] != digest(source / "bundle/manifest.json"):
            raise ValueError("Reproduction used different reconstruction inputs")
        if receipt["retarget_payload_sha256"] != digest(
            source / "retarget/retarget.npz"
        ):
            raise ValueError("Reproduction used different frozen IK targets")
        if receipt["source_sha256"] != digest(
            ROOT / "scripts/dex3_review/frozen/replay_continuity_dex3_sim6.py"
        ):
            raise ValueError("Reproduction used a different renderer source")
        for key in [
            "robot_asset_lock_verified",
            "source_hand_observation_masks_preserved",
            "all_displayed_robot_poses_are_inferred",
        ]:
            if receipt[key] is not True:
                raise ValueError(f"Missing replay provenance check: {key}")
        for key in [
            "gravity_enabled",
            "collision_resolution_enabled",
            "learned_policy_rollout",
            "physical_task_accepted",
        ]:
            if receipt[key] is not False:
                raise ValueError(f"This verifier covers kinematic replay only: {key}")
        limits = {
            key: 1e-4
            for key in [
                "maximum_root_position_error_m",
                "maximum_joint_error_rad",
                "maximum_root_orientation_error_rad",
                "maximum_rendered_usd_link_position_error_m",
                "maximum_physics_link_position_error_m",
                "maximum_camera_position_error_m",
                "maximum_camera_rotation_error_rad",
            ]
        }
        limits["maximum_camera_intrinsics_error_px"] = 1e-3
        for key, limit in limits.items():
            if not math.isfinite(receipt[key]) or not 0 <= receipt[key] <= limit:
                raise ValueError(
                    f"Replay readback exceeds the original tolerance: {key}"
                )
        media = json.loads((review / "media-verification.json").read_text())
        required = {
            "comparison.mp4",
            "human-input.mp4",
            "isaac-source.mp4",
            "isaac-oblique.mp4",
        }
        if set(media) != required:
            raise ValueError("Incomplete four-video review")
        for name, info in media.items():
            if (
                int(info["nb_read_frames"]) != len(bundle["frames"])
                or info["presentation_timestamps_strictly_increasing"] is not True
                or info["sha256"] != digest(review / name)
            ):
                raise ValueError(f"Media verification differs: {name}")
        rows.append(
            {
                "episode": ep,
                "frames": receipt["source_frames"],
                "simulator_receipt_sha256": digest(review / "isaac-receipt.json"),
                "input_and_frozen_targets_match": True,
            }
        )
    return {
        "schema": "v2d.team-reproduction-check.v1",
        "automated_replay_checks_pass": True,
        "episodes": rows,
        "full_manual_visual_review_performed": False,
        "physical_task_accepted": False,
        "cross_gpu_pixel_identity_required": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--episodes", default="11,40,41")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    episodes = [int(x) for x in args.episodes.split(",")]
    if not episodes or len(set(episodes)) != len(episodes):
        parser.error("Choose distinct episodes")
    report = verify(args.inputs, args.outputs, episodes)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
