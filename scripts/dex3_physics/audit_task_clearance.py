"""Check both inferred hand paths against other objects and each other on cooked hulls.

This samples nominal commands, not measured dynamics or a continuous collision
certificate. Intended own-object contact is excluded from these diagnostics.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from cooked_grasp_audit import convex_escape, world_convexes
from grasp_preload import URDFKinematics

sys.path.insert(0, str(Path(__file__).parents[1] / "reproduction"))
from _dex3_task_grip import TaskGripReference


def audit(spec_file, cooked, urdfs):
    spec = json.loads(spec_file.read_text())
    if spec["schema"] != "v2d.dex3-task-grips.v1" or spec["physical_task_accepted"]:
        raise ValueError("Need an explicit inferred task reference")
    colliders = json.loads(cooked.read_text())["colliders"]
    geometry = {}
    kinematics = {}
    for side, asset in [("left", "hand-0"), ("right", "hand-2")]:
        geometry[side] = {
            c["body_name"]: world_convexes(c, [0, 0, 0, 0, 0, 0, 1])
            for c in colliders
            if c["asset"] == asset
        }
        if not geometry[side]:
            raise ValueError("Missing canonical cooked hand")
        kinematics[side] = URDFKinematics(urdfs / f"dex3_{side}.urdf")
    ref = TaskGripReference(spec)
    objects = spec["smooth_robot_object_reference"]
    local_objects = spec["scene_projection"]["expected_body_local_convexes"]
    worst = {
        key: dict(penetration_m=0.0)
        for key in ["left_other_object", "right_other_object", "hand_hand"]
    }
    hand_clearance_failures = []

    def compare(key, a, b, time, name):
        for first in a:
            for second in b:
                gap = np.maximum(
                    np.maximum(
                        first.min(0) - second.max(0), second.min(0) - first.max(0)
                    ),
                    0,
                )
                if key == "hand_hand" and np.linalg.norm(gap) < 0.003:
                    depth, _ = convex_escape(first, second)
                    if depth > -0.003:
                        hand_clearance_failures.append(
                            dict(
                                source_time_s=float(time),
                                links=name,
                                signed_depth_m=float(depth),
                            )
                        )
                if np.any(first.min(0) > second.max(0)) or np.any(
                    second.min(0) > first.max(0)
                ):
                    continue
                depth, _ = convex_escape(first, second)
                if depth > worst[key]["penetration_m"]:
                    worst[key] = dict(
                        penetration_m=float(depth),
                        source_time_s=float(time),
                        links=name,
                    )

    for i, time in enumerate(spec["source_times_s"]):
        poses = {oi: np.array(p[i]) for oi, p in objects.items()}
        object_hulls = {
            oi: [
                Rotation.from_quat(poses[oi][3:]).apply(v) + poses[oi][:3]
                for v in local_objects[int(oi)]
            ]
            for oi in poses
        }
        hands = {}
        for side in ("left", "right"):
            phases = spec["phases"][side]
            oi = str(phases[0]["object"])
            command = ref.command(side, time, poses[oi], commit=False)
            phase = next(p for p in phases if p["sample"] == command["phase"])
            fk = kinematics[side].poses(
                command["pose"],
                dict(zip(phase["joint_names"], command["fingers"], strict=True)),
            )
            hands[side] = {
                name: [v @ fk[name][:3, :3].T + fk[name][:3, 3] for v in pieces]
                for name, pieces in geometry[side].items()
            }
            for other, hulls in object_hulls.items():
                if other == oi:
                    continue
                for name, pieces in hands[side].items():
                    compare(side + "_other_object", pieces, hulls, time, name)
        for a, first in hands["left"].items():
            for b, second in hands["right"].items():
                compare("hand_hand", first, second, time, [a, b])
    return dict(
        schema="v2d.nominal-bimanual-clearance-audit.v1",
        task_spec_sha256=hashlib.sha256(spec_file.read_bytes()).hexdigest(),
        cooked_geometry_sha256=hashlib.sha256(cooked.read_bytes()).hexdigest(),
        sampled_source_frames=len(spec["source_times_s"]),
        worst=worst,
        hand_clearance_at_least_3mm_on_sampled_frames=not hand_clearance_failures,
        hand_clearance_failed_piece_pairs=len(hand_clearance_failures),
        closest_hand_pair_samples=sorted(
            hand_clearance_failures, key=lambda row: row["signed_depth_m"], reverse=True
        )[:5],
        measured_dynamics=False,
        continuous_collision_certificate=False,
        physical_task_accepted=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("spec", "cooked", "urdfs", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    r = audit(a.spec, a.cooked, a.urdfs)
    with a.out.open("x") as f:
        json.dump(r, f, indent=2)
        f.write("\n")
    print(json.dumps(r, indent=2))
