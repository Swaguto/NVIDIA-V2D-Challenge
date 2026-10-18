"""Inspect native rigid-object collisions at every reference knot on CPU.

The finite support and convex collision geometry are the same as the native
runtime. This teleports objects to reference poses and never advances physics;
robot contacts, continuous swept collisions and grasp success are not assessed.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from audit_initial_contacts import reference_contacts, summarize_contacts
from check_native_scene import compile_scene


def summarize_trajectory(frames, tolerance_m=0.001):
    if not frames or [row["reference_frame"] for row in frames] != list(
        range(len(frames))
    ):
        raise ValueError("Need every ordered reference knot, including empty contacts")
    pairs = {}
    for row in frames:
        seen = set()
        for contact in row["contact_pairs"]:
            key = tuple(contact["bodies"])
            depth = contact["maximum_penetration_m"]
            if key in seen or not np.isfinite(depth) or depth < 0:
                raise ValueError("Invalid or repeated reference contact pair")
            seen.add(key)
            pairs.setdefault(key, []).append((row["reference_frame"], depth))
    return [
        {
            "bodies": list(pair),
            "frames_over_tolerance": sum(depth > tolerance_m for _, depth in values),
            "maximum_penetration_m": max(depth for _, depth in values),
            "worst_reference_frame": max(values, key=lambda item: item[1])[0],
        }
        for pair, values in sorted(pairs.items())
    ]


def audit(inputs):
    start = time.monotonic()
    scene, reference, solver, manifest_sha, runtime_commit = compile_scene(inputs)
    count = len(reference.object_body_pos_w())
    frames = []
    for frame in range(count):
        objects, support, rows = reference_contacts(scene, reference, solver, frame)
        frames.append(
            {
                "reference_frame": frame,
                "contact_pairs": summarize_contacts(rows, objects, support),
            }
        )
    return {
        "schema": "v2d.native-reference-object-contact-audit.v1",
        "input_manifest_sha256": manifest_sha,
        "runtime_commit": runtime_commit,
        "reference_knots": count,
        "reference_object_poses_verified": True,
        "objects": [obj.name for obj in scene.objects],
        "reporting_tolerance_m": 0.001,
        "summary": summarize_trajectory(frames),
        "frames": frames,
        "physics_stepped": False,
        "robot_contacts_assessed": False,
        "native_mesh_checks_relaxed": False,
        "source_targets_modified": False,
        "physical_task_accepted": False,
        "scope": "CPU native finite-support/object collision detection at every reference knot; no solver step, swept contact or robot grasp certificate",
        "wall_seconds": time.monotonic() - start,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.inputs.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
