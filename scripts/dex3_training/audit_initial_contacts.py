"""Inspect object/support collisions at the exact reference start on CPU.

No policy, stabilization or simulation step is run. Robot contacts are excluded
because the scene compiler has not reset the robot to the reference. This is an
initial object-geometry diagnostic, not a GPU physics or grasp certificate.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from check_native_scene import compile_scene


def summarize_contacts(rows, objects, support):
    pairs = {}
    for a, b, distance in rows:
        if not np.isfinite(distance):
            raise ValueError("Nonfinite initial contact distance")
        if a in objects and (b in objects or b == support):
            first, second = a, b
        elif b in objects and a == support:
            first, second = b, a
        else:
            continue
        key = tuple(
            sorted((objects.get(first, "support"), objects.get(second, "support")))
        )
        pairs.setdefault(key, []).append(distance)
    return [
        {
            "bodies": list(pair),
            "contact_samples": len(distances),
            "minimum_signed_gap_m": float(min(distances)),
            "maximum_penetration_m": float(max(0, -min(distances))),
        }
        for pair, distances in sorted(pairs.items())
    ]


def audit(inputs):
    import mujoco

    scene, reference, solver, manifest_sha, runtime_commit = compile_scene(inputs)
    model = solver.mj_model
    state = mujoco.MjData(model)
    objects = {}
    position, quaternion = reference.object_body_pos_w(), reference.object_body_quat_w()
    # Newton resets free object roots after compilation; MjModel.qpos0 is zero.
    # Apply the exact reference start explicitly before checking collisions.
    for obj in scene.objects:
        if obj.articulations:
            raise ValueError("This diagnostic currently covers rigid scored objects")
        bid, rid = obj.root.body_id + 1, obj.root.reference_body_id
        joint = int(model.body_jntadr[bid])
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Expected one native free root per scored object")
        start = int(model.jnt_qposadr[joint])
        state.qpos[start : start + 3] = position[0, rid]
        state.qpos[start + 3 : start + 7] = quaternion[0, rid]
    mujoco.mj_forward(model, state)
    for obj in scene.objects:
        for binding in obj.bodies:
            bid, rid = binding.body_id + 1, binding.reference_body_id
            if (
                not np.allclose(state.xpos[bid], position[0, rid], atol=1e-6, rtol=0)
                or min(
                    np.linalg.norm(state.xquat[bid] - quaternion[0, rid]),
                    np.linalg.norm(state.xquat[bid] + quaternion[0, rid]),
                )
                > 1e-6
            ):
                raise ValueError(
                    "Compiled body pose does not match the reference start: "
                    + json.dumps(
                        {
                            "newton_id": binding.body_id,
                            "mujoco_id": bid,
                            "native_name": model.body(bid).name,
                            "expected_name": obj.name,
                            "native_position": state.xpos[bid].tolist(),
                            "reference_position": position[0, rid].tolist(),
                            "native_quaternion_wxyz": state.xquat[bid].tolist(),
                            "reference_quaternion_wxyz": quaternion[0, rid].tolist(),
                            "solver_body_attributes": [
                                name
                                for name in dir(solver)
                                if "body" in name
                                and ("map" in name or "id" in name or "index" in name)
                            ],
                        }
                    )
                )
            objects[bid] = obj.name
    support = scene.support.body_id + 1 if scene.support is not None else None
    rows = [
        (
            int(model.geom_bodyid[c.geom1]),
            int(model.geom_bodyid[c.geom2]),
            float(c.dist),
        )
        for c in state.contact[: state.ncon]
    ]
    return {
        "schema": "v2d.native-initial-object-contact-audit.v1",
        "input_manifest_sha256": manifest_sha,
        "runtime_commit": runtime_commit,
        "reference_object_start_poses_verified": True,
        "reference_start_applied_to_native_free_joints": True,
        "objects": list(objects.values()),
        "contact_pairs": summarize_contacts(rows, objects, support),
        "physics_stepped": False,
        "robot_contacts_assessed": False,
        "native_mesh_checks_relaxed": False,
        "scope": "CPU MuJoCo contact detection on native convex collision geometry at reference frame zero; no GPU solver or policy step",
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physical_task_accepted": False,
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
    print(json.dumps(result, indent=2))
