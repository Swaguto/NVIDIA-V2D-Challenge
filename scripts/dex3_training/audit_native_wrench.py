"""Audit actual hand contact load capacity separately from support contact.

All forces use the pinned runtime convention: force acts on body0, with the
normal directed from body0 to body1. Uses recorded contacts without simulation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.wrench_capacity import capacity


def audit(run, out, side="left", object_name="white_pot", stride_frames=5):
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((run / "report.json").read_text())
    labels = json.loads((run / "body-labels.json").read_text())
    state = np.load(run / "native-state-readback.npz")
    model = np.load(run / "native-model-readback.npz")
    contacts = np.load(run / "native-contacts-readback.npz")
    poses = state["body_poses"]
    worlds = report["metrics"]["world_count"]
    bodies = len(labels) // worlds
    local = labels[:bodies]
    oi = local.index(object_name + "/object")
    mass = float(model["body_mass"][oi + 1])
    hand = np.array(
        [i for i, label in enumerate(local) if label.startswith("dex3_" + side + "/")]
    )
    if not len(hand) or stride_frames < 1:
        raise ValueError("Need a mapped hand and positive frame stride")
    # MuJoCo mixes friction componentwise by maximum at equal priorities.
    # Read actual per-geom values and require a single sliding coefficient.
    geom_body = model["geom_bodyid"] - 1
    friction = model["geom_friction"][np.isin(geom_body, np.r_[hand, oi]), 0]
    if not np.allclose(friction, friction[0]):
        raise ValueError("Mixed contact friction needs explicit per-contact metadata")
    mu = float(friction[0])
    rows = []
    for frame in range(0, len(poses), stride_frames):
        start, end = contacts["frame_offsets"][frame : frame + 2]
        a, b = contacts["body0_ids"][start:end], contacts["body1_ids"][start:end]
        point = contacts[
            "force_application_pos_w"
            if "force_application_pos_w" in contacts
            else "contact_pos_w"
        ][start:end]
        force = contacts["contact_force_w"][start:end]
        normal = contacts["contact_normal_w"][start:end]
        torque = contacts["contact_torque_w"][start:end]
        for world in range(worlds):
            obj = oi + world * bodies
            assigned = ((a == obj) & np.isin(b, hand + world * bodies)) | (
                (b == obj) & np.isin(a, hand + world * bodies)
            )
            sign = np.where(a[assigned] == obj, 1.0, -1.0)
            f = force[assigned] * sign[:, None]
            inward = -normal[assigned] * sign[:, None]
            budget = np.maximum(0, np.einsum("ij,ij->i", f, inward))
            active = budget > 0.01
            pose = poses[frame, obj]
            com = pose[:3] + Rotation.from_quat(pose[3:]).apply(
                model["body_ipos"][oi + 1]
            )
            observed_wrench = np.r_[
                f.sum(0),
                (
                    np.cross(point[assigned] - com, f)
                    + torque[assigned] * sign[:, None]
                ).sum(0),
            ]
            load = np.array([0, 0, mass * 9.81, 0, 0, 0])
            solved = capacity(
                point[assigned][active], inward[active], budget[active], com, load, mu
            )
            other = np.where(a == obj, b, a)
            external = ((a == obj) | (b == obj)) & ~assigned
            external_sign = np.where(a[external] == obj, 1.0, -1.0)
            support = np.array(
                [
                    local[int(x) % bodies] == "support" if x >= 0 else True
                    for x in other[external]
                ],
                dtype=bool,
            )
            support_force = (
                force[external][support] * external_sign[support, None]
            ).sum(0)
            rows.append(
                dict(
                    source_time_s=frame / report["control_fps"],
                    frame=frame,
                    world=world,
                    **solved,
                    measured_hand_wrench_about_object_com=observed_wrench.tolist(),
                    gravity_load_N=load[2],
                    measured_support_force_N=support_force.tolist(),
                    sampled_support_bears_weight=bool(support_force[2] > 0.1 * load[2]),
                    sampled_hand_could_balance_gravity_with_current_normal_budget=solved[
                        "load_multiplier"
                    ]
                    >= 1.0,
                )
            )
    result = dict(
        schema="v2d.native-contact-wrench-audit.v1",
        rows=rows,
        side=side,
        object_name=object_name,
        friction=mu,
        source_report_sha256=hashlib.sha256(
            (run / "report.json").read_bytes()
        ).hexdigest(),
        scope="Offline inscribed 8-edge point-contact friction pyramids with measured normal budgets; support excluded; no motor, compliance, torsional friction or dynamic feasibility certificate",
        force_application_position_source=(
            "Native MuJoCo contact midpoint"
            if "force_application_pos_w" in contacts
            else "Archived Newton surface point0 approximation"
        ),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physical_task_accepted=False,
    )
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--object", default="white_pot")
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    result = audit(args.run, args.out, args.side, args.object, args.stride)
    print(json.dumps(dict(frames=len(result["rows"]), output=str(args.out))))
