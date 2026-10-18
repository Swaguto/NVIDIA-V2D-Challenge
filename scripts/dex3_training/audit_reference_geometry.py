"""Check reconstructed and retargeted proximity before spending on policy training.

Distances are unsigned nearest-vertex diagnostics. They do not establish grip,
penetration, force closure, or observed contact. Intentional releases remain in
full-episode summaries. Source human summaries require both hand metric support
and an observed object pose; inferred object targets are reported separately.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"samples": 0, "median_m": None, "p95_m": None}
    return {
        "samples": len(values),
        "median_m": float(np.median(values)),
        "p95_m": float(np.quantile(values, 0.95)),
        "fraction_under_2cm": float(np.mean(values < 0.02)),
        "fraction_over_10cm": float(np.mean(values > 0.1)),
    }


def nearest(tree, points):
    points = np.asarray(points)
    if points.shape[-1] != 3:
        raise ValueError("Expected XYZ points")
    valid = np.isfinite(points).all(axis=-1)
    distances = np.full(points.shape[:-1], np.nan)
    distances[valid] = tree.query(points[valid])[0]
    return distances


def body_coordinates(points, matrices):
    if matrices.shape != (len(points), 4, 4):
        raise ValueError("Object transforms and source frames differ")
    extra_axes = (1,) * (points.ndim - 2)
    origin = matrices[:, :3, 3].reshape((len(points), *extra_axes, 3))
    return np.einsum("t...i,tij->t...j", points - origin, matrices[:, :3, :3])


def robot_tips(qpos, kin, ordered_joint_names, side):
    """The payload follows declared joint order, never XML traversal order."""
    if len(ordered_joint_names) != len(set(ordered_joint_names)) or set(
        ordered_joint_names
    ) != set(kin.names):
        raise ValueError("Declared finger joint names differ from the URDF")
    if (
        qpos.shape != (len(qpos), 7 + len(ordered_joint_names))
        or not np.isfinite(qpos).all()
    ):
        raise ValueError("Need finite declared root and finger commands")
    result = []
    for row in qpos:
        poses = kin.poses(row[:7], dict(zip(ordered_joint_names, row[7:], strict=True)))
        result.append(
            [
                poses[f"{side}_{digit}_tip"][:3, 3]
                for digit in ("thumb", "index", "middle")
            ]
        )
    return np.asarray(result)


def audit(bundle, inputs):
    meta = json.loads((bundle / "manifest.json").read_text())
    payload = meta["payload"]
    if sha(bundle / payload["path"]) != payload["sha256"]:
        raise ValueError("Bundle payload changed")
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = (inputs / name).resolve()
        if not path.is_relative_to(inputs.resolve()) or sha(path) != expected:
            raise ValueError("Training input checksum or path differs")
    with np.load(bundle / payload["path"], allow_pickle=False) as f:
        source = {k: f[k] for k in f.files}
    with np.load(inputs / "targets.npz", allow_pickle=False) as f:
        targets = {k: f[k] for k in f.files}
    np.testing.assert_array_equal(source["frame_indices"], targets["frame_indices"])
    np.testing.assert_allclose(
        source["timestamps_s"], targets["timestamps_s"], atol=1e-8, rtol=0
    )
    if [o["name"] for o in meta["objects"]] != [o["name"] for o in manifest["objects"]]:
        raise ValueError("Object ordering differs")
    report = {
        "schema": "v2d.reference-geometry-diagnostic.v1",
        "episode": meta["episode"],
        "bundle_manifest_sha256": sha(bundle / "manifest.json"),
        "input_manifest_sha256": sha(inputs / "manifest.json"),
        "diagnostic_source_sha256": sha(Path(__file__)),
        "source_frames": len(source["frame_indices"]),
        "hand_metric_frames": source["hand_metric_valid"].sum(0).tolist(),
        "objects": [],
        "contact_labels_created": False,
        "physical_task_accepted": False,
        "scope": __doc__,
    }
    robot = {
        side: robot_tips(
            targets["robot_qpos"][:, si],
            URDFKinematics(inputs / f"robot/urdfs/g1/dex3_{side}.urdf"),
            manifest["joint_names"][side],
            side,
        )
        for si, side in enumerate(("left", "right"))
    }
    for oi, obj in enumerate(meta["objects"]):
        if obj["index"] != oi:
            raise ValueError("Object index differs from payload ordering")
        tree = cKDTree(
            source[f"object_{oi}_vertices"] * source[f"object_{oi}_scale"][0]
        )
        observed = source[f"object_{oi}_valid"]
        hp = source["hand_joints_world"][:, :, [4, 8, 12]]
        hd = nearest(tree, body_coordinates(hp, source[f"object_{oi}_world"]))
        pose = targets["object_poses"][:, oi]
        matrix = np.tile(np.eye(4), (len(pose), 1, 1))
        matrix[:, :3, :3] = Rotation.from_quat(pose[:, 3:]).as_matrix()
        matrix[:, :3, 3] = pose[:, :3]
        item = {
            "name": obj["name"],
            "observed_frames": int(observed.sum()),
            "inferred_frames": int((~observed).sum()),
            "hands": {},
        }
        for si, side in enumerate(("left", "right")):
            supported = source["hand_metric_valid"][:, si] & observed
            rd = nearest(tree, body_coordinates(robot[side], matrix))
            item["hands"][side] = {
                "jointly_observed_hand_object_frames": int(supported.sum()),
                "human_nearest_of_three_on_supported_frames": summary(
                    hd[supported, si].min(-1)
                ),
                "human_median_of_three_on_supported_frames": summary(
                    np.median(hd[supported, si], axis=-1)
                ),
                "robot_median_of_three_on_same_supported_frames": summary(
                    np.median(rd[supported], axis=-1)
                ),
                "robot_median_of_three_all_frames_including_releases_and_inference": summary(
                    np.median(rd, axis=-1)
                ),
            }
        report["objects"].append(item)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "inputs", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    value = audit(a.bundle, a.inputs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(value, indent=2, allow_nan=False))
