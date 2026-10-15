"""Export the preserved Dex3 target contract to diagnostic motion_v1 references.

Unknown geometry stays NaN; observation/contact masks remain in a bound sidecar.
The existing dual-hand trainer does not consume those masks, so this exporter
does not declare any file training-ready or fabricate legacy MANO contacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robotic_grounding/source/robotic_grounding"))
from robotic_grounding.motion_schema import DUAL_HAND, MotionData, save_motion_parquet
from robotic_grounding.retarget.track3_bundle import SIDES, digest, validate_bundle


def pose_wxyz(pose):
    """Convert explicitly declared xyzw poses; preserve NaN rows without inference."""
    result = np.asarray(pose).copy()
    result[..., 3:7] = result[..., [6, 3, 4, 5]]
    return result


def export(bundle: Path, retarget: Path, out: Path) -> dict:
    verified = validate_bundle(bundle)
    metadata = json.loads((bundle / "manifest.json").read_text())
    contract = json.loads((retarget / "manifest.json").read_text())
    if contract["episode_bundle_manifest_sha256"] != verified[
        "manifest_sha256"
    ] or contract["payload"]["sha256"] != digest(retarget / "retarget.npz"):
        raise ValueError("Source/retarget provenance differs")
    with np.load(bundle / "episode.npz", allow_pickle=False) as data:
        source = {key: data[key].copy() for key in data.files}
    with np.load(retarget / "retarget.npz", allow_pickle=False) as data:
        target = {key: data[key].copy() for key in data.files}
    np.testing.assert_array_equal(source["frame_indices"], target["frame_indices"])
    np.testing.assert_array_equal(source["timestamps_s"], target["timestamps_s"])
    times = source["timestamps_s"]
    if len(times) < 2 or not np.allclose(
        np.diff(times), 1 / metadata["fps"], rtol=1e-5, atol=1e-6
    ):
        raise ValueError("motion_v1 requires a uniform, complete source time axis")
    qpos = target["robot_qpos"]
    np.testing.assert_array_equal(target["robot_valid"], source["hand_valid"])
    md = MotionData(
        sequence_id=f"track3_e{metadata['episode']:03d}",
        robot_name="dex3",
        motion_kind=DUAL_HAND,
        fps=float(metadata["fps"]),
        coord_frame="original reconstruction world; metres; wxyz",
        source_dataset=metadata.get("source_dataset", "v2d_track3_unspecified_split"),
        raw_motion_file="hash-bound episode.npz and retarget.npz",
    )
    md.hand_sides = list(SIDES)
    md.ee_link_names = [f"{side}_hand_palm_link" for side in SIDES]
    md.ee_pose_w = pose_wxyz(qpos[:, :, :7])
    md.hand_frame_names = [contract["joints"][side]["frame_names"] for side in SIDES]
    md.hand_frames_w = [
        pose_wxyz(target[f"robot_{side}_frame_poses"]) for side in SIDES
    ]
    md.hand_finger_joint_names = [
        contract["joints"][side]["finger_joint_names"] for side in SIDES
    ]
    md.hand_finger_joints = [qpos[:, si, 7:].copy() for si in range(2)]
    md.object_body_names = [obj["name"] for obj in metadata["objects"]]
    matrices = np.stack(
        [source[f"object_{obj['index']}_world"] for obj in metadata["objects"]], axis=1
    )
    md.object_body_position = matrices[:, :, :3, 3]
    xyzw = (
        Rotation.from_matrix(matrices[:, :, :3, :3].reshape(-1, 3, 3))
        .as_quat()
        .reshape(len(times), -1, 4)
    )
    md.object_body_wxyz = xyzw[..., [3, 0, 1, 2]]
    # Do not map five human fingertip probabilities into seven-joint Dex3 contact
    # geometry, or encode unknown contact as measured no-contact.
    masks = {
        key: value
        for key, value in source.items()
        if key
        in {
            "frame_indices",
            "timestamps_s",
            "hand_valid",
            "hand_directly_observed",
            "hand_metric_valid",
            "contact_known",
            "contact_probability",
            "wrench_known",
            "wrench_world_N_Nm",
        }
        or key.endswith("_valid")
    }
    masks["robot_valid"] = target["robot_valid"]
    if out.exists():
        raise FileExistsError("Use a new diagnostic reference directory")
    out.mkdir(parents=True)
    partition = save_motion_parquet(
        md, out / "motion", file_name="reference.diagnostic.parquet"
    )
    np.savez_compressed(out / "source-masks.npz", **masks)
    report = {
        "schema": "v2d.dex3.reference-adapter.v1",
        "episode": metadata["episode"],
        "source_frames": len(times),
        "source_frame_axis_preserved": True,
        "source_bundle_manifest_sha256": verified["manifest_sha256"],
        "retarget_manifest_sha256": digest(retarget / "manifest.json"),
        "retarget_payload_sha256": digest(retarget / "retarget.npz"),
        "exporter_sha256": digest(Path(__file__)),
        "motion_path": str(
            (partition / "reference.diagnostic.parquet").relative_to(out)
        ),
        "motion_sha256": digest(partition / "reference.diagnostic.parquet"),
        "masks_sha256": digest(out / "source-masks.npz"),
        "finite_wrist_target_rows_by_hand": np.isfinite(qpos[:, :, :7])
        .all(axis=-1)
        .sum(axis=0)
        .tolist(),
        "contact_or_wrench_fabricated": False,
        "training_ready": False,
        "physical_task_accepted": False,
        "remaining_gates": [
            "complete physically reviewed targets",
            "Dex3 training task registration",
            "confidence-aware dual-hand motion_v1 consumer",
            "contact geometry/asset adapter",
            "policy checkpoint compatibility with the official evaluator",
        ],
    }
    (out / "reference-report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--retarget", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.bundle, args.retarget, args.out), indent=2))


if __name__ == "__main__":
    main()
