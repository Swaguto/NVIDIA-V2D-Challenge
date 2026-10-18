"""Build a private, portable pilot from the longest fully confident source window.

Window selection uses source confidence only, never validation contact intervals.
The entire window is retained at source rate. No targets are filled or fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def longest_window(valid):
    valid = np.asarray(valid)
    if valid.ndim != 1 or valid.dtype != bool:
        raise ValueError("Expected a one-dimensional confidence mask")
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    intervals = list(
        zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    )
    if not intervals:
        raise ValueError("No complete confident interval")
    return tuple(map(int, max(intervals, key=lambda pair: pair[1] - pair[0])))


def select_rows(valid, qpos, *, full_episode=False):
    """Keep source confidence separate from explicitly inferred robot commands."""
    valid = np.asarray(valid)
    qpos = np.asarray(qpos)
    if valid.ndim != 1 or valid.dtype != bool or qpos.shape != (len(valid), 2, 14):
        raise ValueError("Source confidence and dual-hand command axes differ")
    if full_episode:
        if len(valid) < 2 or not np.isfinite(qpos).all():
            raise ValueError(
                "Full episodes need finite inferred commands for every frame"
            )
        if not np.allclose(np.linalg.norm(qpos[:, :, 3:7], axis=-1), 1, atol=1e-4):
            raise ValueError("Full-episode wrist quaternions must be normalized")
        return 0, len(valid)
    start, stop = longest_window(valid)
    if stop - start < 20:
        raise ValueError("Need at least one second of fully confident source targets")
    return start, stop


def prepare(case, static_inputs, robot_assets, out, *, full_episode=False):
    declared = json.loads((case / "file-manifest.json").read_text())["files_sha256"]
    for relative in (
        "bundle/episode.npz",
        "bundle/manifest.json",
        "proposal/robot-motion-proposal.npz",
        "retarget/manifest.json",
        "support/inferred-support-sheet.npz",
    ):
        if sha(case / relative) != declared[relative]:
            raise ValueError(f"Source bytes changed: {relative}")
    static = json.loads((static_inputs / "manifest.json").read_text())
    if (
        static["source_bundle_payload_sha256"] != declared["bundle/episode.npz"]
        or static["motion_payload_sha256"]
        != declared["proposal/robot-motion-proposal.npz"]
    ):
        raise ValueError("Geometry/motion source mismatch")
    metadata = json.loads((case / "bundle/manifest.json").read_text())
    contract = json.loads((case / "retarget/manifest.json").read_text())
    with np.load(case / "bundle/episode.npz", allow_pickle=False) as data:
        source = {k: data[k].copy() for k in data.files}
    with np.load(
        case / "proposal/robot-motion-proposal.npz", allow_pickle=False
    ) as data:
        qpos = data["robot_qpos_inferred"].copy()
        if full_episode and (
            data["robot_pose_is_inferred"].shape != qpos.shape[:2]
            or not data["robot_pose_is_inferred"].all()
            or data["robot_motion_target_validated"].any()
            or not np.array_equal(data["frame_indices"], source["frame_indices"])
        ):
            raise ValueError(
                "Full-episode proposal must retain its inferred robot provenance and complete frame axis"
            )
    hand_mask_key = (
        "hand_metric_valid" if "hand_metric_valid" in source else "hand_valid"
    )
    valid = source[hand_mask_key].all(axis=1) & source["hand_directly_observed"].all(
        axis=1
    )
    for obj in metadata["objects"]:
        key = f"object_{obj['index']}_valid"
        if key not in source:
            provenance = metadata["object_observation_provenance"][obj["name"]]
            source[key] = np.array(
                [provenance.get(str(f)) == "observed" for f in source["frame_indices"]]
            )
        valid &= source[key]
    valid &= np.isfinite(qpos).all(axis=(1, 2))
    start, stop = select_rows(valid, qpos, full_episode=full_episode)
    if full_episode and not np.array_equal(
        source["frame_indices"], np.arange(len(valid))
    ):
        raise ValueError(
            "Full-episode commands must cover every original frame and remain inferred"
        )
    indices = source["frame_indices"][start:stop]
    times = source["timestamps_s"][start:stop]
    if not np.all(np.diff(indices) == 1) or not np.allclose(
        np.diff(times), 1 / metadata["fps"]
    ):
        raise ValueError("Nonuniform source axis")
    rotation = Rotation.from_matrix(
        static["stage_rotation_from_inferred_support_normal"]
    )
    q = qpos[start:stop].copy()
    q[:, :, :3] = rotation.apply(q[:, :, :3].reshape(-1, 3)).reshape(q.shape[0], 2, 3)
    q[:, :, 3:7] = (
        (rotation * Rotation.from_quat(q[:, :, 3:7].reshape(-1, 4)))
        .as_quat()
        .reshape(q.shape[0], 2, 4)
    )
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    if full_episode:
        report = case / "proposal/report.json"
        if sha(report) != declared.get("proposal/report.json"):
            raise ValueError("Command proposal report changed or is undeclared")
        shutil.copyfile(report, out / "command-proposal-report.json")
    # Copy only the private robot asset closure; dataset inputs stay out of images.
    robot_sources = {}
    for side in ("left", "right"):
        urdf = robot_assets / f"urdfs/g1/dex3_{side}.urdf"
        tree = ET.parse(urdf)
        for mesh in tree.findall(".//mesh"):
            name = mesh.attrib["filename"]
            if not name.startswith("package://g1/"):
                raise ValueError("Unexpected robot mesh resolver")
            relative = name.removeprefix("package://g1/")
            original = robot_assets / "meshes/g1" / relative
            dest = out / "robot/meshes/g1" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, dest)
            mesh.set("filename", "../../meshes/g1/" + relative)
        dest = out / f"robot/urdfs/g1/dex3_{side}.urdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tree.write(dest, encoding="utf-8", xml_declaration=True)
        robot_sources[side] = sha(urdf)
    poses, objects = [], []
    for obj in metadata["objects"]:
        oi = obj["index"]
        matches = [o for o in static["objects"].values() if o["name"] == obj["name"]]
        if len(matches) != 1:
            raise ValueError("Need one prepared rigid object per source object")
        asset = matches[0]
        scales = source[f"object_{oi}_scale"][start:stop]
        if np.max(np.abs(scales / np.median(scales) - 1)) > 1e-6:
            raise ValueError("Pilot only supports rigid fixed-scale objects")
        for kind in ("mesh", "urdf"):
            if sha(static_inputs / asset[kind]) != asset[kind + "_sha256"]:
                raise ValueError("Prepared asset changed")
            dest = out / asset[kind]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(static_inputs / asset[kind], dest)
        matrix = source[f"object_{oi}_world"][start:stop]
        if not np.isfinite(matrix).all():
            raise ValueError(
                "Object references need explicit finite inferred poses, with confidence retained separately"
            )
        position = rotation.apply(matrix[:, :3, 3])
        quaternion = (rotation * Rotation.from_matrix(matrix[:, :3, :3])).as_quat()
        poses.append(np.c_[position, quaternion])
        radius = float(
            np.linalg.norm(
                source[f"object_{oi}_vertices"] * np.median(scales), axis=1
            ).max()
        )
        objects.append(
            dict(
                name=obj["name"],
                mesh=asset["mesh"],
                urdf=asset["urdf"],
                radius_m=radius,
            )
        )
    frames, frame_names, joint_names = {}, {}, {}
    for si, side in enumerate(("left", "right")):
        kin = URDFKinematics(out / f"robot/urdfs/g1/dex3_{side}.urdf")
        joint_names[side] = contract["joints"][side]["finger_joint_names"]
        frame_names[side] = [f"{side}_hand_palm_link"] + [
            f"{side}_hand_{digit}_{2 if digit == 'thumb' else 1}_link"
            for digit in ("thumb", "middle", "index")
        ]
        frames[side] = np.array(
            [
                [fk[name][:3, 3] for name in frame_names[side]]
                for fk in [
                    kin.poses(
                        row[:7], dict(zip(joint_names[side], row[7:], strict=True))
                    )
                    for row in q[:, si]
                ]
            ]
        )
    with np.load(
        case / "support/inferred-support-sheet.npz", allow_pickle=False
    ) as support:
        vertices = rotation.apply(support["vertices_world_m"])
        faces = support["faces"].copy()
    # The upstream support reader accepts a colliding static USD mesh.
    points = ", ".join(f"({x:.10g}, {y:.10g}, {z:.10g})" for x, y, z in vertices)
    (out / "support.usda").write_text(
        '#usda 1.0\n( metersPerUnit = 1\n upAxis = "Z" )\n'
        'def Xform "Support" {\n def Mesh "surface" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {\n'
        f"  point3f[] points = [{points}]\n"
        f"  int[] faceVertexCounts = {json.dumps([3] * len(faces))}\n"
        f"  int[] faceVertexIndices = {json.dumps(faces.ravel().tolist())}\n"
        '  bool physics:collisionEnabled = true\n  uniform token subdivisionScheme = "none"\n }\n}\n'
    )
    np.savez_compressed(
        out / "targets.npz",
        robot_qpos=q,
        object_poses=np.stack(poses, axis=1),
        left_frames=frames["left"],
        right_frames=frames["right"],
        confidence_valid=valid[start:stop],
        frame_indices=indices,
        timestamps_s=times,
    )
    # Preserve all original masks separately, including the excluded rows.
    np.savez_compressed(
        out / "source-confidence.npz",
        **{
            k: v
            for k, v in source.items()
            if k
            in {
                "hand_metric_valid",
                "hand_directly_observed",
                "frame_indices",
                "timestamps_s",
            }
            or k.endswith("_valid")
            or k in {"contact_known", "contact_probability", "wrench_known"}
        },
    )
    pq.write_table(
        pa.Table.from_pylist(
            [dict(schema_version="dex3_dual_hand_v1", fps=float(metadata["fps"]))]
        ),
        out / "reference.parquet",
    )
    manifest = dict(
        schema="v2d.dex3-training-retargeted-task.v4"
        if full_episode
        else "v2d.dex3-training-window.v1",
        sequence_id=f"e{metadata['episode']:03d}_"
        + ("full_retargeted" if full_episode else "pilot"),
        episode=metadata["episode"],
        source_rows_half_open=[start, stop],
        source_frame_indices=indices.tolist(),
        source_frame_count=len(valid),
        selected_frame_count=stop - start,
        selection=(
            "Every source frame; inferred retargeted commands and object poses; original source confidence preserved"
            if full_episode
            else f"longest contiguous interval with both hands {hand_mask_key}/directly observed and every object observed"
        ),
        all_selected_targets_confident=bool(valid[start:stop].all()),
        contacts_known=False,
        source_observations_changed=False,
        contact_labels_fabricated=False,
        object_mass_inertia="existing nominal generator defaults, not measurements",
        gravity_orientation="inferred support normal, not independent gravity measurement",
        objects=objects,
        frame_names=frame_names,
        joint_names=joint_names,
        robot_source_urdf_sha256=robot_sources,
        robot_urdf_change="package:// mesh paths rewritten to equivalent portable relative paths only",
        source_files_sha256={
            k: declared[k]
            for k in declared
            if k
            in {
                "bundle/episode.npz",
                "proposal/robot-motion-proposal.npz",
                "support/inferred-support-sheet.npz",
            }
        },
        files={
            p.relative_to(out).as_posix(): sha(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
        physical_task_accepted=False,
        full_episode_covered=full_episode,
    )
    if full_episode:
        manifest.update(
            target_contract="inferred_retargeted_robot_commands.v1",
            allows_inferred_robot_targets=True,
            observed_source_frame_count=int(valid.sum()),
            hand_object_assignment="not prescribed; generic retargeted motion baseline",
            grasp_plan_used=False,
            source_motion_speed=1.0,
            command_anchor_policy=static.get("command_anchor_policy", "visual"),
            command_selected_anchor_slots=static.get("selected_anchor_slots"),
            command_source_report_sha256=sha(out / "command-proposal-report.json"),
            target_preparation_source_sha256=sha(Path(__file__)),
        )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "static-inputs", "robot-assets", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--full-episode",
        action="store_true",
        help="Keep all inferred robot commands; never promote unknown source confidence",
    )
    args = parser.parse_args()
    result = prepare(
        args.case,
        args.static_inputs,
        args.robot_assets,
        args.out,
        full_episode=args.full_episode,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
