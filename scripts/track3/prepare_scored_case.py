"""Freeze generic retargeted full-episode commands and nominal training assets.

Unknown human observations remain unknown. Gap completion is an explicit robot
command hypothesis; it supplies no grasp/contact labels or physical certificate.
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "scripts/reproduction"),
    str(ROOT / "robotic_grounding/source/robotic_grounding"),
]
from _dex3_static_load import load_functions
from robotic_grounding.retarget.track3_bundle import digest, validate_bundle


def complete_commands(qpos, valid, times, limits, parking):
    """Interpolate available IK; park a hand only when no fit exists for it."""
    qpos = np.asarray(qpos, float)
    valid = np.asarray(valid)
    times = np.asarray(times, float)
    if (
        qpos.shape != (len(times), 2, 14)
        or valid.shape != (len(times), 2)
        or valid.dtype != bool
        or len(times) < 2
        or not np.isfinite(times).all()
        or not (np.diff(times) > 0).all()
    ):
        raise ValueError("Need complete ordered source times and dual-hand IK validity")
    result = np.empty_like(qpos)
    interpolation = np.zeros(valid.shape, bool)
    parked = np.zeros(valid.shape, bool)
    for si, side in enumerate(("left", "right")):
        rows = np.flatnonzero(valid[:, si])
        low = np.asarray(limits[side]["lower_radians"])
        high = np.asarray(limits[side]["upper_radians"])
        if low.shape != (7,) or high.shape != (7,) or not (low <= high).all():
            raise ValueError("Need seven bounded Dex3 finger joints")
        if len(rows):
            selected = qpos[rows, si]
            if (
                not np.isfinite(selected).all()
                or not np.allclose(
                    np.linalg.norm(selected[:, 3:7], axis=1), 1, atol=1e-5
                )
                or (selected[:, 7:] < low - 1e-6).any()
                or (selected[:, 7:] > high + 1e-6).any()
            ):
                raise ValueError("Supported IK has invalid poses or finger limits")
            for col in [0, 1, 2, *range(7, 14)]:
                result[:, si, col] = np.interp(times, times[rows], selected[:, col])
            result[:, si, 3:7] = (
                np.repeat(selected[:, 3:7], len(times), axis=0)
                if len(rows) == 1
                else Slerp(times[rows], Rotation.from_quat(selected[:, 3:7]))(
                    np.clip(times, times[rows[0]], times[rows[-1]])
                ).as_quat()
            )
            result[rows, si] = selected
            interpolation[:, si] = ~valid[:, si]
        else:
            pose = np.asarray(parking[side], float)
            if (
                pose.shape != (7,)
                or not np.isfinite(pose).all()
                or not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-5)
            ):
                raise ValueError(
                    "Always-unobserved hand needs an explicit finite parking pose"
                )
            result[:, si, :7] = pose
            result[:, si, 7:] = np.clip(np.zeros(7), low, high)
            parked[:, si] = True
    return result, interpolation, parked


def prepare(bundle, retarget, support, out):
    validate_bundle(bundle)
    metadata = json.loads((bundle / "manifest.json").read_text())
    if (
        metadata.get("provided_meshes_read") is not False
        or metadata.get("reference_pose_labels_read") is not False
    ):
        raise ValueError(
            "Scored cases require participant reconstructed geometry and label-free targets"
        )
    contract = json.loads((retarget / "manifest.json").read_text())
    if (
        contract["episode_bundle_manifest_sha256"] != digest(bundle / "manifest.json")
        or digest(retarget / contract["payload"]["path"])
        != contract["payload"]["sha256"]
    ):
        raise ValueError("Retargeting belongs to different reconstruction bytes")
    plane = json.loads((support / "report.json").read_text())
    if plane["bundle_manifest_sha256"] != digest(bundle / "manifest.json") or plane[
        "payload_sha256"
    ] != digest(support / "support-plane-hypothesis.npz"):
        raise ValueError("Support hypothesis belongs to a different bundle")
    with np.load(bundle / "episode.npz", allow_pickle=False) as f:
        source = {k: f[k] for k in f.files}
    with np.load(retarget / contract["payload"]["path"], allow_pickle=False) as f:
        np.testing.assert_array_equal(f["frame_indices"], source["frame_indices"])
        np.testing.assert_allclose(
            f["timestamps_s"], source["timestamps_s"], atol=1e-8, rtol=0
        )
        q, valid = f["robot_qpos"].copy(), f["robot_valid"].copy()
    up = np.asarray(plane["plane_world"][:3], float)
    if not np.isclose(np.linalg.norm(up), 1, atol=1e-5):
        raise ValueError("Support normal is not normalized")
    stage = Rotation.align_vectors([[0.0, 0.0, 1.0]], up[None])[0]
    with np.load(support / "inferred-support-sheet.npz", allow_pickle=False) as f:
        center = f["vertices_world_m"].mean(0)
        with np.load(
            support / "support-plane-hypothesis.npz", allow_pickle=False
        ) as hypothesis:
            np.testing.assert_array_equal(
                f["vertices_world_m"], hypothesis["vertices_world_m"]
            )
            np.testing.assert_array_equal(f["faces"], hypothesis["faces"])
    right = source["camera_c2w"][0, :3, 0]
    right = right - up * (right @ up)
    if np.linalg.norm(right) < 0.01:
        raise ValueError(
            "Camera lateral direction cannot define an unseen-hand parking region"
        )
    right /= np.linalg.norm(right)
    parking = {
        side: np.r_[center + sign * 0.7 * right + 0.35 * up, stage.inv().as_quat()]
        for side, sign in [("left", -1), ("right", 1)]
    }
    q, inferred, parked = complete_commands(
        q, valid, source["timestamps_s"], contract["joints"], parking
    )
    if out.exists():
        raise FileExistsError(out)
    import shutil

    for name, folder in [
        ("bundle", bundle),
        ("retarget", retarget),
        ("support", support),
    ]:
        shutil.copytree(folder, out / "case" / name)
    proposal = out / "case/proposal"
    proposal.mkdir()
    np.savez_compressed(
        proposal / "robot-motion-proposal.npz",
        frame_indices=source["frame_indices"],
        timestamps_s=source["timestamps_s"],
        robot_qpos_inferred=q,
        robot_pose_is_inferred=np.ones(valid.shape, bool),
        robot_motion_target_validated=np.zeros(valid.shape, bool),
        robot_interpolated_from_supported_ik=inferred,
        robot_always_unobserved_parked=parked,
        source_robot_valid=valid,
    )
    static = out / "static-inputs"
    static.mkdir()
    generator = ROOT / "robotic_grounding/scripts/generate_rigid_urdfs.py"
    functions, signature = load_functions(
        generator, {"_generate_urdf"}, {"textwrap": textwrap}
    )
    objects = {}
    for obj in metadata["objects"]:
        oi = obj["index"]
        name = obj["name"]
        if not name.isidentifier():
            raise ValueError("Object name cannot be used as a portable URDF identifier")
        scale = source[f"object_{oi}_scale"]
        if not np.allclose(scale, scale[0], atol=1e-6, rtol=0):
            raise ValueError("Scored baseline requires a fixed rigid mesh scale")
        mesh = static / "objects" / (name + ".obj")
        mesh.parent.mkdir(exist_ok=True)
        trimesh.Trimesh(
            source[f"object_{oi}_vertices"] * scale[0],
            source[f"object_{oi}_faces"],
            process=False,
        ).export(mesh, include_normals=False, include_color=False, digits=10)
        urdf = mesh.with_suffix(".urdf")
        urdf.write_text(functions["_generate_urdf"](name, "", mesh.name))
        objects[name] = dict(
            name=name,
            mesh=str(mesh.relative_to(static)),
            urdf=str(urdf.relative_to(static)),
            mesh_sha256=digest(mesh),
            urdf_sha256=digest(urdf),
        )
    report = dict(
        schema="v2d.scored-generic-retarget-case.v1",
        episode=metadata["episode"],
        source_bundle_payload_sha256=digest(bundle / "episode.npz"),
        motion_payload_sha256=digest(proposal / "robot-motion-proposal.npz"),
        stage_rotation_from_inferred_support_normal=stage.as_matrix().tolist(),
        objects=objects,
        object_mass_inertia="existing nominal URDF generator defaults, not measurements",
        object_generator_source_sha256=digest(generator),
        object_generator_ast_sha256=signature,
        interpolated_hand_slots=inferred.sum(axis=0).tolist(),
        parked_hand_slots=parked.sum(axis=0).tolist(),
        original_source_confidence_preserved=True,
        contact_labels_fabricated=False,
        robot_motion_is_inferred=True,
        physical_task_accepted=False,
        producer_sha256=digest(Path(__file__)),
    )
    (static / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (proposal / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    case = out / "case"
    (case / "file-manifest.json").write_text(
        json.dumps(
            dict(
                schema="v2d.scored-case-files.v1",
                files_sha256={
                    p.relative_to(case).as_posix(): digest(p)
                    for p in sorted(case.rglob("*"))
                    if p.is_file()
                },
            ),
            indent=2,
        )
        + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["bundle", "retarget", "support", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.bundle, a.retarget, a.support, a.out), indent=2))
