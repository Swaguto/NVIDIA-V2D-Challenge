"""Prepare and execute an untrained, fixed-wrist Dex3 load diagnostic.

Called by reproduce_frozen_dex3_replay.py. The source trajectory and observations
remain immutable. A static grasp test does not certify the complete episode.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_functions(path: Path, names: set[str], namespace: dict) -> tuple[dict, str]:
    """Reuse exact existing definitions without executing an unrelated CLI/task."""
    definitions = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8-sig")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(definitions) != len(names) or {node.name for node in definitions} != names:
        raise ValueError("Expected reusable definitions are missing or ambiguous")
    module = ast.Module(body=definitions, type_ignores=[])

    def portable_ast(node):
        if isinstance(node, ast.AST):
            return {
                "type": type(node).__name__,
                "fields": {
                    name: portable_ast(value)
                    for name, value in ast.iter_fields(node)
                    if value is not None and value != []
                },
            }
        if isinstance(node, list):
            return [portable_ast(value) for value in node]
        if isinstance(node, bytes):
            return {"bytes_hex": node.hex()}
        return node

    signature = hashlib.sha256(
        json.dumps(portable_ast(module), sort_keys=True).encode()
    ).hexdigest()
    # Only whitelisted definitions from repository/verified snapshot sources.
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace, signature


def object_tracking_summary(rows):
    """Keep unobserved object error unavailable, including short diagnostic prefixes."""
    import numpy as np

    result = []
    for oi in sorted({row["object"] for row in rows}):
        observed = [
            row for row in rows if row["object"] == oi and row["source_object_observed"]
        ]
        errors = np.asarray([row["position_error_m"] for row in observed])
        angles = np.asarray([row["rotation_error_rad"] for row in observed])
        result.append(
            {
                "object": oi,
                "observed_measurements": len(observed),
                "position_median_m": float(np.median(errors)) if observed else None,
                "position_p90_m": float(np.quantile(errors, 0.9)) if observed else None,
                "position_max_m": float(errors.max()) if observed else None,
                "orientation_median_rad": float(np.median(angles))
                if observed
                else None,
                "orientation_p90_rad": float(np.quantile(angles, 0.9))
                if observed
                else None,
            }
        )
    return result


def apply_grasp_seeds(samples, proposal):
    """Apply bounded robot-only seeds after matching each original source pose."""
    import copy
    import numpy as np
    from scipy.spatial.transform import Rotation

    if (
        proposal["schema"] != "v2d.dex3-antipodal-seed.v1"
        or proposal["physical_task_accepted"]
    ):
        raise ValueError("Need explicit unaccepted geometric grasp seeds")
    result = copy.deepcopy(samples)
    translation_cap = float(proposal.get("wrist_translation_cap_m", 0.03))
    rotation_cap = float(proposal.get("wrist_rotation_cap_rad", 0.5))
    finger_cap = float(proposal.get("finger_delta_cap_rad", 0.6))
    if not (0 < translation_cap <= 0.2 and 0 < rotation_cap <= np.pi):
        raise ValueError(
            "Experimental grasp seed wrist bounds exceed the permitted search"
        )
    if not 0 < finger_cap <= 1.2:
        raise ValueError("Experimental grasp finger correction exceeds 1.2 rad")
    changed = []
    for record in proposal["results"]:
        candidate = record["candidate"]
        if candidate is None:
            continue
        matches = [s for s in result if s["prefix"] == record["sample"]]
        if len(matches) != 1:
            raise ValueError("Grasp seed does not identify one source sample")
        sample = matches[0]
        source = record["source_sample"]
        for key in ("source_frame", "object", "side"):
            if sample[key] != source[key]:
                raise ValueError("Grasp seed source identity differs")
        for key in (
            "hand_position_stage_m",
            "hand_quaternion_stage_xyzw",
            "object_position_stage_m",
            "object_quaternion_stage_xyzw",
        ):
            if not np.allclose(sample[key], source[key], atol=1e-6, rtol=0):
                raise ValueError("Grasp seed source pose differs")
        original = dict(
            zip(sample["joint_names"], sample["finger_target_rad"], strict=True)
        )
        old = dict(zip(source["joint_names"], source["finger_target_rad"], strict=True))
        new = dict(
            zip(candidate["joint_names"], candidate["finger_target_rad"], strict=True)
        )
        if set(new) != set(original) or any(
            abs(original[n] - old[n]) > 1e-6 for n in original
        ):
            raise ValueError("Grasp seed finger names/source values differ")
        values = np.array(list(new.values()))
        pos = np.array(candidate["hand_position_stage_m"])
        quat = np.array(candidate["hand_quaternion_stage_xyzw"])
        geometry = np.array(
            [
                candidate["pad_escape_direction_dot"],
                candidate["maximum_piece_penetration_lower_bound_m"],
            ]
        )
        if (
            not np.isfinite(np.r_[values, pos, quat, geometry]).all()
            or abs(np.linalg.norm(quat) - 1) > 1e-6
        ):
            raise ValueError("Nonfinite or invalid grasp seed")
        angle = (
            Rotation.from_quat(quat)
            * Rotation.from_quat(sample["hand_quaternion_stage_xyzw"]).inv()
        ).magnitude()
        if (
            np.max(np.abs(pos - sample["hand_position_stage_m"]))
            > translation_cap + 1e-6
            or angle > rotation_cap
            or max(abs(new[n] - original[n]) for n in original) > finger_cap + 1e-6
            or candidate["pad_escape_direction_dot"] > -0.5
            or candidate["pad_escape_direction_dot"] < -1.000001
            or candidate["maximum_piece_penetration_lower_bound_m"] < 0
            or candidate["maximum_piece_penetration_lower_bound_m"] > 0.003
        ):
            raise ValueError("Grasp seed exceeds the declared geometric search bounds")
        sample["hand_position_stage_m"] = pos.tolist()
        sample["hand_quaternion_stage_xyzw"] = quat.tolist()
        sample["finger_target_rad"] = [new[n] for n in sample["joint_names"]]
        if "required_contact_digits" in candidate:
            digits = candidate["required_contact_digits"]
            if len(digits) != 3 or set(digits) != {"thumb", "index", "middle"}:
                raise ValueError(
                    "A load-balanced grasp requires all three real finger contacts"
                )
            sample["required_contact_digits"] = list(digits)
        if "finger_stiffness_Nm_per_rad" in candidate:
            stiffness = float(candidate["finger_stiffness_Nm_per_rad"])
            if not math.isfinite(stiffness) or not 2 <= stiffness <= 8:
                raise ValueError("Candidate finger stiffness exceeds [2, 8] Nm/rad")
            sample["finger_stiffness_Nm_per_rad"] = stiffness
        if "finger_feedforward_Nm" in candidate:
            effort = np.asarray(candidate["finger_feedforward_Nm"], dtype=float)
            if (
                proposal.get("grip_effort", {}).get("schema")
                != "v2d.dex3-jacobian-grip-effort.v1"
                or effort.shape != (len(candidate["joint_names"]),)
                or not np.isfinite(effort).all()
                or np.max(np.abs(effort)) > 0.55
            ):
                raise ValueError("Invalid bounded finger squeeze feedforward")
            by_name = dict(zip(candidate["joint_names"], effort.tolist(), strict=True))
            sample["finger_feedforward_Nm"] = [
                by_name[n] for n in sample["joint_names"]
            ]
        if "finger_pregrasp_rad" in candidate:
            opened = np.asarray(candidate["finger_pregrasp_rad"], dtype=float)
            if (
                opened.shape != values.shape
                or not np.isfinite(opened).all()
                or np.max(np.abs(opened - values)) > 0.400001
            ):
                raise ValueError("Invalid bounded open pregrasp")
            by_name = dict(zip(candidate["joint_names"], opened.tolist(), strict=True))
            sample["finger_pregrasp_rad"] = [by_name[n] for n in sample["joint_names"]]
        sample["robot_grasp_seed_is_experimental"] = True
        changed.append(sample["prefix"])
    if not changed or len(set(changed)) != len(changed):
        raise ValueError("Need unique nonempty grasp seed changes")
    return result, changed


def acquisition_pairs(originals, seeded):
    """Matched source/seed fixtures, with the same initial fingers and support."""
    import copy
    import numpy as np

    result, changed = [], []
    for candidate in seeded:
        if not candidate.get("robot_grasp_seed_is_experimental"):
            continue
        source_index = next(
            i for i, s in enumerate(originals) if s["prefix"] == candidate["prefix"]
        )
        original = originals[source_index]
        force_comparison = "finger_pregrasp_rad" in candidate
        if force_comparison:
            baseline = copy.deepcopy(candidate)
            baseline["finger_feedforward_Nm"] = [0.0] * len(candidate["joint_names"])
            variants = (("position-only", baseline), ("squeeze", candidate))
        else:
            variants = (("baseline", original), ("antipodal", candidate))
        for variant, value in variants:
            sample = copy.deepcopy(value)
            shift = np.array([3.0 * (len(result) - source_index), 0.0, 0.0])
            for key in ("hand_position_stage_m", "object_position_stage_m"):
                sample[key] = (np.asarray(sample[key]) + shift).tolist()
            sample["source_grasp_prefix"] = original["prefix"]
            sample["prefix"] += "-" + variant
            sample["variant"] = variant
            sample["finger_initial_rad"] = (
                candidate["finger_pregrasp_rad"]
                if force_comparison
                else original["finger_target_rad"]
            ).copy()
            sample["same_geometry_motor_effort_comparison"] = force_comparison
            result.append(sample)
            if variant in {"antipodal", "squeeze"}:
                changed.append(sample["prefix"])
    if len(result) != 4:
        raise ValueError(
            "Controlled acquisition currently needs two complete paired source/seed grasps"
        )
    return result, changed


def acquisition_support(vertices, position, quaternion):
    """A finite diagnostic tray under the mesh's nearly horizontal thin axis."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    vertices = np.asarray(vertices, dtype=float)
    _, vectors = np.linalg.eigh(np.cov(vertices.T))
    normal = Rotation.from_quat(quaternion).apply(vectors[:, 0])
    if normal[2] < 0:
        normal = -normal
    if normal[2] < 0.7:
        raise ValueError(
            "Controlled acquisition tray requires a nearly horizontal thin object"
        )
    rotation = Rotation.align_vectors(normal[None], np.array([[0.0, 0.0, 1.0]]))[0]
    points = rotation.inv().apply(Rotation.from_quat(quaternion).apply(vertices))
    lower, upper = points.min(0), points.max(0)
    thickness = 0.02
    center = np.r_[(lower[:2] + upper[:2]) / 2, lower[2] - 0.0001 - thickness / 2]
    return dict(
        position_stage_m=(rotation.apply(center) + position).tolist(),
        quaternion_stage_xyzw=rotation.as_quat().tolist(),
        dimensions_m=np.r_[upper[:2] - lower[:2] + 0.04, thickness].tolist(),
        initial_minimum_clearance_m=0.0001,
        scope="temporary diagnostic tray; not reconstructed source support or an evaluation scene",
    )


def cooked_acquisition_support(world_vertices, original):
    """Seat a diagnostic tray against a broad lower face of actual cooked geometry."""
    import numpy as np
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    vertices = np.asarray(world_vertices)
    centered = vertices - vertices.mean(0)
    hull = ConvexHull(centered)
    triangles = centered[hull.simplices]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    eligible = np.flatnonzero(hull.equations[:, 2] < -0.7)
    if not len(eligible):
        raise ValueError("Cooked object has no nearly horizontal lower support face")
    dimensions = np.asarray(original["dimensions_m"])
    old_rotation = Rotation.from_quat(original["quaternion_stage_xyzw"])
    # A long handle can make the largest global-hull face a steep bridge between
    # disconnected bottom regions. Search the remaining faces within the same
    # fixture bounds rather than rejecting the object after that single choice.
    for index in eligible[np.argsort(-areas[eligible], kind="stable")]:
        normal = -hull.equations[index, :3]
        rotation = Rotation.align_vectors(normal[None], np.array([[0.0, 0.0, 1.0]]))[0]
        if (rotation * old_rotation.inv()).magnitude() > 0.3:
            continue
        local = rotation.inv().apply(vertices)
        lower, upper = local.min(0), local.max(0)
        if np.any(upper[:2] - lower[:2] >= dimensions[:2]):
            continue
        center = np.r_[
            (lower[:2] + upper[:2]) / 2, lower[2] - dimensions[2] / 2 - 0.0001
        ]
        position = rotation.apply(center)
        if np.linalg.norm(position - original["position_stage_m"]) > 0.02:
            continue
        return {
            **original,
            "position_stage_m": position.tolist(),
            "quaternion_stage_xyzw": rotation.as_quat().tolist(),
            "support_uses_actual_cooked_lower_face": True,
            "eligible_lower_faces_considered": len(eligible),
        }
    raise ValueError("No cooked support face fits the unchanged diagnostic bounds")


def prepare(
    case: Path,
    repo: Path,
    plane: Path,
    plane_sha: str,
    out: Path,
    *,
    task_motion: Path | None = None,
    controller_mode: str = "project_pd",
    free_space: bool = False,
    source_duration_s: float | None = None,
    physics_dt: float = 0.01,
    grasp_seed: Path | None = None,
    grasp_seed_sha256: str | None = None,
    controlled_acquisition: bool = False,
    controlled_grip_lift: bool = False,
    task_grasps: Path | None = None,
    task_grasps_sha256: str | None = None,
) -> dict:
    """Validate byte/coordinate contracts and build portable nominal load inputs."""
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation

    if controller_mode not in {
        "project_pd",
        "implicit_pd",
        "implicit_pd_ff",
        "implicit_pid_ff",
    }:
        raise ValueError("Unknown wrist controller")
    if not math.isfinite(physics_dt) or not 0 < physics_dt <= 0.01:
        raise ValueError("Physics dt must be finite and in (0, 0.01]")
    if source_duration_s is not None and (
        not math.isfinite(source_duration_s) or source_duration_s < 0.5
    ):
        raise ValueError("Diagnostic source duration must be at least 0.5 seconds")
    if task_motion is None and (
        controller_mode != "project_pd"
        or free_space
        or source_duration_s is not None
        or physics_dt != 0.01
    ):
        raise ValueError("Wrist diagnostic options require supported-task mode")
    if out.exists():
        raise FileExistsError("Use a fresh load-input directory")
    if controlled_acquisition and (grasp_seed is None or task_motion is not None):
        raise ValueError("Controlled acquisition requires static mode and grasp seeds")
    if controlled_grip_lift and not controlled_acquisition:
        raise ValueError("Controlled lift requires acquisition")
    if task_grasps is not None and (
        task_motion is None
        or grasp_seed is not None
        or digest(task_grasps) != task_grasps_sha256
        or controller_mode != "implicit_pd_ff"
    ):
        raise ValueError(
            "Task grasps need the supported task, exact SHA and implicit PD velocity tracking"
        )
    files = json.loads((case / "file-manifest.json").read_text())["files_sha256"]
    for name, expected in files.items():
        path = case / name
        if (
            not path.resolve().is_relative_to(case.resolve())
            or digest(path) != expected
        ):
            raise ValueError("Changed or escaping frozen case input: " + name)
    if digest(plane) != plane_sha:
        raise ValueError("Support-plane report hash mismatch")
    plane_report = json.loads(plane.read_text())
    if plane_report["bundle_manifest_sha256"] != digest(case / "bundle/manifest.json"):
        raise ValueError("Plane and source bundle differ")
    up = np.asarray(plane_report["plane_world"][:3], dtype=float)
    up /= np.linalg.norm(up)
    stage_rotation = Rotation.align_vectors(np.array([[0.0, 0.0, 1.0]]), up[None])[
        0
    ].as_matrix()
    np.testing.assert_allclose(stage_rotation @ up, [0, 0, 1], atol=1e-12)
    with np.load(case / "bundle/episode.npz", allow_pickle=False) as archive:
        source = {name: archive[name].copy() for name in archive.files}
    with np.load(
        case / "proposal/robot-motion-proposal.npz", allow_pickle=False
    ) as archive:
        motion = {name: archive[name].copy() for name in archive.files}
    np.testing.assert_array_equal(source["frame_indices"], motion["frame_indices"])
    np.testing.assert_array_equal(source["hand_valid"], motion["source_hand_valid"])
    np.testing.assert_array_equal(
        source["hand_directly_observed"], motion["source_hand_directly_observed"]
    )
    if (
        motion["robot_motion_target_validated"].any()
        or not motion["robot_pose_is_inferred"].all()
    ):
        raise ValueError("Load diagnostic requires explicit unaccepted inferred motion")
    bundle = json.loads((case / "bundle/manifest.json").read_text())
    contract = json.loads((case / "retarget/manifest.json").read_text())
    generator = repo / "robotic_grounding/scripts/generate_rigid_urdfs.py"
    functions, generator_ast = load_functions(
        generator, {"_generate_urdf"}, {"textwrap": textwrap}
    )
    out.mkdir(parents=True)
    objects_dir = out / "objects"
    objects_dir.mkdir()
    samples = []
    for si, side in enumerate(("left", "right")):
        for obj in bundle["objects"]:
            oi = obj["index"]
            phase_ids = np.unique(
                motion["robot_grasp_phase_index_inferred"][:, si][
                    motion["robot_contact_intent_inferred"][:, si, oi]
                ]
            )
            for phase in phase_ids:
                rows = np.flatnonzero(
                    motion["robot_contact_intent_inferred"][:, si, oi]
                    & (motion["robot_grasp_phase_index_inferred"][:, si] == phase)
                )
                if phase < 0 or not len(rows):
                    raise ValueError("Closed intent has no unambiguous phase")
                row = int(rows[0])
                q = motion["robot_qpos_inferred"][row, si]
                object_pose = source[f"object_{oi}_world"][row]
                offset = np.array([3.0 * len(samples), 0.0, 0.0])
                root_q = (
                    Rotation.from_matrix(stage_rotation) * Rotation.from_quat(q[3:7])
                ).as_quat()
                object_q = Rotation.from_matrix(
                    stage_rotation @ object_pose[:3, :3]
                ).as_quat()
                samples.append(
                    {
                        "prefix": f"{side}-object{oi}-grasp{int(phase)}",
                        "side": side,
                        "object": oi,
                        "object_scale_at_selected_frame": float(
                            source[f"object_{oi}_scale"][row]
                        ),
                        "source_row": row,
                        "source_frame": int(source["frame_indices"][row]),
                        "source_hand_valid": bool(source["hand_valid"][row, si]),
                        "hand_position_stage_m": (
                            stage_rotation @ q[:3] + offset
                        ).tolist(),
                        "hand_quaternion_stage_xyzw": root_q.tolist(),
                        "joint_names": contract["joints"][side]["finger_joint_names"],
                        "finger_target_rad": q[7:].tolist(),
                        "object_position_stage_m": (
                            stage_rotation @ object_pose[:3, 3] + offset
                        ).tolist(),
                        "object_quaternion_stage_xyzw": object_q.tolist(),
                    }
                )
    if not samples:
        raise ValueError("No source-derived closed grasps to test")
    seed_changes = []
    grip_spec = None
    if task_grasps is not None:
        import sys

        sys.path.insert(0, str(Path(__file__).parents[1] / "dex3_physics"))
        from task_grasp import build

        grip_spec = build(
            case,
            samples,
            json.loads(task_grasps.read_text()),
            stage_rotation,
            case
            / "source/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/g1",
        )
        seed_changes = [s["prefix"] for s in samples]
        write(out / "task-grips.json", grip_spec)
        (out / "task-grasp-seeds.json").write_bytes(task_grasps.read_bytes())
    if grasp_seed is not None:
        if task_motion is not None or digest(grasp_seed) != grasp_seed_sha256:
            raise ValueError("Grasp seed requires static mode and its exact SHA256")
        originals = samples
        samples, seed_changes = apply_grasp_seeds(
            samples, json.loads(grasp_seed.read_text())
        )
        if controlled_acquisition:
            samples, seed_changes = acquisition_pairs(originals, samples)
        (out / "grasp-seeds.json").write_bytes(grasp_seed.read_bytes())
    if task_motion is not None:
        producer = json.loads((case / "proposal/report.json").read_text())
        if digest(task_motion) != producer["motion_module_sha256"]:
            raise ValueError("Task evaluator differs from the frozen motion producer")
        if len(bundle["objects"]) != 2:
            raise ValueError(
                "Coupled task adapter currently requires two complete source objects"
            )
        samples = [
            next(sample for sample in samples if sample["side"] == side)
            for side in ("left", "right")
        ]
        if len({sample["object"] for sample in samples}) != 2:
            raise ValueError(
                "Coupled task requires distinct automatically bound primary objects"
            )
        for si, sample in enumerate(samples):
            oi = sample["object"]
            q = motion["robot_qpos_inferred"][0, si]
            object_pose = source[f"object_{oi}_world"][0]
            scales = source[f"object_{oi}_scale"]
            scale = float(np.median(scales))
            if np.max(np.abs(scales / scale - 1)) > 1e-6:
                raise ValueError(
                    "Source shape scale is not rigid within the declared numerical tolerance"
                )
            sample.update(
                prefix=f"{sample['side']}-object{oi}-task",
                source_row=0,
                source_frame=int(source["frame_indices"][0]),
                source_hand_valid=bool(source["hand_valid"][0, si]),
                object_scale_at_selected_frame=scale,
                hand_position_stage_m=(stage_rotation @ q[:3]).tolist(),
                hand_quaternion_stage_xyzw=(
                    Rotation.from_matrix(stage_rotation) * Rotation.from_quat(q[3:7])
                )
                .as_quat()
                .tolist(),
                finger_target_rad=q[7:].tolist(),
                object_position_stage_m=(stage_rotation @ object_pose[:3, 3]).tolist(),
                object_quaternion_stage_xyzw=Rotation.from_matrix(
                    stage_rotation @ object_pose[:3, :3]
                )
                .as_quat()
                .tolist(),
            )
            if grip_spec is not None:
                from _dex3_task_grip import TaskGripReference

                command = TaskGripReference(grip_spec).command(
                    sample["side"],
                    float(source["timestamps_s"][0]),
                    np.r_[
                        sample["object_position_stage_m"],
                        sample["object_quaternion_stage_xyzw"],
                    ],
                )
                sample.update(
                    hand_position_stage_m=command["pose"][:3].tolist(),
                    hand_quaternion_stage_xyzw=command["pose"][3:].tolist(),
                    finger_target_rad=command["fingers"].tolist(),
                    finger_stiffness_Nm_per_rad=grip_spec["phases"][sample["side"]][0][
                        "stiffness_Nm_per_rad"
                    ],
                )
                if "scene_projection" in grip_spec:
                    sample["object_position_stage_m"][2] += grip_spec[
                        "scene_projection"
                    ]["upward_corrections_m"][0][oi]
        (out / "motion-evaluator.py").write_bytes(task_motion.read_bytes())
        sheet = case / "support/inferred-support-sheet.npz"
        if "support/inferred-support-sheet.npz" not in files:
            raise ValueError("Task requires the hash-bound source support sheet")
        (out / "inferred-support-sheet.npz").write_bytes(sheet.read_bytes())
    object_files = {}
    for sample in samples:
        oi = sample["object"]
        obj = next(obj for obj in bundle["objects"] if obj["index"] == oi)
        vertices = (
            source[f"object_{oi}_vertices"] * sample["object_scale_at_selected_frame"]
        )
        faces = source[f"object_{oi}_faces"]
        if controlled_acquisition:
            sample["acquisition_support"] = acquisition_support(
                vertices,
                np.asarray(sample["object_position_stage_m"]),
                sample["object_quaternion_stage_xyzw"],
            )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh_file = objects_dir / f"{sample['prefix']}.obj"
        mesh.export(mesh_file, include_normals=False, include_color=False, digits=10)
        restored = trimesh.load(
            mesh_file, force="mesh", process=False, maintain_order=True
        )
        np.testing.assert_allclose(restored.vertices, vertices, atol=1e-8, rtol=0)
        np.testing.assert_array_equal(restored.faces, faces)
        urdf_file = mesh_file.with_suffix(".urdf")
        urdf_file.write_text(
            functions["_generate_urdf"](obj["name"], "", mesh_file.name),
            encoding="utf-8",
            newline="\n",
        )
        object_files[sample["prefix"]] = {
            "name": obj["name"],
            "mesh": mesh_file.relative_to(out).as_posix(),
            "urdf": urdf_file.relative_to(out).as_posix(),
            "mesh_sha256": digest(mesh_file),
            "urdf_sha256": digest(urdf_file),
        }
    renderer = case / "scripts/replay_generated_dex3_source_camera_sim6_v1.py"
    if not renderer.is_file():
        raise ValueError("Frozen simulator package lacks its verified hand factory")
    result = {
        "schema": "v2d.dex3-static-load-inputs.v1",
        "case_manifest_sha256": digest(case / "file-manifest.json"),
        "source_bundle_payload_sha256": digest(case / "bundle/episode.npz"),
        "motion_payload_sha256": digest(case / "proposal/robot-motion-proposal.npz"),
        "plane_report_sha256": plane_sha,
        "stage_rotation_from_inferred_support_normal": stage_rotation.tolist(),
        "gravity_stage_m_s2": [0.0, 0.0, -9.81],
        "gravity_is_independently_measured": False,
        "object_generator_source_sha256": digest(generator),
        "object_generator_AST_sha256": generator_ast,
        "verified_hand_factory_source_sha256": digest(renderer),
        "object_physics_source_sha256": digest(
            repo
            / "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/rigid_object.py"
        ),
        "motor_settings_source_sha256": digest(
            repo
            / "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/g1.py"
        ),
        "objects": object_files,
        "samples": samples,
        "nominal_object_mass_kg": 0.3,
        "nominal_object_inertia_diagonal_kg_m2": [0.001] * 3,
        "mass_and_inertia_are_existing_generator_defaults_not_measurements": True,
        "fixed_wrists_are_a_fixture_not_whole_body_feasibility": task_motion is None,
        "source_observations_or_robot_motion_changed": bool(seed_changes),
        "source_observations_changed": False,
        "experimental_grasp_seed_samples": seed_changes,
        "grasp_seed_sha256": grasp_seed_sha256,
        "per_frame_source_scale_preserved_for_each_static_fixture": task_motion is None,
        "whole_body_feasibility_claimed": False,
        "physical_task_accepted": False,
        "supported_task_mode": task_motion is not None,
        "controlled_acquisition": controlled_acquisition,
        "controlled_grip_lift": controlled_grip_lift,
        "temporary_support_release_s": 0.5 if controlled_acquisition else None,
        "task_grasps_sha256": task_grasps_sha256,
        "task_grip_spec_sha256": digest(out / "task-grips.json")
        if grip_spec is not None
        else None,
        "task_grip_controller_sha256": digest(
            Path(__file__).with_name("_dex3_task_grip.py")
        )
        if grip_spec is not None
        else None,
    }
    if task_motion is not None:
        result.update(
            motion_evaluator_sha256=digest(out / "motion-evaluator.py"),
            support_sheet_sha256=digest(out / "inferred-support-sheet.npz"),
            wrist_controller_source_sha256=digest(
                repo
                / "robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp/actions/action_track_residual.py"
            ),
            wrist_controller_cfg_source_sha256=digest(
                repo
                / "robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp/actions/actions_cfg.py"
            ),
            source_motion_speed=0.5,
            wrist_diagnostic={
                "controller": controller_mode,
                "free_space": free_space,
                "source_duration_s": source_duration_s,
                "physics_dt_s": physics_dt,
                "linear_integral_gain_N_per_m_s": 50.0
                if controller_mode == "implicit_pid_ff"
                else 0.0,
                "integral_force_component_cap_N": 10.0
                if controller_mode == "implicit_pid_ff"
                else 0.0,
                "integral_reset_on_collision_mask_transition": controller_mode
                == "implicit_pid_ff",
                "initial_free_space_position_offset_world_m": [0.01, 0.0, 0.0]
                if free_space
                else [0.0, 0.0, 0.0],
                "initial_free_space_rotation_offset_world_rotvec_rad": [
                    0.08726646259971647,
                    0.0,
                    0.0,
                ]
                if free_space
                else [0.0, 0.0, 0.0],
                "candidate_controller_sha256": digest(
                    Path(__file__).with_name("_dex3_wrist_control.py")
                ),
            },
        )
    write(out / "manifest.json", result)
    return result


def execute(
    case: Path, repo: Path, inputs: Path, out: Path, manifest_sha: str, device: str
) -> None:
    """Run gravity/contact with fixed wrists and bounded finger drives, never object playback."""
    import copy
    import platform as runtime_platform
    from types import SimpleNamespace

    import cv2
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import (
        Articulation,
        ArticulationCfg,
        RigidObject,
        RigidObjectCfg,
    )
    from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
    from isaaclab_physx.physics import PhysxCfg
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    from scipy.spatial.transform import Rotation

    manifest = inputs / "manifest.json"
    if digest(manifest) != manifest_sha:
        raise ValueError("Prepared physical input manifest changed")
    prepared = json.loads(manifest.read_text())
    task_mode = prepared.get("supported_task_mode", False)
    lift_mode = prepared.get("controlled_grip_lift", False)
    if digest(case / "file-manifest.json") != prepared["case_manifest_sha256"]:
        raise ValueError("Physical input and frozen case differ")
    for name, expected in json.loads((case / "file-manifest.json").read_text())[
        "files_sha256"
    ].items():
        path = case / name
        if (
            not path.resolve().is_relative_to(case.resolve())
            or digest(path) != expected
        ):
            raise ValueError("Changed frozen case: " + name)
    for obj in prepared["objects"].values():
        for kind in ("mesh", "urdf"):
            path = inputs / obj[kind]
            if (
                not path.resolve().is_relative_to(inputs.resolve())
                or digest(path) != obj[kind + "_sha256"]
            ):
                raise ValueError("Changed prepared physical asset")
    if out.exists():
        raise FileExistsError("Use a fresh physical result directory")
    out.mkdir(parents=True)
    assets = (
        case
        / "source/robotic_grounding/source/robotic_grounding/robotic_grounding/assets"
    )
    factory = case / "scripts/replay_generated_dex3_source_camera_sim6_v1.py"
    config_root = (
        repo / "robotic_grounding/source/robotic_grounding/robotic_grounding/assets"
    )
    object_source = config_root / "rigid_object.py"
    motor_source = config_root / "g1.py"
    for path, key in (
        (factory, "verified_hand_factory_source_sha256"),
        (object_source, "object_physics_source_sha256"),
        (motor_source, "motor_settings_source_sha256"),
    ):
        if digest(path) != prepared[key]:
            raise ValueError("Reusable physics definition changed: " + key)
    motor_names = {
        "ARMATURE_1515",
        "STIFFNESS_1515",
        "DAMPING_1515",
        "EFFORT_LIMIT_1515",
        "VELOCITY_LIMIT_1515",
    }
    motors = {}
    for node in ast.parse(motor_source.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in motor_names
        ):
            motors[node.targets[0].id] = float(ast.literal_eval(node.value))
    if set(motors) != motor_names:
        raise ValueError("Existing Dex3 motor constants missing")
    namespace = {
        "sim_utils": sim_utils,
        "RigidObjectCfg": RigidObjectCfg,
        "math": math,
        "ASSET_DIR": assets,
        "object_name": "arctic/box_rigid",
    }
    assignments = [
        node
        for node in ast.parse(object_source.read_text()).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "RIGID_OBJECT_CFG" for t in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError("Existing rigid object preset missing")
    migrated_collision_keywords = 0
    for node in ast.walk(assignments[0]):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "UrdfFileCfg"
        ):
            for keyword in node.keywords:
                if keyword.arg == "collider_type":
                    if ast.literal_eval(keyword.value) != "convex_decomposition":
                        raise ValueError("Unexpected legacy collision import setting")
                    keyword.arg = "collision_type"
                    keyword.value = ast.copy_location(
                        ast.Constant(value="Convex Decomposition"), keyword.value
                    )
                    migrated_collision_keywords += 1
    if migrated_collision_keywords != 1:
        raise ValueError("Expected one explicit Sim6 collision import migration")
    exec(  # noqa: S102
        compile(
            ast.Module(body=assignments, type_ignores=[]), str(object_source), "exec"
        ),
        namespace,
    )
    preset = namespace["RIGID_OBJECT_CFG"]
    diagnostic = prepared.get("wrist_diagnostic", {})
    dt = diagnostic.get("physics_dt_s", 0.01)
    sim_cfg = sim_utils.SimulationCfg(
        dt=dt,
        device=device,
        use_fabric=device != "cpu",
        gravity=tuple(prepared["gravity_stage_m_s2"]),
        render_interval=5,
        physics=PhysxCfg(bounce_threshold_velocity=0.2),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    light = sim_utils.DomeLightCfg(intensity=1800.0)
    light.func("/World/Light", light)
    hands, objects, sensors, cameras, provenance = [], [], [], [], []
    acquisition_mode = prepared.get("controlled_acquisition", False)
    acquisition_platforms = []

    if task_mode:
        sheet_file = inputs / "inferred-support-sheet.npz"
        if digest(sheet_file) != prepared["support_sheet_sha256"]:
            raise ValueError("Task support sheet changed")
        with np.load(sheet_file, allow_pickle=False) as sheet:
            vertices = (
                sheet["vertices_world_m"]
                @ np.asarray(prepared["stage_rotation_from_inferred_support_normal"]).T
            )
            faces = sheet["faces"]
        support = UsdGeom.Mesh.Define(sim.stage, "/World/TaskSupport")
        support.CreatePointsAttr(
            Vt.Vec3fArray([Gf.Vec3f(*point) for point in vertices])
        )
        support.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        support.CreateFaceVertexIndicesAttr(
            Vt.IntArray(faces.astype(np.int32).reshape(-1).tolist())
        )
        support.CreateSubdivisionSchemeAttr("none")
        support.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.35, 0.35)])
        UsdPhysics.CollisionAPI.Apply(support.GetPrim()).CreateCollisionEnabledAttr(
            True
        )
        UsdPhysics.MeshCollisionAPI.Apply(support.GetPrim()).CreateApproximationAttr(
            "none"
        )
        contact = PhysxSchema.PhysxCollisionAPI.Apply(support.GetPrim())
        contact.CreateContactOffsetAttr(0.0)
        contact.CreateRestOffsetAttr(0.0)

    def bind_material(prim_path, coefficient):
        material_path = prim_path + "/DiagnosticMaterial"
        cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=coefficient, dynamic_friction=coefficient, restitution=0.0
        )
        if prepared.get("task_grip_spec_sha256"):
            cfg.friction_combine_mode = "average"
            cfg.restitution_combine_mode = "average"
        cfg.func(material_path, cfg)
        material = UsdShade.Material(sim.stage.GetPrimAtPath(material_path))
        material_prim = material.GetPrim()
        physical = UsdPhysics.MaterialAPI(material_prim)
        actual_material = dict(
            static_friction=physical.GetStaticFrictionAttr().Get(),
            dynamic_friction=physical.GetDynamicFrictionAttr().Get(),
            friction_combine_mode=PhysxSchema.PhysxMaterialAPI(material_prim)
            .GetFrictionCombineModeAttr()
            .Get(),
        )
        if prepared.get("task_grip_spec_sha256") and (
            actual_material["static_friction"] != coefficient
            or actual_material["dynamic_friction"] != coefficient
            or actual_material["friction_combine_mode"] != "average"
        ):
            raise ValueError("Actual physical material differs from task contact model")
        colliders = 0
        collider_readback = []
        for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(prim_path)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders += int(
                    bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
                )
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, materialPurpose="physics"
                )
                contact_offset = prim.GetAttribute("physxCollision:contactOffset").Get()
                rest_offset = prim.GetAttribute("physxCollision:restOffset").Get()
                if contact_offset != 0.0 or rest_offset != 0.0:
                    raise ValueError(
                        "Requested collision margins were not applied: "
                        + str(prim.GetPath())
                    )
                collider_readback.append(
                    {
                        "prim": str(prim.GetPath()),
                        "contact_offset_m": contact_offset,
                        "rest_offset_m": rest_offset,
                        "approximation": UsdPhysics.MeshCollisionAPI(prim)
                        .GetApproximationAttr()
                        .Get()
                        if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                        else None,
                    }
                )
        if not colliders:
            raise ValueError("No enabled physical colliders on " + prim_path)
        return {
            "enabled_colliders": colliders,
            "collider_readback": collider_readback,
            "static_friction": coefficient,
            "dynamic_friction": coefficient,
            "restitution": 0.0,
            "friction_combine_mode": cfg.friction_combine_mode,
            "restitution_combine_mode": cfg.restitution_combine_mode,
            "physical_material_readback": actual_material,
        }

    for index, sample in enumerate(prepared["samples"]):
        fixture = out / sample["prefix"]
        fixture.mkdir()
        prim_root = f"/World/Fixture{index}"
        urdf = ET.parse(assets / f"urdfs/g1/dex3_{sample['side']}.urdf")
        limits = {
            joint.attrib["name"]: joint.find("limit").attrib
            for joint in urdf.getroot().findall("joint")
            if joint.attrib["type"] != "fixed"
        }
        target = dict(
            zip(sample["joint_names"], sample["finger_target_rad"], strict=True)
        )
        for name, value in target.items():
            if (
                not float(limits[name]["lower"])
                <= value
                <= float(limits[name]["upper"])
            ):
                raise ValueError("Frozen finger target exceeds URDF limit")
        if acquisition_mode:
            for name, value in zip(
                sample["joint_names"], sample["finger_initial_rad"], strict=True
            ):
                if (
                    not float(limits[name]["lower"])
                    <= value
                    <= float(limits[name]["upper"])
                ):
                    raise ValueError("Pregrasp initial finger exceeds URDF limit")

        def physical_hand(
            *, cfg, prim_root=prim_root, sample=sample, target=target, limits=limits
        ):
            cfg.prim_path = prim_root + "/Hand"
            cfg.spawn.fix_base = not (task_mode or lift_mode)
            cfg.spawn.activate_contact_sensors = True
            cfg.spawn.rigid_props.disable_gravity = False
            cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.0, rest_offset=0.0
            )
            cfg.init_state.pos = tuple(sample["hand_position_stage_m"])
            cfg.init_state.rot = tuple(sample["hand_quaternion_stage_xyzw"])
            cfg.init_state.joint_pos = target
            cfg.actuators["fingers"] = ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=sample.get(
                    "finger_stiffness_Nm_per_rad", motors["STIFFNESS_1515"]
                ),
                damping=motors["DAMPING_1515"],
                armature=motors["ARMATURE_1515"],
                effort_limit_sim={
                    name: min(
                        motors["EFFORT_LIMIT_1515"], float(limits[name]["effort"])
                    )
                    for name in target
                },
                velocity_limit_sim={
                    name: min(
                        motors["VELOCITY_LIMIT_1515"], float(limits[name]["velocity"])
                    )
                    for name in target
                },
            )
            return Articulation(cfg)

        hand_namespace, factory_ast = load_functions(
            factory,
            {"resolved_urdf", "create_hand"},
            {
                "ET": ET,
                "Path": Path,
                "Any": Any,
                "args": SimpleNamespace(out=fixture),
                "ASSETS": assets,
                "digest": digest,
                "Articulation": physical_hand,
                "ArticulationCfg": ArticulationCfg,
                "sim_utils": sim_utils,
                "ImplicitActuatorCfg": ImplicitActuatorCfg,
            },
        )
        hand, hand_provenance = hand_namespace["create_hand"](sample["side"])
        sim_utils.make_uninstanceable(hand.cfg.prim_path)
        sim_utils.modify_collision_properties(
            hand.cfg.prim_path, hand.cfg.spawn.collision_props
        )
        hand_material = bind_material(
            hand.cfg.prim_path, 2.0
        )  # Lower bound of project's startup material range [2.0, 2.01].
        cfg = copy.deepcopy(preset)
        cfg.prim_path = prim_root + "/Object"
        cfg.spawn.asset_path = str(
            (inputs / prepared["objects"][sample["prefix"]]["urdf"]).resolve()
        )
        cfg.spawn.make_instanceable = False
        cfg.spawn.force_usd_conversion = True
        cfg.spawn.usd_dir = str(fixture / "object-usd")
        cfg.init_state.pos = tuple(sample["object_position_stage_m"])
        cfg.init_state.rot = tuple(sample["object_quaternion_stage_xyzw"])
        obj = RigidObject(cfg)
        sim_utils.make_uninstanceable(cfg.prim_path)
        # Native Sim6 conversion can ignore the requested importer approximation.
        # Author the intended project preset on the composed collision meshes.
        converted_meshes = 0
        for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(cfg.prim_path)):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetTypeName() == "Mesh":
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                    "convexDecomposition"
                )
                PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                converted_meshes += 1
        if not converted_meshes:
            raise ValueError("Object has no editable mesh collider")
        sim_utils.modify_collision_properties(cfg.prim_path, cfg.spawn.collision_props)
        object_material = bind_material(
            cfg.prim_path, float(sim_cfg.physics_material.dynamic_friction)
        )
        if (
            prepared.get("task_grip_spec_sha256")
            and object_material["dynamic_friction"] != 0.5
        ):
            raise ValueError(
                "Task force model expects the original 0.5 object friction"
            )
        if any(
            collider["approximation"] != "convexDecomposition"
            for collider in object_material["collider_readback"]
        ):
            raise ValueError(
                "Actual object approximation differs from the intended decomposition"
            )
        sensor = ContactSensor(
            ContactSensorCfg(
                prim_path=cfg.prim_path + "/.*", update_period=0.0, history_length=1
            )
        )
        position = np.asarray(sample["object_position_stage_m"])
        camera = (
            None
            if task_mode
            else Camera(
                CameraCfg(
                    prim_path=prim_root + "/Camera",
                    update_period=0.0,
                    width=320,
                    height=320,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.01, 100.0),
                    ),
                )
            )
        )
        hands.append(hand)
        objects.append(obj)
        sensors.append(sensor)
        if camera is not None:
            cameras.append((camera, position))
        provenance.append(
            {
                "hand": hand_provenance,
                "factory_AST_sha256": factory_ast,
                "hand_material": hand_material,
                "object_material": object_material,
                "motor_effort_caps_Nm": hand.cfg.actuators["fingers"].effort_limit_sim,
                "finger_stiffness_Nm_per_rad": hand.cfg.actuators["fingers"].stiffness,
                "finger_damping_Nm_s_per_rad": hand.cfg.actuators["fingers"].damping,
                "sample": sample,
            }
        )

    if acquisition_mode:
        for index, sample in enumerate(prepared["samples"]):
            support = sample["acquisition_support"]
            path = f"/World/AcquisitionSupport{index}"
            root = UsdGeom.Xform.Define(sim.stage, path)
            root.AddTranslateOp().Set(Gf.Vec3d(*support["position_stage_m"]))
            q = support["quaternion_stage_xyzw"]
            root.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
            box = UsdGeom.Cube.Define(sim.stage, path + "/Box")
            box.CreateSizeAttr(1.0)
            box.AddScaleOp().Set(Gf.Vec3f(*support["dimensions_m"]))
            collision = UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            collision.CreateCollisionEnabledAttr(True)
            physx = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
            physx.CreateContactOffsetAttr(0.0)
            physx.CreateRestOffsetAttr(0.0)
            material = bind_material(path, 1.0)
            acquisition_platforms.append(
                dict(
                    prim=path + "/Box",
                    material=material,
                    initial_collision_enabled=bool(
                        collision.GetCollisionEnabledAttr().Get()
                    ),
                    pose_and_dimensions=support,
                )
            )
    if task_mode:
        with np.load(case / "bundle/episode.npz", allow_pickle=False) as source:
            image_width, image_height = map(int, source["image_size"])
            intrinsics = source["intrinsics"].copy()
        margin_x = int(np.ceil(abs(float(intrinsics[0, 2]) - image_width / 2))) + 4
        margin_y = int(np.ceil(abs(float(intrinsics[1, 2]) - image_height / 2))) + 4
        canvas_width, canvas_height = (
            image_width + 2 * margin_x,
            image_height + 2 * margin_y,
        )
        task_camera = Camera(
            CameraCfg(
                prim_path="/World/ReviewCamera",
                update_period=0.0,
                width=canvas_width,
                height=canvas_height,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    horizontal_aperture=24.0,
                    clipping_range=(0.01, 100.0),
                ),
            )
        )
        cameras.append(
            (
                task_camera,
                {
                    "width": image_width,
                    "height": image_height,
                    "crop_x": canvas_width / 2 - float(intrinsics[0, 2]),
                    "crop_y": canvas_height / 2 - float(intrinsics[1, 2]),
                    "intrinsics": intrinsics.tolist(),
                },
            )
        )
    # Two independent physics controls: unsupported free fall and a sphere on a local fixed platform.
    control_objects, control_sensors = [], []
    for index, x in enumerate((14.0, 16.0, 18.0) if acquisition_mode else (14.0, 16.0)):
        cfg = RigidObjectCfg(
            prim_path=f"/World/Control{index}/Ball",
            spawn=sim_utils.SphereCfg(
                radius=0.025,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(x, 0.0, 2.0 if index == 0 else 0.22), rot=(0.0, 0.0, 0.0, 1.0)
            ),
        )
        control_objects.append(RigidObject(cfg))
        control_sensors.append(
            ContactSensor(
                ContactSensorCfg(
                    prim_path=cfg.prim_path, update_period=0.0, history_length=1
                )
            )
        )
    platform_cfg = sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(restitution=0.0),
    )
    platform_cfg.func(
        "/World/Control1/Platform", platform_cfg, translation=(16.0, 0.0, 0.0)
    )
    release_control_platforms = []
    if acquisition_mode:
        box = UsdGeom.Cube.Define(sim.stage, "/World/Control2/Platform")
        box.CreateSizeAttr(1.0)
        box.AddTranslateOp().Set(Gf.Vec3d(18.0, 0.0, 0.0))
        box.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.05))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim()).CreateCollisionEnabledAttr(True)
        px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
        px.CreateContactOffsetAttr(0.0)
        px.CreateRestOffsetAttr(0.0)
        release_control_platforms.append(dict(prim="/World/Control2/Platform"))
    # Sim6's importer activates contact reporting only on the root body.
    # Pair-wise finger measurements need reporting on every real hand body.
    for asset in hands + objects:
        for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(asset.cfg.prim_path)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    if diagnostic.get("free_space", False):
        for hand in hands:
            sim_utils.modify_collision_properties(
                hand.cfg.prim_path,
                sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            )
    sim.reset()
    for camera, position in [] if task_mode else cameras:
        camera.set_world_poses_from_view(
            torch.tensor(
                (position + [0.0, -0.55, 0.35])[None],
                dtype=torch.float32,
                device=sim.device,
            ),
            torch.tensor(position[None], dtype=torch.float32, device=sim.device),
        )
    targets, initial_targets, effort_targets = [], [], []
    for hand, sample in zip(hands, prepared["samples"], strict=True):
        values = dict(
            zip(sample["joint_names"], sample["finger_target_rad"], strict=True)
        )
        if set(hand.joint_names) != set(values) or hand.is_fixed_base == (
            task_mode or lift_mode
        ):
            raise ValueError(
                "Physical articulation topology differs from declared fixture"
            )
        if lift_mode:
            initial = hand.data.default_root_state.clone()
            initial[:, :3] = torch.tensor(
                sample["hand_position_stage_m"], device=sim.device
            )
            initial[:, 3:7] = torch.tensor(
                sample["hand_quaternion_stage_xyzw"], device=sim.device
            )
            initial[:, 7:] = 0
            hand.write_root_state_to_sim(initial)
            hand.reset()
        q = torch.tensor(
            [[values[name] for name in hand.joint_names]],
            dtype=torch.float32,
            device=sim.device,
        )
        initial_values = dict(
            zip(
                sample["joint_names"],
                sample["finger_initial_rad"]
                if acquisition_mode
                else sample["finger_target_rad"],
                strict=True,
            )
        )
        initial_q = torch.tensor(
            [[initial_values[name] for name in hand.joint_names]],
            dtype=torch.float32,
            device=sim.device,
        )
        hand.write_joint_state_to_sim(initial_q, torch.zeros_like(q))
        hand.set_joint_position_target(initial_q)
        hand.write_data_to_sim()
        # Reuse the frozen renderer's explicit FK refresh before the first render/contact step.
        _ = hand.data.body_link_pose_w
        targets.append(q)
        initial_targets.append(initial_q)
        efforts = dict(
            zip(
                sample["joint_names"],
                sample.get("finger_feedforward_Nm", [0.0] * len(sample["joint_names"])),
                strict=True,
            )
        )
        effort_targets.append(
            torch.tensor(
                [[efforts[n] for n in hand.joint_names]],
                dtype=torch.float32,
                device=sim.device,
            )
        )
    for obj in objects + control_objects:
        # One initialization only. No object state writes or external forces during load.
        obj.write_root_state_to_sim(obj.data.default_root_state.clone())
        obj.reset()
    if task_mode:
        run_supported_task(
            case,
            repo,
            inputs,
            out,
            prepared,
            sim,
            hands,
            objects,
            sensors,
            cameras,
            control_objects,
            control_sensors,
            provenance,
        )
        return
    from isaaclab_physx.physics import PhysxManager

    static_contact_views, static_contact_bindings = [], []
    for hand, obj in zip(hands, objects, strict=True):
        paths = [
            path
            for path in hand.root_physx_view.link_paths[0]
            if any(
                prim.HasAPI(UsdPhysics.CollisionAPI)
                for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(path))
            )
        ]
        object_path = obj.root_physx_view.prim_paths[0]
        view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
            paths,
            filter_patterns=[[object_path]] * len(paths),
            max_contact_data_count=64 * len(paths),
        )
        if view.sensor_count != len(paths) or view.filter_count != 1:
            raise ValueError("Static finger/object contact bindings differ")
        static_contact_views.append(view)
        static_contact_bindings.append(
            dict(hand_body_paths=paths, object_body_path=object_path)
        )
    from _dex3_collision_evidence import (
        capture_cooked_colliders,
        contact_separation_summary,
    )

    cooked_collider_summary = capture_cooked_colliders(
        sim.stage,
        [(f"hand-{i}", hand) for i, hand in enumerate(hands)]
        + [(f"object-{i}", obj) for i, obj in enumerate(objects)],
        write,
        out / "cooked-colliders.json",
    )
    cooked_support_alignment = None
    if acquisition_mode:
        helper = repo / "scripts/dex3_physics/cooked_grasp_audit.py"
        spatial, helper_ast = load_functions(
            helper, {"world_convexes"}, {"np": np, "Rotation": Rotation}
        )
        cooked = json.loads((out / "cooked-colliders.json").read_text())["colliders"]
        for index, support in enumerate(acquisition_platforms):
            pose = objects[index].data.root_link_pose_w[0].detach().cpu().numpy()
            vertices = np.concatenate(
                [
                    v
                    for c in cooked
                    if c["asset"] == f"object-{index}"
                    for v in spatial["world_convexes"](c, pose)
                ]
            )
            updated = cooked_acquisition_support(
                vertices, support["pose_and_dimensions"]
            )
            root = sim.stage.GetPrimAtPath(support["prim"]).GetParent()
            root.GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(*updated["position_stage_m"])
            )
            q = updated["quaternion_stage_xyzw"]
            root.GetAttribute("xformOp:orient").Set(
                Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3]))
            )
            support["actual_cooked_surface_pose_and_dimensions"] = updated
        cooked_support_alignment = dict(
            helper_source_sha256=digest(helper),
            helper_AST_sha256=helper_ast,
            cooked_colliders_sha256=digest(out / "cooked-colliders.json"),
        )
    frames = []
    records = []
    control_records = []
    sim.forward()
    sim.render()
    initial_readback = []
    for hand, obj, target, sample in zip(
        hands, objects, initial_targets, prepared["samples"], strict=True
    ):
        record = {"sample": sample["prefix"]}
        for asset, prefix in ((hand, "hand"), (obj, "object")):
            pose = asset.data.root_link_pose_w[0].detach().cpu().numpy()
            expected_position = np.asarray(sample[prefix + "_position_stage_m"])
            expected_quat = np.asarray(sample[prefix + "_quaternion_stage_xyzw"])
            error = float(np.max(np.abs(pose[:3] - expected_position)))
            quat_dot = float(abs(np.dot(pose[3:7], expected_quat)))
            if error > 1e-5 or abs(quat_dot - 1.0) > 1e-5:
                raise ValueError(
                    "Initial imported pose differs from the frozen target: "
                    + sample["prefix"]
                    + "/"
                    + prefix
                )
            record[prefix + "_translation_error_m"] = error
            record[prefix + "_quaternion_absolute_dot"] = quat_dot
        joint_error = float(torch.max(torch.abs(hand.data.joint_pos - target)).item())
        if joint_error > 1e-5:
            raise ValueError("Initial fingers differ from the frozen target")
        record["finger_maximum_error_rad"] = joint_error
        expected_stiffness = sample.get(
            "finger_stiffness_Nm_per_rad", motors["STIFFNESS_1515"]
        )
        actual_stiffness = hand.data.joint_stiffness[0].detach().cpu().numpy()
        if not np.allclose(actual_stiffness, expected_stiffness, atol=1e-6, rtol=0):
            raise ValueError(
                "Imported joint stiffness differs from the declared controller"
            )
        record["finger_stiffness_Nm_per_rad_sim_order"] = actual_stiffness.tolist()
        actual_caps = hand.data.joint_effort_limits[0].detach().cpu().numpy()
        expected_caps = np.array(
            [
                hand.cfg.actuators["fingers"].effort_limit_sim[name]
                for name in hand.joint_names
            ]
        )
        if not np.allclose(actual_caps, expected_caps, atol=1e-6, rtol=0):
            raise ValueError(
                "Imported joint effort limits differ from the source motor caps"
            )
        record["finger_effort_caps_Nm_sim_order"] = actual_caps.tolist()
        body_poses = hand.data.body_link_pose_w[0].detach().cpu().numpy()
        record["hand_body_link_pose_stage_xyzw"] = {
            name: pose.tolist()
            for name, pose in zip(hand.body_names, body_poses, strict=True)
        }
        initial_readback.append(record)
    last_step = 150 if acquisition_mode else 100
    load_start_step = 50 if acquisition_mode else 0
    release_readback = []
    applied_efforts = [torch.zeros_like(t) for t in effort_targets]
    from _dex3_grip_control import (
        lift_target,
        lift_summary,
        payload_gravity_wrench,
        payload_contact_hold,
    )
    from _dex3_wrist_control import implicit_root_wrench

    wrist_wrenches = [np.zeros(6) for _ in hands]
    payload_contact_timers = [0.0 for _ in hands]
    for step in range(last_step + 1):
        if acquisition_mode and step == 50:
            for support in acquisition_platforms + release_control_platforms:
                prim = sim.stage.GetPrimAtPath(support["prim"])
                api = UsdPhysics.CollisionAPI(prim)
                before = bool(api.GetCollisionEnabledAttr().Get())
                api.GetCollisionEnabledAttr().Set(False)
                UsdGeom.Imageable(prim).MakeInvisible()
                after = bool(api.GetCollisionEnabledAttr().Get())
                if not before or after:
                    raise RuntimeError("Diagnostic tray collision did not switch off")
                release_readback.append(
                    dict(
                        prim=support["prim"],
                        step=step,
                        before_enabled=before,
                        after_enabled=after,
                        visibility=UsdGeom.Imageable(prim).GetVisibilityAttr().Get(),
                    )
                )
            sim.render()
        for index, (obj, sensor, hand, sample) in enumerate(
            zip(objects, sensors, hands, prepared["samples"], strict=True)
        ):
            state = obj.data.root_state_w[0].detach().cpu().numpy()
            contact = sensor.data.net_forces_w[0].detach().cpu().numpy().sum(axis=0)
            joint = hand.data.joint_pos[0].detach().cpu().numpy()
            if not np.isfinite(np.concatenate((state, contact, joint))).all():
                raise ValueError("Nonfinite physical state")
            records.append(
                {
                    "step": step,
                    "time_s": step * dt,
                    "sample_index": index,
                    "phase": (
                        "support_release_boundary"
                        if acquisition_mode and step == 50
                        else "supported_closure"
                        if acquisition_mode and step < 50
                        else "unsupported_load"
                    ),
                    "object_root_state_stage_xyzw": state.tolist(),
                    "contact_force_stage_N": contact.tolist(),
                    "finger_position_rad_sim_order": joint.tolist(),
                    "finger_names_sim_order": hand.joint_names,
                    "estimated_implicit_actuator_torque_Nm": hand.data.applied_torque[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "commanded_finger_feedforward_Nm_sim_order": applied_efforts[index][
                        0
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "object_normal_force_by_hand_link_stage_N": static_contact_views[
                        index
                    ]
                    .get_contact_force_matrix(dt)
                    .numpy()[:, 0, :]
                    .tolist(),
                    "hand_object_solver_contacts": contact_separation_summary(
                        static_contact_views[index], dt
                    ),
                    "solver_contact_snapshot_valid_for_current_pose": step > 0,
                    "actual_wrist_pose_stage_xyzw": hand.data.root_link_pose_w[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "target_wrist_pose_stage_xyzw": lift_target(
                        np.r_[
                            sample["hand_position_stage_m"],
                            sample["hand_quaternion_stage_xyzw"],
                        ],
                        step * dt,
                    ).tolist()
                    if lift_mode
                    else None,
                    "applied_wrist_wrench_world": wrist_wrenches[index].tolist()
                    if lift_mode
                    else None,
                    "payload_contact_hold_remaining_s": payload_contact_timers[index]
                    if lift_mode
                    else None,
                }
            )
        for index, (obj, sensor) in enumerate(
            zip(control_objects, control_sensors, strict=True)
        ):
            if step > 100 and index != 2:
                continue
            control_records.append(
                {
                    "step": step,
                    "time_s": step * dt,
                    "control": index,
                    "root_state_stage_xyzw": obj.data.root_state_w[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "contact_force_stage_N": sensor.data.net_forces_w[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .sum(axis=0)
                    .tolist(),
                }
            )
        if step % 5 == 0:
            images = []
            for (camera, _), sample in zip(cameras, prepared["samples"], strict=True):
                camera.update(dt, force_recompute=True)
                rgb = (
                    camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                )
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    bgr,
                    f"{sample['side']} f{sample['source_frame']} {sample.get('variant', '')}",
                    (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    bgr,
                    f"gravity ON; tray {'ON' if acquisition_mode and step < 50 else 'OFF'}  t={step * dt:.2f}s",
                    (6, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                images.append(bgr)
            frames.append(np.concatenate(images, axis=1))
        if step == last_step:
            break
        blend = min(1.0, (step + 1) / 25.0) if acquisition_mode else 1.0
        blend = blend * blend * (3 - 2 * blend)
        for index, (hand, target, initial, effort) in enumerate(
            zip(hands, targets, initial_targets, effort_targets, strict=True)
        ):
            hand.set_joint_position_target(initial + blend * (target - initial))
            squeeze_blend = (
                min(1.0, max(0.0, (step + 1 - 25) / 25)) if acquisition_mode else 1.0
            )
            squeeze_blend = squeeze_blend * squeeze_blend * (3 - 2 * squeeze_blend)
            applied_efforts[index] = squeeze_blend * effort
            hand.set_joint_effort_target(applied_efforts[index])
            if lift_mode:
                sample = prepared["samples"][index]
                initial_pose = np.r_[
                    sample["hand_position_stage_m"],
                    sample["hand_quaternion_stage_xyzw"],
                ]

                def cpu(value):
                    return (
                        value.detach().cpu().numpy()
                        if hasattr(value, "detach")
                        else value.numpy()
                    )

                gravity = cpu(hand.root_physx_view.get_gravity_compensation_forces())[
                    0, :6
                ].copy()
                # Expected payload weight is applied only through the wrist actuator.
                # Objects receive no external force, attachment, or pose playback.
                payload_contact_timers[index] = payload_contact_hold(
                    static_contact_views[index]
                    .get_contact_force_matrix(dt)
                    .numpy()[:, 0, :],
                    static_contact_bindings[index]["hand_body_paths"],
                    payload_contact_timers[index],
                    dt,
                    sample.get("required_contact_digits", ("thumb", "index")),
                )
                if step >= 50 and payload_contact_timers[index] > 0:
                    gravity += payload_gravity_wrench(
                        cpu(hand.data.root_com_pos_w)[0],
                        cpu(objects[index].data.root_com_pos_w)[0],
                        prepared["nominal_object_mass_kg"],
                    )
                wrench, _ = implicit_root_wrench(
                    link_pose_xyzw=cpu(hand.data.root_link_pose_w)[0],
                    target_link_pose_xyzw=lift_target(initial_pose, step * dt),
                    target_link_pose_next_xyzw=lift_target(
                        initial_pose, (step + 1) * dt
                    ),
                    root_com_offset_b=cpu(hand.root_physx_view.get_coms())[0, 0, :3],
                    root_com_velocity_w=cpu(hand.data.root_com_vel_w)[0],
                    root_mass_matrix_w=cpu(
                        hand.root_physx_view.get_generalized_mass_matrices()
                    )[0, :6, :6],
                    gravity_compensation_w=gravity,
                    dt=dt,
                )
                wrist_wrenches[index] = wrench
                hand.set_external_force_and_torque(
                    forces=torch.tensor(
                        wrench[None, None, :3], dtype=torch.float32, device=sim.device
                    ),
                    torques=torch.tensor(
                        wrench[None, None, 3:], dtype=torch.float32, device=sim.device
                    ),
                    body_ids=[0],
                    is_global=True,
                )
            hand.write_data_to_sim()
        sim.step(render=step % 5 == 4)
        for asset in hands + objects + control_objects:
            asset.update(dt)
        for sensor in sensors + control_sensors:
            sensor.update(dt, force_recompute=True)
    summaries = []
    for index, sample in enumerate(prepared["samples"]):
        rows = [row for row in records if row["sample_index"] == index]
        positions = np.array([row["object_root_state_stage_xyzw"][:3] for row in rows])
        quats = np.array([row["object_root_state_stage_xyzw"][3:7] for row in rows])
        distance = np.linalg.norm(
            positions[load_start_step:] - positions[load_start_step], axis=1
        )
        angles = (
            Rotation.from_quat(quats[load_start_step:])
            * Rotation.from_quat(quats[load_start_step]).inv()
        ).magnitude()
        forces = np.array([row["contact_force_stage_N"] for row in rows])
        summaries.append(
            {
                "sample": sample["prefix"],
                "source_frame": sample["source_frame"],
                "variant": sample.get("variant", "source-static"),
                "load_interval_s": [load_start_step * dt, last_step * dt],
                "displacement_during_supported_closure_m": float(
                    np.linalg.norm(positions[load_start_step] - positions[0])
                ),
                "final_displacement_m": float(distance[-1]),
                "maximum_displacement_m": float(distance.max()),
                "maximum_orientation_change_deg": float(np.rad2deg(angles.max())),
                "maximum_contact_force_N": float(
                    np.linalg.norm(
                        forces[load_start_step + int(acquisition_mode) :], axis=1
                    ).max()
                ),
                "mean_vertical_contact_force_last_0_2s_N": float(
                    forces[-20:, 2].mean()
                ),
                "held_within_10mm_for_1s_diagnostic_only": bool(distance.max() <= 0.01),
            }
        )
        if lift_mode:
            summaries[-1].update(lift_summary(rows))
    freefall = [row for row in control_records if row["control"] == 0]
    platform = [row for row in control_records if row["control"] == 1]
    drop = (
        freefall[0]["root_state_stage_xyzw"][2]
        - freefall[-1]["root_state_stage_xyzw"][2]
    )
    supported_z = platform[-1]["root_state_stage_xyzw"][2]
    supported_force = float(
        np.mean([row["contact_force_stage_N"][2] for row in platform[-20:]])
    )
    controls = {
        "free_fall_drop_m": drop,
        "free_fall_pass": bool(4.8 <= drop <= 5.1),
        "supported_sphere_final_z_m": supported_z,
        "supported_sphere_mean_vertical_contact_force_N": supported_force,
        "supported_sphere_pass": bool(
            abs(supported_z - 0.05) < 0.005 and abs(supported_force - 0.3 * 9.81) < 0.3
        ),
    }
    if acquisition_mode:
        released = [row for row in control_records if row["control"] == 2]
        release_z = released[50]["root_state_stage_xyzw"][2]
        release_drop = release_z - released[-1]["root_state_stage_xyzw"][2]
        residual_force = float(
            np.max(
                np.linalg.norm(
                    np.array([r["contact_force_stage_N"] for r in released[-20:]]),
                    axis=1,
                )
            )
        )
        controls.update(
            temporary_support_release_control_z_at_release_m=release_z,
            temporary_support_release_control_drop_m=release_drop,
            temporary_support_release_control_final_contact_force_max_N=residual_force,
            temporary_support_release_control_pass=bool(
                abs(release_z - 0.05) < 0.005
                and 4.8 <= release_drop <= 5.1
                and residual_force < 0.001
            ),
        )
    controls_pass = (
        controls["free_fall_pass"]
        and controls["supported_sphere_pass"]
        and controls.get("temporary_support_release_control_pass", True)
    )
    write(out / "states.json", {"records": records, "controls": control_records})
    video_path = out / "static-load.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError("Physical diagnostic video writer failed")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    cv2.imwrite(str(out / "initial.png"), frames[0])
    cv2.imwrite(str(out / "final.png"), frames[-1])
    write(
        out / "report.json",
        {
            "schema": "v2d.dex3-static-load-result.v1",
            "cooked_collider_summary": cooked_collider_summary,
            "executor_source_sha256": digest(Path(__file__)),
            "inputs_manifest_sha256": manifest_sha,
            "runtime_python": runtime_platform.python_version(),
            "platform": runtime_platform.platform(),
            "device": device,
            "simulation_dt_s": dt,
            "duration_s": last_step * dt,
            "controlled_acquisition": acquisition_mode,
            "controlled_grip_lift": lift_mode,
            "wrist_load_feedforward": {
                "nominal_payload_mass_kg": prepared["nominal_object_mass_kg"],
                "force_applied_to": "wrist only",
                "includes_payload_lever_arm_moment": True,
                "activation": "measured thumb and index contact >=0.5 N each; 30 ms dropout hold",
                "force_cap_N": 60,
                "torque_cap_Nm": 60,
            }
            if lift_mode
            else None,
            "load_start_s": load_start_step * dt,
            "temporary_support_is_a_diagnostic_fixture": acquisition_mode,
            "acquisition_platforms": acquisition_platforms,
            "cooked_support_alignment": cooked_support_alignment,
            "support_release_readback": release_readback,
            "gravity_m_s2": list(sim_cfg.gravity),
            "motor_constants": motors,
            "object_collision_import_migration": "Legacy collider_type=convex_decomposition becomes Sim6 collision_type=Convex Decomposition; other preset fields are reused",
            "provenance": provenance,
            "initial_pose_readback": initial_readback,
            "static_contact_bindings": static_contact_bindings,
            "initial_solver_contact_buffer_is_cached_not_current_pose": True,
            "finger_feedforward_scope": "bounded real finger motor torques; original combined simulator motor effort limits unchanged; no external object wrench",
            "summaries": summaries,
            "controls": controls,
            "gravity_and_contact_controls_pass": controls_pass,
            "states_sha256": digest(out / "states.json"),
            "video_sha256": digest(video_path),
            "not_a_complete_episode_or_official_policy_rollout": True,
            "source_observations_or_robot_motion_changed": bool(
                prepared.get("experimental_grasp_seed_samples")
            ),
            "source_observations_changed": False,
            "experimental_grasp_seed_samples": prepared.get(
                "experimental_grasp_seed_samples", []
            ),
            "grasp_seed_sha256": prepared.get("grasp_seed_sha256"),
            "physical_task_accepted": False,
        },
    )
    sim.stop()
    if not controls_pass:
        raise RuntimeError("Physics controls failed; do not interpret grasp outcomes")


def run_supported_task(
    case,
    repo,
    inputs,
    out,
    prepared,
    sim,
    hands,
    objects,
    sensors,
    cameras,
    controls,
    control_sensors,
    provenance,
):
    """Execute the full locked motion through the existing wrist controller with zero policy residual."""
    import importlib.util
    import sys

    import imageio.v2 as imageio
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    from _dex3_collision_evidence import (
        capture_cooked_colliders,
        contact_separation_summary,
    )
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation
    from _dex3_wrist_control import finger_position_target, implicit_root_wrench
    from _dex3_grip_control import payload_contact_hold
    from _dex3_task_grip import (
        TaskGripReference,
        contact_wrench,
        filtered_reaction_compensation,
    )

    diagnostic = prepared.get("wrist_diagnostic", {})
    controller_mode = diagnostic.get("controller", "project_pd")
    task_grip = None
    if prepared.get("task_grip_spec_sha256"):
        spec_file = inputs / "task-grips.json"
        if (
            digest(spec_file) != prepared["task_grip_spec_sha256"]
            or digest(Path(__file__).with_name("_dex3_task_grip.py"))
            != prepared["task_grip_controller_sha256"]
            or digest(inputs / "task-grasp-seeds.json")
            != prepared["task_grasps_sha256"]
        ):
            raise ValueError("Task grasp commands or controller changed")
        task_grip = TaskGripReference(json.loads(spec_file.read_text()))
    candidate_file = Path(__file__).with_name("_dex3_wrist_control.py")
    if (
        diagnostic
        and digest(candidate_file) != diagnostic["candidate_controller_sha256"]
    ):
        raise ValueError("Experimental wrist controller changed after preparation")

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def controller_tensor(value):
        # Lab3 data proxies implement eager tensor operations, but TorchScript
        # requires their explicit zero-copy Torch view at the call boundary.
        import warp as wp

        if hasattr(value, "torch"):
            return value.torch
        if isinstance(value, wp.array):
            return wp.to_torch(value)
        return value

    evaluator_file = inputs / "motion-evaluator.py"
    if digest(evaluator_file) != prepared["motion_evaluator_sha256"]:
        raise ValueError("Task evaluator changed")
    evaluator = load_module("_v2d_locked_task_motion", evaluator_file)
    motion = evaluator.LockedReplayMotion(
        case / "bundle",
        None,
        None,
        None,
        None,
        case / "proposal",
        source=case / "source/robotic_grounding",
    )
    controller_file = (
        repo
        / "robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp/actions/action_track_residual.py"
    )
    cfg_file = controller_file.with_name("actions_cfg.py")
    if (
        digest(controller_file) != prepared["wrist_controller_source_sha256"]
        or digest(cfg_file) != prepared["wrist_controller_cfg_source_sha256"]
    ):
        raise ValueError("Existing tracking controller changed")
    controller = load_module("_v2d_existing_task_controller", controller_file)
    cfg_class = next(
        node
        for node in ast.parse(cfg_file.read_text()).body
        if isinstance(node, ast.ClassDef)
        and node.name == "JointResidualWithTrackingActionCfg"
    )
    keys = {
        "tracking_controller_linear_stiffness",
        "tracking_controller_linear_damping",
        "tracking_controller_angular_stiffness",
        "tracking_controller_angular_damping",
        "max_force",
        "max_torque",
    }
    gains = {
        node.target.id: float(ast.literal_eval(node.value))
        for node in cfg_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in keys
    }
    if gains.keys() != keys:
        raise ValueError("Expected project wrist-controller settings missing")
    with np.load(
        case / "proposal/robot-motion-proposal.npz", allow_pickle=False
    ) as archive:
        proposal = {name: archive[name].copy() for name in archive.files}
    display = np.zeros_like(proposal["source_hand_valid"], dtype=bool)
    for name, value in proposal.items():
        if "display_only" in name and value.shape == display.shape:
            display |= value.astype(bool)
    source = motion.source
    stage_rotation = Rotation.from_matrix(
        np.asarray(prepared["stage_rotation_from_inferred_support_normal"])
    )
    camera, camera_cfg = cameras[0]
    camera.set_intrinsic_matrices(
        torch.tensor(
            np.asarray(camera_cfg["intrinsics"])[None],
            dtype=torch.float32,
            device=sim.device,
        )
    )
    sim.set_setting("/physics/updateToUsd", True)
    joint_order, com_positions = [], []
    for hand, sample in zip(hands, prepared["samples"], strict=True):
        if (
            hand.is_fixed_base
            or hand.body_names[0] != sample["side"] + "_hand_palm_link"
        ):
            raise ValueError("Task wrist must be the floating original palm")
        joint_order.append(
            [sample["joint_names"].index(name) for name in hand.joint_names]
        )
        com_positions.append(
            controller_tensor(hand.root_physx_view.get_coms())[:, 0, :3].to(sim.device)
        )
    from isaaclab_physx.physics import PhysxManager
    from pxr import Usd, UsdPhysics

    contact_views, contact_bindings = [], []
    filters = [obj.root_physx_view.prim_paths[0] for obj in objects]
    for hand in hands:
        paths = [
            path
            for path in hand.root_physx_view.link_paths[0]
            if any(
                prim.HasAPI(UsdPhysics.CollisionAPI)
                for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(path))
            )
        ]
        view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
            paths,
            filter_patterns=[filters] * len(paths),
            max_contact_data_count=64 * len(paths),
        )
        if view.sensor_count != len(paths) or view.filter_count != len(objects):
            raise ValueError(
                "Actual physical contact-body bindings differ from the two-object task"
            )
        contact_views.append(view)
        contact_bindings.append(
            {"hand_body_paths": paths, "object_body_paths": filters}
        )
    for hand, sample, order in zip(
        hands, prepared["samples"], joint_order, strict=True
    ):
        # Kit startup may advance a floating hand before the controller exists.
        # Reset each hand once to the declared initial root pose and zero velocity.
        initial = controller_tensor(hand.data.default_root_state).clone()
        initial[:, :3] = torch.tensor(
            sample["hand_position_stage_m"], device=sim.device
        )
        initial[:, 3:7] = torch.tensor(
            sample["hand_quaternion_stage_xyzw"], device=sim.device
        )
        if diagnostic.get("free_space", False):
            initial[:, :3] += torch.tensor(
                diagnostic["initial_free_space_position_offset_world_m"],
                device=sim.device,
            )
            initial[:, 3:7] = torch.tensor(
                (
                    Rotation.from_rotvec(
                        diagnostic[
                            "initial_free_space_rotation_offset_world_rotvec_rad"
                        ]
                    )
                    * Rotation.from_quat(sample["hand_quaternion_stage_xyzw"])
                ).as_quat(),
                dtype=torch.float32,
                device=sim.device,
            )
        initial[:, 7:] = 0
        hand.write_root_state_to_sim(initial)
        hand.reset()
        q = torch.tensor(
            np.asarray(sample["finger_target_rad"])[order][None],
            dtype=torch.float32,
            device=sim.device,
        )
        hand.write_joint_state_to_sim(q, torch.zeros_like(q))
        hand.set_joint_position_target(q)
        hand.write_data_to_sim()
        _ = hand.data.body_link_pose_w
    for obj in objects + controls:
        obj.write_root_state_to_sim(obj.data.default_root_state.clone())
        obj.reset()
    cooked_summary = capture_cooked_colliders(
        sim.stage,
        [
            (f"hand-{side}", hand)
            for side, hand in zip(("left", "right"), hands, strict=True)
        ]
        + [
            (f"object-{sample['object']}", obj)
            for sample, obj in zip(prepared["samples"], objects, strict=True)
        ],
        write,
        out / "cooked-colliders.json",
    )
    print("Cooked collision representation captured", flush=True)
    initial_scene_check = None
    if task_grip is not None and "scene_projection" in task_grip.spec:
        sys.path.insert(0, str(Path(__file__).parents[1] / "dex3_physics"))
        from scene_projection import verify_initial_scene

        initial_scene_check = verify_initial_scene(
            out / "cooked-colliders.json",
            task_grip.spec["scene_projection"],
            [
                controller_tensor(obj.data.root_link_pose_w)[0].cpu().numpy()
                for obj in objects
            ],
        )
        write(out / "initial-scene-clearance.json", initial_scene_check)
    source_times = source["timestamps_s"]
    original_source_frames = len(source_times)
    if diagnostic.get("source_duration_s") is not None:
        source_times = source_times[
            source_times <= source_times[0] + diagnostic["source_duration_s"] + 1e-10
        ]
    speed = prepared["source_motion_speed"]
    dt = sim.get_physics_dt()
    postroll_s = (
        1.0 if task_grip is not None and not diagnostic.get("free_space") else 0.0
    )
    steps = int(np.ceil((source_times[-1] - source_times[0] + postroll_s) / speed / dt))
    rows, frames, hand_rows, controls_rows = [], [], [], []
    postroll_frames = []
    collision_state = [None, None]
    collision_transitions = []
    integral_force = [np.zeros(3), np.zeros(3)]
    reaction_compensation = [np.zeros(6), np.zeros(6)]
    contact_hold = [0.0, 0.0]
    task_motor_readbacks = []
    for hand, sample in zip(hands, prepared["samples"], strict=True):
        stiffness = controller_tensor(hand.data.joint_stiffness)[0].cpu().numpy()
        caps = controller_tensor(hand.data.joint_effort_limits)[0].cpu().numpy()
        if task_grip is not None and (
            not np.allclose(stiffness, sample["finger_stiffness_Nm_per_rad"], atol=1e-6)
            or np.max(caps) > 0.760001
        ):
            raise ValueError("Task motor gains or limits differ from declared commands")
        task_motor_readbacks.append(
            dict(
                side=sample["side"],
                stiffness_Nm_per_rad=stiffness.tolist(),
                effort_caps_Nm=caps.tolist(),
            )
        )
    source_row = 0
    writer = imageio.get_writer(
        out / "supported-task.mp4",
        fps=1 / float(np.median(np.diff(source_times))),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    initial_links = []
    for hand in hands:
        initial_links.append(
            {
                name: pose.tolist()
                for name, pose in zip(
                    hand.body_names,
                    hand.data.body_link_pose_w[0].cpu().numpy(),
                    strict=True,
                )
            }
        )
    try:
        for step in range(steps + 1):
            grip_time = float(source_times[0] + step * dt * speed)
            timestamp = min(grip_time, float(source_times[-1]))
            index = int(
                np.clip(
                    np.searchsorted(source_times, timestamp + 1e-10, side="right") - 1,
                    0,
                    len(source_times) - 1,
                )
            )
            desired, _kinds, _details = motion.evaluate(timestamp)
            desired_next = (
                motion.evaluate(min(timestamp + dt * speed, float(source_times[-1])))[0]
                if controller_mode in {"implicit_pd_ff", "implicit_pid_ff"}
                else None
            )
            for si, (hand, order) in enumerate(zip(hands, joint_order, strict=True)):
                enabled = not diagnostic.get("free_space", False) and (
                    task_grip is not None or not bool(display[index, si])
                )
                if collision_state[si] != enabled:
                    integral_force[si] = np.zeros(3)
                    sim_utils.modify_collision_properties(
                        hand.cfg.prim_path,
                        sim_utils.CollisionPropertiesCfg(
                            collision_enabled=enabled,
                            contact_offset=0.0,
                            rest_offset=0.0,
                        ),
                    )
                    readback = {
                        str(prim.GetPath()): bool(
                            UsdPhysics.CollisionAPI(prim)
                            .GetCollisionEnabledAttr()
                            .Get()
                        )
                        for prim in Usd.PrimRange(
                            sim.stage.GetPrimAtPath(hand.cfg.prim_path)
                        )
                        if prim.HasAPI(UsdPhysics.CollisionAPI)
                    }
                    if not readback or any(
                        value != enabled for value in readback.values()
                    ):
                        raise ValueError(
                            "Actual hand collision flags differ from requested mask"
                        )
                    collision_transitions.append(
                        {
                            "step": step,
                            "side": ("left", "right")[si],
                            "usd_collision_enabled": readback,
                        }
                    )
                    collision_state[si] = enabled
                target_pose = np.r_[
                    stage_rotation.apply(desired[si, :3]),
                    (stage_rotation * Rotation.from_quat(desired[si, 3:7])).as_quat(),
                ]
                grip_command, grip_next = None, None
                if task_grip is not None:
                    oi = prepared["samples"][si]["object"]
                    measured = (
                        controller_tensor(objects[si].data.root_link_pose_w)[0]
                        .cpu()
                        .numpy()
                    )

                    def grip_at(t):
                        r, p, _scale = motion.object_pose(
                            oi, min(t, float(source_times[-1]))
                        )
                        obj_pose = np.r_[
                            stage_rotation.apply(p),
                            (stage_rotation * Rotation.from_matrix(r)).as_quat(),
                        ]
                        return task_grip.command(
                            ("left", "right")[si],
                            t,
                            obj_pose,
                            None if diagnostic.get("free_space") else measured,
                        )

                    grip_command = grip_at(grip_time)
                    grip_next = grip_at(grip_time + dt * speed)
                    target_pose = grip_command["pose"]
                target = torch.tensor(
                    target_pose[None], dtype=torch.float32, device=sim.device
                )
                finger_target = torch.tensor(
                    (
                        grip_command["fingers"]
                        if grip_command is not None
                        else desired[si, 7:]
                    )[order][None],
                    dtype=torch.float32,
                    device=sim.device,
                )
                control_details = None
                if controller_mode == "project_pd":
                    force, torque, fingers = (
                        controller._compute_wrist_wrench_and_finger_target(
                            wrist_position=controller_tensor(hand.data.root_link_pos_w),
                            wrist_wxyz=controller_tensor(hand.data.root_link_quat_w),
                            wrist_link_vel_w=controller_tensor(
                                hand.data.root_com_vel_w
                            ).clone(),
                            wrist_link_quat_w=controller_tensor(
                                hand.data.root_link_quat_w
                            ),
                            wrist_com_pos_b=com_positions[si],
                            wrist_pose_target=target,
                            finger_joint_target=finger_target,
                            gravity_comp_e=controller_tensor(
                                hand.root_physx_view.get_gravity_compensation_forces()
                            ),
                            joint_damping=controller_tensor(hand.data.joint_damping),
                            joint_vel=controller_tensor(hand.data.joint_vel),
                            joint_effort_limits=controller_tensor(
                                hand.data.joint_effort_limits
                            ),
                            joint_stiffness=controller_tensor(
                                hand.data.joint_stiffness
                            ),
                            joint_pos=controller_tensor(hand.data.joint_pos),
                            linear_stiffness=gains[
                                "tracking_controller_linear_stiffness"
                            ],
                            linear_damping=gains["tracking_controller_linear_damping"],
                            angular_stiffness=gains[
                                "tracking_controller_angular_stiffness"
                            ],
                            angular_damping=gains[
                                "tracking_controller_angular_damping"
                            ],
                            max_force=gains["max_force"],
                            max_torque=gains["max_torque"],
                        )
                    )
                    is_global = False
                else:

                    def cpu(value):
                        return controller_tensor(value).detach().cpu().numpy()

                    wrench, control_details = implicit_root_wrench(
                        link_pose_xyzw=cpu(hand.data.root_link_pose_w)[0],
                        target_link_pose_xyzw=target_pose,
                        target_link_pose_next_xyzw=grip_next["pose"]
                        if grip_next is not None
                        else np.r_[
                            stage_rotation.apply(desired_next[si, :3]),
                            (
                                stage_rotation
                                * Rotation.from_quat(desired_next[si, 3:7])
                            ).as_quat(),
                        ]
                        if desired_next is not None
                        else None,
                        integral_force_world=integral_force[si]
                        if controller_mode == "implicit_pid_ff"
                        else None,
                        linear_integral_gain=diagnostic.get(
                            "linear_integral_gain_N_per_m_s", 0.0
                        )
                        if controller_mode == "implicit_pid_ff"
                        else 50.0,
                        max_integral_force=diagnostic.get(
                            "integral_force_component_cap_N", 0.0
                        )
                        if controller_mode == "implicit_pid_ff"
                        else 10.0,
                        root_com_offset_b=cpu(com_positions[si])[0],
                        root_com_velocity_w=cpu(hand.data.root_com_vel_w)[0],
                        root_mass_matrix_w=cpu(
                            hand.root_physx_view.get_generalized_mass_matrices()
                        )[0, :6, :6],
                        gravity_compensation_w=cpu(
                            hand.root_physx_view.get_gravity_compensation_forces()
                        )[0, :6],
                        dt=dt,
                        linear_stiffness=gains["tracking_controller_linear_stiffness"],
                        linear_damping=gains["tracking_controller_linear_damping"],
                        angular_stiffness=gains[
                            "tracking_controller_angular_stiffness"
                        ],
                        angular_damping=gains["tracking_controller_angular_damping"],
                        max_force=gains["max_force"],
                        max_torque=gains["max_torque"],
                    )
                    integral_force[si] = np.array(
                        control_details["integral_force_world_next_N"]
                    )
                    if grip_command is not None:
                        measured_wrench = np.zeros(6)
                        if step > 0 and enabled:
                            normal_matrix = (
                                contact_views[si]
                                .get_contact_force_matrix(dt)
                                .numpy()
                                .copy()
                            )
                            normal_data = [
                                x.numpy().copy()
                                for x in contact_views[si].get_contact_data(dt)
                            ]
                            friction_data = [
                                x.numpy().copy()
                                for x in contact_views[si].get_friction_data(dt)
                            ]
                            current_pose = cpu(hand.data.root_link_pose_w)[0]
                            com = current_pose[:3] + Rotation.from_quat(
                                current_pose[3:]
                            ).apply(cpu(com_positions[si])[0])
                            measured_wrench = contact_wrench(
                                normal_data, friction_data, com
                            )
                            normal_sum = contact_wrench(
                                normal_data,
                                [
                                    np.zeros((0, 3)),
                                    np.zeros((0, 3)),
                                    np.zeros_like(normal_data[-2]),
                                    np.zeros_like(normal_data[-1]),
                                ],
                                com,
                            )[:3]
                            if not np.allclose(
                                normal_sum,
                                normal_matrix.sum(axis=(0, 1)),
                                atol=1e-3,
                                rtol=1e-3,
                            ):
                                write(
                                    out / "contact-buffer-mismatch.json",
                                    dict(
                                        step=step,
                                        side=("left", "right")[si],
                                        aggregate_normal_force=normal_matrix.tolist(),
                                        detailed_normal_force=normal_sum.tolist(),
                                        normal_buffers=[
                                            x.tolist() for x in normal_data
                                        ],
                                        friction_buffers=[
                                            x.tolist() for x in friction_data
                                        ],
                                    ),
                                )
                                raise ValueError(
                                    f"Detailed normal-force buffer differs from aggregate hand forces: step {step}, side {si}, detailed {normal_sum.tolist()}, aggregate {normal_matrix.sum(axis=(0, 1)).tolist()}"
                                )
                            contact_hold[si] = payload_contact_hold(
                                normal_matrix.sum(axis=1),
                                contact_bindings[si]["hand_body_paths"],
                                contact_hold[si],
                                dt,
                                grip_command["required_digits"],
                            )
                        reaction_compensation[si] = filtered_reaction_compensation(
                            measured_wrench,
                            reaction_compensation[si],
                            dt,
                            enabled
                            and contact_hold[si] > 0
                            and grip_command["squeeze_scale"] > 0,
                        )
                        wrench = np.clip(
                            wrench + reaction_compensation[si],
                            -np.array(
                                [gains["max_force"]] * 3 + [gains["max_torque"]] * 3
                            ),
                            np.array(
                                [gains["max_force"]] * 3 + [gains["max_torque"]] * 3
                            ),
                        )
                        control_details.update(
                            measured_object_reaction_wrench_world=measured_wrench.tolist(),
                            applied_reaction_compensation_world=reaction_compensation[
                                si
                            ].tolist(),
                            contact_hold_remaining_s=contact_hold[si],
                        )
                    force = torch.tensor(
                        wrench[None, :3], dtype=torch.float32, device=sim.device
                    )
                    torque = torch.tensor(
                        wrench[None, 3:], dtype=torch.float32, device=sim.device
                    )
                    fingers = torch.tensor(
                        finger_position_target(
                            cpu(finger_target),
                            cpu(hand.data.joint_pos),
                            cpu(hand.data.joint_vel),
                            cpu(hand.data.joint_stiffness),
                            cpu(hand.data.joint_damping),
                            cpu(hand.data.joint_effort_limits),
                        ),
                        dtype=torch.float32,
                        device=sim.device,
                    )
                    is_global = True
                if not all(
                    torch.isfinite(value).all() for value in (force, torque, fingers)
                ):
                    raise ValueError("Nonfinite task control")
                hand.set_external_force_and_torque(
                    forces=force[:, None, :],
                    torques=torque[:, None, :],
                    body_ids=[0],
                    is_global=is_global,
                )
                hand.set_joint_position_target(fingers)
                if grip_command is not None:
                    hand.set_joint_effort_target(
                        torch.tensor(
                            grip_command["effort"][order][None],
                            dtype=torch.float32,
                            device=sim.device,
                        )
                    )
                hand.write_data_to_sim()
                hand_rows.append(
                    {
                        "step": step,
                        "source_time_s": timestamp,
                        "inferred_grip_time_s": grip_time
                        if task_grip is not None
                        else None,
                        "side": ("left", "right")[si],
                        "display_only": bool(display[index, si]),
                        "collisions_enabled": enabled,
                        "source_hand_valid": bool(source["hand_valid"][index, si]),
                        "source_directly_observed": bool(
                            source["hand_directly_observed"][index, si]
                        ),
                        "target_root_pose_stage_xyzw": target_pose.tolist(),
                        "actual_root_pose_stage_xyzw": hand.data.root_link_pose_w[0]
                        .cpu()
                        .numpy()
                        .tolist(),
                        "actual_fingers_rad": hand.data.joint_pos[0]
                        .cpu()
                        .numpy()
                        .tolist(),
                        "target_fingers_rad": finger_target[0].cpu().numpy().tolist(),
                        "actual_body_link_poses_stage_xyzw": {
                            name: pose.tolist()
                            for name, pose in zip(
                                hand.body_names,
                                hand.data.body_link_pose_w[0].cpu().numpy(),
                                strict=True,
                            )
                        }
                        if step % 10 == 0 or step == steps
                        else None,
                        "finger_joint_names": hand.joint_names,
                        "applied_wrist_force_world_N"
                        if is_global
                        else "applied_wrist_force_local_N": force[0]
                        .cpu()
                        .numpy()
                        .tolist(),
                        "applied_wrist_torque_world_Nm"
                        if is_global
                        else "applied_wrist_torque_local_Nm": torque[0]
                        .cpu()
                        .numpy()
                        .tolist(),
                        "root_com_velocity_world": controller_tensor(
                            hand.data.root_com_vel_w
                        )[0]
                        .cpu()
                        .numpy()
                        .tolist(),
                        "controller_details": control_details,
                        "inferred_grasp_phase": grip_command["phase"]
                        if grip_command is not None
                        else None,
                        "squeeze_scale": grip_command["squeeze_scale"]
                        if grip_command is not None
                        else None,
                        "measured_object_feedback_weight": grip_command[
                            "object_feedback_weight"
                        ]
                        if grip_command is not None
                        else None,
                        "finger_feedforward_Nm": grip_command["effort"][order].tolist()
                        if grip_command is not None
                        else None,
                        "object_normal_contact_force_N": contact_views[si]
                        .get_contact_force_matrix(dt)
                        .numpy()
                        .sum(axis=0)
                        .tolist(),
                        "object_contact_force_magnitudes_N": np.linalg.norm(
                            contact_views[si].get_contact_force_matrix(dt).numpy(),
                            axis=-1,
                        )
                        .sum(axis=0)
                        .tolist(),
                        "solver_contacts": contact_separation_summary(
                            contact_views[si], dt
                        ),
                    }
                )
            for obj, sample, sensor in zip(
                objects, prepared["samples"], sensors, strict=True
            ):
                oi = sample["object"]
                rotation, position, _scale = motion.object_pose(oi, timestamp)
                reference = np.r_[
                    stage_rotation.apply(position),
                    (stage_rotation * Rotation.from_matrix(rotation)).as_quat(),
                ]
                actual = obj.data.root_link_pose_w[0].cpu().numpy()
                if not np.isfinite(actual).all():
                    raise ValueError("Nonfinite dynamic object state")
                error = float(np.linalg.norm(actual[:3] - reference[:3]))
                angle = float(
                    (
                        Rotation.from_quat(actual[3:])
                        * Rotation.from_quat(reference[3:]).inv()
                    ).magnitude()
                )
                rows.append(
                    {
                        "step": step,
                        "source_time_s": timestamp,
                        "object": oi,
                        "source_object_observed": bool(
                            source[f"object_{oi}_observed"][index]
                        )
                        if f"object_{oi}_observed" in source
                        else bool(proposal["source_object_observed"][index, oi]),
                        "actual_root_pose_stage_xyzw": actual.tolist(),
                        "reference_root_pose_stage_xyzw": reference.tolist(),
                        "position_error_m": error,
                        "rotation_error_rad": angle,
                        "normal_contact_force_stage_N": sensor.data.net_forces_w[0]
                        .cpu()
                        .numpy()
                        .sum(axis=0)
                        .tolist(),
                    }
                )
            if step <= round(1 / dt):
                for ci, (obj, sensor) in enumerate(
                    zip(controls, control_sensors, strict=True)
                ):
                    controls_rows.append(
                        {
                            "step": step,
                            "control": ci,
                            "root_pose": obj.data.root_link_pose_w[0]
                            .cpu()
                            .numpy()
                            .tolist(),
                            "normal_contact_force": sensor.data.net_forces_w[0]
                            .cpu()
                            .numpy()
                            .sum(axis=0)
                            .tolist(),
                        }
                    )
            render_source = (
                source_row < len(source_times)
                and timestamp + 1e-10 >= source_times[source_row]
            )
            render_postroll = (
                task_grip is not None
                and grip_time > source_times[-1] + 1e-9
                and step % round(0.05 / (dt * speed)) == 0
            )
            if render_source or render_postroll:
                camera_row = min(source_row, len(source_times) - 1)
                matrix = source["camera_c2w"][camera_row]
                camera.set_world_poses(
                    torch.tensor(
                        stage_rotation.apply(matrix[:3, 3])[None],
                        dtype=torch.float32,
                        device=sim.device,
                    ),
                    torch.tensor(
                        (
                            stage_rotation * Rotation.from_matrix(matrix[:3, :3])
                        ).as_quat()[None],
                        dtype=torch.float32,
                        device=sim.device,
                    ),
                    convention="ros",
                )
                sim.render()
                camera.update(dt, force_recompute=True)
                canvas = (
                    camera.data.output["rgb"][0, :, :, :3]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                rgb = Image.fromarray(canvas).transform(
                    (camera_cfg["width"], camera_cfg["height"]),
                    Image.Transform.AFFINE,
                    (1, 0, camera_cfg["crop_x"], 0, 1, camera_cfg["crop_y"]),
                    resample=Image.Resampling.BILINEAR,
                )
                draw = ImageDraw.Draw(rgb)
                draw.rectangle((0, 0, camera_cfg["width"], 42), fill=(20, 20, 20))
                draw.text(
                    (10, 7),
                    f"{'POSTROLL release/settle' if render_postroll else 'FREE SPACE' if diagnostic.get('free_space') else 'Dynamic objects'}; {controller_mode}; source frame {int(source['frame_indices'][camera_row])}; t={grip_time:.2f}s",
                    fill=(255, 255, 255),
                )
                writer.append_data(np.asarray(rgb))
                (postroll_frames if render_postroll else frames).append(
                    {
                        "source_frame": int(source["frame_indices"][camera_row]),
                        "source_time_s": timestamp,
                        "inferred_grip_time_s": grip_time,
                        "sim_time_s": step * dt,
                    }
                )
                if (
                    source_row % 20 == 0
                    or source_row == len(source_times) - 1
                    or step == steps
                ):
                    print(
                        f"Physical source frame {source_row}/{len(source_times) - 1}; object errors {[row['position_error_m'] for row in rows[-2:]]}",
                        flush=True,
                    )
                    rgb.save(
                        out
                        / (
                            f"postroll-step-{step:06d}.png"
                            if render_postroll
                            else f"source-frame-{int(source['frame_indices'][camera_row]):06d}.png"
                        )
                    )
                if render_source:
                    source_row += 1
            if step == steps:
                break
            sim.step(render=False)
            for asset in hands + objects + controls:
                asset.update(dt)
            for sensor in sensors + control_sensors:
                sensor.update(dt, force_recompute=True)
    finally:
        try:
            writer.close()
        finally:
            # Preserve partial/failed readback before summary or acceptance code.
            write(
                out / "states.json",
                {
                    "objects": rows,
                    "hands": hand_rows,
                    "controls": controls_rows,
                    "rendered_source_frames": frames,
                    "rendered_inferred_postroll_frames": postroll_frames,
                    "hand_collision_state_readback": collision_transitions,
                },
            )
    if len(frames) != len(source_times):
        raise ValueError("Physical task did not cover all source frames")
    freefall = [row for row in controls_rows if row["control"] == 0]
    supported = [row for row in controls_rows if row["control"] == 1]
    drop = freefall[0]["root_pose"][2] - freefall[-1]["root_pose"][2]
    support_force = float(
        np.mean([row["normal_contact_force"][2] for row in supported[-20:]])
    )
    controls_pass = (
        4.8 <= drop <= 5.1
        and abs(supported[-1]["root_pose"][2] - 0.05) < 0.005
        and abs(support_force - 0.3 * 9.81) < 0.3
    )
    summaries = object_tracking_summary(rows)
    write(
        out / "report.json",
        {
            "schema": "v2d.dex3-supported-task-result.v1",
            "executor_source_sha256": digest(Path(__file__)),
            "inputs_manifest_sha256": digest(inputs / "manifest.json"),
            "controller_source_sha256": digest(controller_file),
            "controller_gains": gains,
            "wrist_diagnostic": diagnostic,
            "hand_collision_state_readback_verified": bool(collision_transitions),
            "full_source_frames": original_source_frames,
            "complete_source_episode": len(frames) == original_source_frames,
            "physics_dt_s": dt,
            "initial_hand_root_reset_performed": True,
            "collision_evidence_source_sha256": digest(
                Path(__file__).with_name("_dex3_collision_evidence.py")
            ),
            "cooked_collider_summary": cooked_summary,
            "physical_contact_bindings": contact_bindings,
            "policy_residual": "zero; no policy loaded or trained",
            "task_grip_spec_sha256": prepared.get("task_grip_spec_sha256"),
            "task_grasp_seed_sha256": prepared.get("task_grasps_sha256"),
            "task_grip_controller_sha256": prepared.get("task_grip_controller_sha256"),
            "task_motor_readback": task_motor_readbacks,
            "initial_scene_clearance_check": initial_scene_check,
            "scene_projection_maximum_correction_m": task_grip.spec.get(
                "scene_projection", {}
            ).get("maximum_correction_m")
            if task_grip is not None
            else None,
            "object_errors_use_original_source_without_scene_projection": True,
            "robot_colliders_persist_through_human_occlusion": task_grip is not None,
            "robot_targets_are_inferred_grasp_commands": task_grip is not None,
            "source_frames": len(frames),
            "rendered_video_frames": len(frames) + len(postroll_frames),
            "inferred_postroll_source_seconds": postroll_s,
            "simulation_steps": steps,
            "source_motion_knots": len(motion.breakpoints),
            "source_motion_speed": speed,
            "object_teleports_after_initialization": 0,
            "external_object_forces": False,
            "support_is_an_inferred_sheet_not_measured_table": True,
            "initial_link_poses": initial_links,
            "provenance": provenance,
            "object_tracking": summaries,
            "gravity_and_contact_controls_pass": bool(controls_pass),
            "controls": {
                "free_fall_drop_m": drop,
                "supported_sphere_mean_force_N": support_force,
            },
            "state_sha256": digest(out / "states.json"),
            "video_sha256": digest(out / "supported-task.mp4"),
            "source_observation_and_target_approval_flags_changed": False,
            "physical_task_accepted": False,
            "official_evaluation_score_claimed": False,
        },
    )
    sim.stop()
    if not controls_pass:
        raise ValueError("Task physics controls failed")


if __name__ == "__main__":
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    try:
        execute(
            args.case.resolve(),
            args.repo.resolve(),
            args.inputs.resolve(),
            args.out.resolve(),
            args.manifest_sha256,
            args.device,
        )
    except Exception as error:  # noqa: BLE001 - persist errors before the SDK's process-terminating close
        import sys
        import traceback

        args.out.parent.mkdir(parents=True, exist_ok=True)
        write(
            args.out.parent / (args.out.name + ".failure.json"),
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "physical_task_accepted": False,
            },
        )
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        launcher.app.close(skip_cleanup=True)
