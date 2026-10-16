"""Bind locally verified grasp commands to every original e002 phase."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1] / "reproduction"))
from _dex3_task_grip import (  # noqa: E402
    SmoothObjectReference,
    denoise_robot_reference,
    relative_pose,
)
from _dex3_static_load import apply_grasp_seeds  # noqa: E402
from grasp_preload import URDFKinematics  # noqa: E402
from grip_force import point_jacobian, solve_gravity_contacts  # noqa: E402


def build(case, original_samples, proposal, stage_rotation, urdfs):
    """Produce robot-only commands, preserving source masks, frames and geometry."""
    samples, changed = apply_grasp_seeds(original_samples, proposal)
    if set(changed) != {s["prefix"] for s in original_samples}:
        raise ValueError(
            "Full task requires every source grasp, with no missing candidate"
        )
    with np.load(case / "bundle/episode.npz", allow_pickle=False) as f:
        source = {k: f[k] for k in f.files}
    with np.load(case / "proposal/robot-motion-proposal.npz", allow_pickle=False) as f:
        motion = {k: f[k] for k in f.files}
    stage = Rotation.from_matrix(stage_rotation)
    times = source["timestamps_s"]
    minimum_normal = float(proposal.get("task_minimum_pad_normal_N", 2.0))
    payload_mode = proposal.get("wrist_payload_compensation")
    tactile_enabled = proposal.get("experimental_tactile_seating", False)
    live_enabled = proposal.get("experimental_live_contact_wrench", False)
    live_posture = proposal.get("live_contact_posture_mode")
    interpolation = proposal.get("task_reference_interpolation", "piecewise")
    denoising = proposal.get("experimental_reference_denoising", False)
    registration = proposal.get("experimental_register_settled_grasp", False)
    load_model = proposal.get("live_contact_load_model", "gravity")
    bounded_reacquisition = proposal.get("experimental_bounded_reacquisition", False)
    patch_minimum = float(proposal.get("minimum_live_patch_force_N", 0.5))
    speed = float(proposal.get("source_motion_speed", 0.5))
    if (
        interpolation not in {"piecewise", "cubic"}
        or not isinstance(denoising, bool)
        or not isinstance(registration, bool)
        or ((denoising or registration) and interpolation != "cubic")
        or (registration and not bounded_reacquisition)
        or load_model not in {"gravity", "reference_acceleration"}
        or not np.isfinite(speed)
        or not 0.1 <= speed <= 0.5
        or not isinstance(bounded_reacquisition, bool)
        or not 0.05 <= patch_minimum <= 0.5
        or (
            load_model == "reference_acceleration"
            and (not live_enabled or interpolation != "cubic")
        )
    ):
        raise ValueError("Invalid explicit robot reference/load configuration")
    if not 2 <= minimum_normal <= 5 or payload_mode not in {
        "nominal_gravity",
        "measured_reaction",
        "measured_primary_reaction",
    }:
        raise ValueError("Unsupported explicit coupled controller configuration")
    if (
        not isinstance(tactile_enabled, bool)
        or not isinstance(live_enabled, bool)
        or (tactile_enabled and live_enabled)
    ):
        raise ValueError(
            "Experimental finger controllers need explicit mutually exclusive booleans"
        )
    if live_enabled and live_posture not in {
        "current_position",
        "frozen_reference",
        "normal_nullspace",
    }:
        raise ValueError("Live contact allocation requires an explicit posture mode")
    if not live_enabled and live_posture is not None:
        raise ValueError("A live posture mode requires live contact allocation")
    result = dict(
        schema="v2d.dex3-task-grips.v1",
        park_height_m=0.12,
        physical_task_accepted=False,
        source_observations_changed=False,
        case_manifest_sha256=hashlib.sha256(
            (case / "file-manifest.json").read_bytes()
        ).hexdigest(),
        source_times_s=times.tolist(),
        phases={"left": [], "right": []},
        observation_contract="source object reference plus measured object pose during approach; measured hand-object reaction for wrist feedback; diagnostic only",
        contact_model_friction=1.25,
        minimum_planned_pad_normal_N=minimum_normal,
        wrist_payload_compensation=payload_mode,
        experimental_tactile_seating=tactile_enabled,
        experimental_live_contact_wrench=live_enabled,
        live_contact_posture_mode=live_posture,
        task_reference_interpolation=interpolation,
        experimental_reference_denoising=denoising,
        experimental_register_settled_grasp=registration,
        live_contact_load_model=load_model,
        source_motion_speed=speed,
        experimental_bounded_reacquisition=bounded_reacquisition,
        minimum_live_patch_force_N=patch_minimum,
        planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        required_physical_materials="hand 2.0 / object 0.5, average combination = 1.25; explicit combine mode and coefficient readback required",
    )
    controls = {r["sample"]: r for r in proposal["grip_effort"]["controls"]}
    if "scene_projection" in proposal:
        scene = proposal["scene_projection"]
        values = np.asarray(scene["upward_corrections_m"])
        if (
            scene["schema"] != "v2d.cooked-scene-projection.v1"
            or scene["original_source_arrays_changed"]
            or scene["physical_task_accepted"]
            or not np.array_equal(scene["source_times_s"], times)
            or values.shape != (len(times), 2)
            or not np.isfinite(values).all()
            or values.min() < 0
            or values.max() > 0.06
            or scene["cap_m"] != 0.06
        ):
            raise ValueError("Invalid bounded scene projection proposal")
        result["scene_projection"] = copy.deepcopy(scene)
    object_references = {}
    if interpolation == "cubic":
        references, audits = {}, {}
        for oi in sorted({sample["object"] for sample in samples}):
            matrix = source[f"object_{oi}_world"]
            poses = np.column_stack(
                (
                    stage.apply(matrix[:, :3, 3]),
                    (stage * Rotation.from_matrix(matrix[:, :3, :3])).as_quat(),
                )
            )
            if "scene_projection" in result:
                poses[:, 2] += np.asarray(
                    result["scene_projection"]["upward_corrections_m"]
                )[:, oi]
            baseline = SmoothObjectReference(times, poses)
            if denoising:
                poses = denoise_robot_reference(times, poses)
            reference = SmoothObjectReference(times, poses)
            clearance_raise = 0.0
            if denoising and "scene_projection" in result:
                # Smoothing rotation/height can reintroduce a table crossing.
                # A constant offset preserves derivatives; dense readback is
                # diagnostic, not a continuous collision certificate.
                scene = result["scene_projection"]
                vertices = np.concatenate(scene["expected_body_local_convexes"][oi])
                grid = np.linspace(times[0], times[-1], 20 * (len(times) - 1) + 1)
                bottom = reference.position(grid)[:, 2] + np.einsum(
                    "ij,kj->ik", reference.rotation(grid).as_matrix()[:, 2, :], vertices
                ).min(axis=1)
                clearance_raise = max(
                    0.0,
                    scene["inferred_support_height_stage_m"]
                    + 0.0002
                    - float(bottom.min()),
                )
                poses[:, 2] += clearance_raise
                reference = SmoothObjectReference(times, poses)
            object_references[oi] = reference
            audit = reference.deviation_audit()
            if denoising:
                grid = np.linspace(times[0], times[-1], 20 * (len(times) - 1) + 1)
                audit.update(
                    denoising_horizon_source_s=0.1,
                    denoising_table_clearance_raise_m=float(clearance_raise),
                    maximum_position_change_from_unsmoothed_cubic_m=float(
                        np.linalg.norm(
                            reference.position(grid) - baseline.position(grid), axis=1
                        ).max()
                    ),
                    maximum_rotation_change_from_unsmoothed_cubic_deg=float(
                        np.rad2deg(
                            (
                                reference.rotation(grid) * baseline.rotation(grid).inv()
                            ).magnitude()
                        ).max()
                    ),
                    knots_preserved=False,
                    original_source_arrays_changed=False,
                    target_knots_only_denoised=True,
                )
                if (
                    audit["maximum_position_change_from_unsmoothed_cubic_m"] > 0.006
                    or audit["maximum_rotation_change_from_unsmoothed_cubic_deg"] > 3
                ):
                    raise ValueError(
                        "Denoised robot target exceeds 6 mm / 3 degree path bound"
                    )
            if (
                audit["maximum_position_deviation_from_piecewise_m"] > 0.006
                or audit["maximum_rotation_deviation_from_piecewise_deg"] > 3
            ):
                raise ValueError(
                    f"Smooth robot reference exceeds 6 mm / 3 degree interpolation bounds: object {oi}: {audit}"
                )
            references[str(oi)], audits[str(oi)] = poses.tolist(), audit
        result["smooth_robot_object_reference"] = references
        result["smooth_robot_reference_audit"] = audits
    for sample in samples:
        side, oi = sample["side"], sample["object"]
        si = ("left", "right").index(side)
        record = next(r for r in proposal["results"] if r["sample"] == sample["prefix"])
        phase_id = motion["robot_grasp_phase_index_inferred"][sample["source_row"], si]
        phase_rows = np.flatnonzero(
            motion["robot_grasp_phase_index_inferred"][:, si] == phase_id
        )
        contact_rows = phase_rows[
            motion["robot_contact_intent_inferred"][phase_rows, si, oi]
        ]
        close, end = (
            float(times[contact_rows[0]]),
            float(times[contact_rows[-1]] + np.median(np.diff(times))),
        )
        start = float(times[phase_rows[0]] - 0.5)
        if close - 0.25 <= start:
            raise ValueError(
                "Insufficient source interval for explicit approach and closure"
            )
        kin = URDFKinematics(urdfs / f"dex3_{side}.urdf")
        q = dict(zip(sample["joint_names"], sample["finger_target_rad"], strict=True))
        root = np.r_[
            sample["hand_position_stage_m"], sample["hand_quaternion_stage_xyzw"]
        ]
        obj = np.r_[
            sample["object_position_stage_m"], sample["object_quaternion_stage_xyzw"]
        ]
        relative = relative_pose(root, obj)
        # Fixture world offsets cancel in the relative transform.
        order = [
            record["candidate"]["joint_names"].index(n) for n in sample["joint_names"]
        ]
        effort_times, efforts = times[contact_rows], []
        if record["candidate"].get("three_pad_geometric_contacts"):
            pads = controls[sample["prefix"]]["pads"]
            poses = kin.poses(root, q)
            points = np.array(
                [
                    poses[p["link"]][:3, :3] @ p["contact_point_link_m"]
                    + poses[p["link"]][:3, 3]
                    for p in pads
                ]
            )
            inward = np.array([p["inward_direction_stage"] for p in pads])
            jac = np.array(
                [
                    point_jacobian(
                        kin, root, q, p["link"], np.asarray(p["contact_point_link_m"])
                    )
                    for p in pads
                ]
            )
            for row in contact_rows:
                rotation = (
                    stage
                    * Rotation.from_matrix(source[f"object_{oi}_world"][row, :3, :3])
                    * Rotation.from_quat(obj[3:]).inv()
                )
                try:
                    _, torque, _ = solve_gravity_contacts(
                        rotation.apply(points - obj[:3]),
                        rotation.apply(inward),
                        np.einsum("ij,pjk->pik", rotation.as_matrix(), jac),
                        np.zeros(3),
                        0.3,
                        10.0,
                        friction=1.25,
                        minimum_normal_N=minimum_normal,
                        desired_wrench_stage=object_references[oi].load_wrench(
                            float(times[row]), speed
                        )
                        if load_model == "reference_acceleration"
                        else None,
                    )
                except ValueError as error:
                    raise ValueError(
                        f"{sample['prefix']} source row {row}: {error}"
                    ) from error
                efforts.append(
                    [torque[kin.names.index(n)] for n in sample["joint_names"]]
                )
        else:
            efforts = [
                np.array(record["candidate"]["finger_feedforward_Nm"])[order].tolist()
            ] * len(contact_rows)
        result["phases"][side].append(
            dict(
                sample=sample["prefix"],
                object=oi,
                joint_names=sample["joint_names"],
                hand_in_object_xyzw=relative.tolist(),
                open_rad=sample["finger_pregrasp_rad"],
                closed_rad=sample["finger_target_rad"],
                effort_times_s=effort_times.tolist(),
                effort_Nm=efforts,
                stiffness_Nm_per_rad=sample.get("finger_stiffness_Nm_per_rad", 2),
                required_digits=sample.get(
                    "required_contact_digits", ["thumb", "index"]
                ),
                pad_normal_jacobians=[
                    dict(
                        link=p["link"],
                        normal_jacobian_m_per_rad=[
                            p["normal_jacobian_m_per_rad"][kin.names.index(n)]
                            for n in sample["joint_names"]
                        ],
                    )
                    for p in controls[sample["prefix"]]["pads"]
                ],
                joint_limits_rad=[list(kin.limits[n]) for n in sample["joint_names"]],
                approach_start_s=start,
                close_s=close,
                release_start_s=end,
            )
        )
    for side, phases in result["phases"].items():
        phases.sort(key=lambda x: x["close_s"])
        if any(
            b["approach_start_s"] < a["release_start_s"] + 0.75
            for a, b in zip(phases, phases[1:])
        ):
            raise ValueError(f"{side}: approach/release schedules overlap")
        if len({p["stiffness_Nm_per_rad"] for p in phases}) != 1:
            raise ValueError("Per-phase gains must agree for the same hand")
    return result


def combine(paths):
    """Combine two exact local seeds; never modify their source identity records."""
    proposals = [json.loads(Path(p).read_text()) for p in paths]
    for p in proposals:
        apply_grasp_seeds([r["source_sample"] for r in p["results"]], p)
    result = copy.deepcopy(proposals[0])
    result["results"] = [copy.deepcopy(r) for p in proposals for r in p["results"]]
    result["grip_effort"]["controls"] = [
        copy.deepcopy(r) for p in proposals for r in p["grip_effort"]["controls"]
    ]
    for key, default in {
        "wrist_translation_cap_m": 0.03,
        "wrist_rotation_cap_rad": 0.5,
        "finger_delta_cap_rad": 0.6,
    }.items():
        result[key] = max(p.get(key, default) for p in proposals)
    result["component_seed_sha256"] = [
        hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths
    ]
    result.update(
        task_minimum_pad_normal_N=2.0,
        wrist_payload_compensation="measured_reaction",
        experimental_tactile_seating=False,
        experimental_live_contact_wrench=False,
    )
    return result
