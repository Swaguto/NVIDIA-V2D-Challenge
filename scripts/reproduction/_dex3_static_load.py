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


def prepare(case: Path, repo: Path, plane: Path, plane_sha: str, out: Path) -> dict:
    """Validate byte/coordinate contracts and build portable nominal load inputs."""
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation

    if out.exists():
        raise FileExistsError("Use a fresh load-input directory")
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
    object_files = {}
    for sample in samples:
        oi = sample["object"]
        obj = next(obj for obj in bundle["objects"] if obj["index"] == oi)
        vertices = (
            source[f"object_{oi}_vertices"] * sample["object_scale_at_selected_frame"]
        )
        faces = source[f"object_{oi}_faces"]
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
        "fixed_wrists_are_a_fixture_not_whole_body_feasibility": True,
        "source_observations_or_robot_motion_changed": False,
        "per_frame_source_scale_preserved_for_each_static_fixture": True,
        "physical_task_accepted": False,
    }
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
    from pxr import Usd, UsdPhysics, UsdShade
    from scipy.spatial.transform import Rotation

    manifest = inputs / "manifest.json"
    if digest(manifest) != manifest_sha:
        raise ValueError("Prepared physical input manifest changed")
    prepared = json.loads(manifest.read_text())
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
    dt = 0.01  # Existing V2DHandEnvCfg simulation step.
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

    def bind_material(prim_path, coefficient):
        material_path = prim_path + "/DiagnosticMaterial"
        cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=coefficient, dynamic_friction=coefficient, restitution=0.0
        )
        cfg.func(material_path, cfg)
        material = UsdShade.Material(sim.stage.GetPrimAtPath(material_path))
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

        def physical_hand(
            *, cfg, prim_root=prim_root, sample=sample, target=target, limits=limits
        ):
            cfg.prim_path = prim_root + "/Hand"
            cfg.spawn.fix_base = True
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
                stiffness=motors["STIFFNESS_1515"],
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
        sim_utils.modify_collision_properties(cfg.prim_path, cfg.spawn.collision_props)
        object_material = bind_material(
            cfg.prim_path, float(sim_cfg.physics_material.dynamic_friction)
        )
        sensor = ContactSensor(
            ContactSensorCfg(
                prim_path=cfg.prim_path + "/.*", update_period=0.0, history_length=1
            )
        )
        position = np.asarray(sample["object_position_stage_m"])
        camera = Camera(
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
        hands.append(hand)
        objects.append(obj)
        sensors.append(sensor)
        cameras.append((camera, position))
        provenance.append(
            {
                "hand": hand_provenance,
                "factory_AST_sha256": factory_ast,
                "hand_material": hand_material,
                "object_material": object_material,
                "motor_effort_caps_Nm": hand.cfg.actuators["fingers"].effort_limit_sim,
                "sample": sample,
            }
        )

    # Two independent physics controls: unsupported free fall and a sphere on a local fixed platform.
    control_objects, control_sensors = [], []
    for index, x in enumerate((14.0, 16.0)):
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
    sim.reset()
    for camera, position in cameras:
        camera.set_world_poses_from_view(
            torch.tensor(
                (position + [0.0, -0.55, 0.35])[None],
                dtype=torch.float32,
                device=sim.device,
            ),
            torch.tensor(position[None], dtype=torch.float32, device=sim.device),
        )
    targets = []
    for hand, sample in zip(hands, prepared["samples"], strict=True):
        values = dict(
            zip(sample["joint_names"], sample["finger_target_rad"], strict=True)
        )
        if set(hand.joint_names) != set(values) or not hand.is_fixed_base:
            raise ValueError(
                "Physical articulation topology differs from declared fixture"
            )
        q = torch.tensor(
            [[values[name] for name in hand.joint_names]],
            dtype=torch.float32,
            device=sim.device,
        )
        hand.write_joint_state_to_sim(q, torch.zeros_like(q))
        hand.set_joint_position_target(q)
        hand.write_data_to_sim()
        # Reuse the frozen renderer's explicit FK refresh before the first render/contact step.
        _ = hand.data.body_link_pose_w
        targets.append(q)
    for obj in objects + control_objects:
        # One initialization only. No object state writes or external forces during load.
        obj.write_root_state_to_sim(obj.data.default_root_state.clone())
        obj.reset()
    frames = []
    records = []
    control_records = []
    sim.render()
    initial_readback = []
    for hand, obj, target, sample in zip(
        hands, objects, targets, prepared["samples"], strict=True
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
        body_poses = hand.data.body_link_pose_w[0].detach().cpu().numpy()
        record["hand_body_link_pose_stage_xyzw"] = {
            name: pose.tolist()
            for name, pose in zip(hand.body_names, body_poses, strict=True)
        }
        initial_readback.append(record)
    for step in range(101):
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
                    "object_root_state_stage_xyzw": state.tolist(),
                    "contact_force_stage_N": contact.tolist(),
                    "finger_position_rad_sim_order": joint.tolist(),
                    "finger_names_sim_order": hand.joint_names,
                    "estimated_implicit_actuator_torque_Nm": hand.data.applied_torque[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                }
            )
        for index, (obj, sensor) in enumerate(
            zip(control_objects, control_sensors, strict=True)
        ):
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
                    f"{sample['side']} object {sample['object']} frame {sample['source_frame']}",
                    (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    bgr,
                    f"gravity/contact ON   t={step * dt:.2f}s",
                    (6, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                images.append(bgr)
            frames.append(np.concatenate(images, axis=1))
        if step == 100:
            break
        for hand, target in zip(hands, targets, strict=True):
            hand.set_joint_position_target(target)
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
        distance = np.linalg.norm(positions - positions[0], axis=1)
        angles = (
            Rotation.from_quat(quats) * Rotation.from_quat(quats[0]).inv()
        ).magnitude()
        forces = np.array([row["contact_force_stage_N"] for row in rows])
        summaries.append(
            {
                "sample": sample["prefix"],
                "source_frame": sample["source_frame"],
                "final_displacement_m": float(distance[-1]),
                "maximum_displacement_m": float(distance.max()),
                "maximum_orientation_change_deg": float(np.rad2deg(angles.max())),
                "maximum_contact_force_N": float(np.linalg.norm(forces, axis=1).max()),
                "mean_vertical_contact_force_last_0_2s_N": float(
                    forces[-20:, 2].mean()
                ),
                "held_within_10mm_for_1s_diagnostic_only": bool(distance.max() <= 0.01),
            }
        )
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
            "executor_source_sha256": digest(Path(__file__)),
            "inputs_manifest_sha256": manifest_sha,
            "runtime_python": runtime_platform.python_version(),
            "platform": runtime_platform.platform(),
            "device": device,
            "simulation_dt_s": dt,
            "duration_s": 1.0,
            "gravity_m_s2": list(sim_cfg.gravity),
            "motor_constants": motors,
            "object_collision_import_migration": "Legacy collider_type=convex_decomposition becomes Sim6 collision_type=Convex Decomposition; other preset fields are reused",
            "provenance": provenance,
            "initial_pose_readback": initial_readback,
            "summaries": summaries,
            "controls": controls,
            "gravity_and_contact_controls_pass": controls["free_fall_pass"]
            and controls["supported_sphere_pass"],
            "states_sha256": digest(out / "states.json"),
            "video_sha256": digest(video_path),
            "not_a_complete_episode_or_official_policy_rollout": True,
            "source_observations_or_robot_motion_changed": False,
            "physical_task_accepted": False,
        },
    )
    sim.stop()
    if not all((controls["free_fall_pass"], controls["supported_sphere_pass"])):
        raise RuntimeError("Physics controls failed; do not interpret grasp outcomes")


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
