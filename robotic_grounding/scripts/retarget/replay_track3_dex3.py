# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render every frozen episode frame in Isaac Lab 2.3.2 / Isaac Sim 5.1.

This is kinematic articulation playback, with gravity and collisions disabled.
It verifies simulator joint/root readback, preserves unknown-hand visibility,
and writes a completion receipt only after the entire video decodes correctly.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source/robotic_grounding"))
import isaaclab  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402
from robotic_grounding.retarget.track3_bundle import (  # noqa: E402
    SIDES,
    digest,
    validate_bundle,
    write_json,
)  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--retarget", type=Path, required=True)
parser.add_argument("--gate", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(f"Use a fresh output directory: {args.out}")
if not args.enable_cameras:
    parser.error("--enable_cameras is required for real renderer evidence")
verified = validate_bundle(args.bundle)
contract = json.loads((args.retarget / "manifest.json").read_text())
gate_manifest = json.loads((args.gate / "manifest.json").read_text())
if (
    contract["episode_bundle_manifest_sha256"] != verified["manifest_sha256"]
    or contract["payload"]["sha256"] != digest(args.retarget / "retarget.npz")
    or gate_manifest["bundle_manifest_sha256"] != verified["manifest_sha256"]
    or gate_manifest["retarget_manifest_sha256"]
    != digest(args.retarget / "manifest.json")
    or gate_manifest["retarget_payload_sha256"] != contract["payload"]["sha256"]
    or gate_manifest["payload"]["sha256"] != digest(args.gate / "penetration.npz")
):
    raise ValueError("Replay inputs do not match the frozen candidate")
data = np.load(args.bundle / "episode.npz")
trajectory = np.load(args.retarget / "retarget.npz")
gate = np.load(args.gate / "penetration.npz")
frames, valid = data["frame_indices"], trajectory["robot_valid"]
if (
    contract["processed_frames"] != len(frames)
    or not np.array_equal(valid, data["hand_valid"])
    or not np.array_equal(frames, trajectory["frame_indices"])
    or not np.array_equal(frames, gate["frame_indices"])
):
    raise ValueError("Replay requires the full, source-preserving trajectory")
if args.width < 64 or args.height < 64 or args.width % 2 or args.height % 2:
    parser.error("Renderer dimensions must be even and at least 64 pixels")
for package, prefix in (("isaacsim", "5.1.0"),):
    version = importlib.metadata.version(package)
    if not (version == prefix or version.startswith(prefix + ".")):
        raise ValueError(f"Expected {package} {prefix}, got {version}")
lab_release_file = Path(isaaclab.__file__).resolve().parents[3] / "VERSION"
lab_release = lab_release_file.read_text().strip()
if lab_release != "2.3.2":
    raise ValueError(f"Expected Isaac Lab release 2.3.2, got {lab_release}")
args.out.mkdir(parents=True)
if args.device == "cpu":
    args.kit_args += " --/app/useFabricSceneDelegate=false --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false"
launcher = AppLauncher(args)
app = launcher.app

# Simulator modules must be imported after the Kit application launches.
import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from omni.physx import get_physx_interface  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdShade, Vt  # noqa: E402

ASSETS = REPO / "source/robotic_grounding/robotic_grounding/assets"
# Proper presentation transform: reference-camera X right, Y down, Z forward
# becomes stage X right, Y forward, Z up. This is not measured gravity.
WORLD_TO_STAGE = np.eye(4)
WORLD_TO_STAGE[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]


def resolved_urdf(side: str) -> tuple[Path, dict[str, Any]]:
    """Resolve local mesh paths and retain URDF and mesh hashes."""
    source = ASSETS / f"urdfs/g1/dex3_{side}.urdf"
    tree = ET.parse(source)
    meshes = {}
    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.attrib["filename"]
        if not filename.startswith("package://"):
            raise ValueError("Unexpected mesh reference")
        path = ASSETS / "meshes" / filename.removeprefix("package://")
        with path.open("rb") as handle:
            if handle.read(80).startswith(b"version https://git-lfs"):
                raise ValueError(f"Unretrieved hand mesh: {path}")
        meshes[filename] = digest(path)
        mesh.attrib["filename"] = path.resolve().as_posix()
    # Review colors identify each robot hand without changing its geometry.
    color = "0.07 0.25 0.7 1" if side == "left" else "0.7 0.15 0.04 1"
    for element in tree.getroot().iter("color"):
        element.attrib["rgba"] = color
    target = args.out / f"dex3_{side}.urdf"
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return target, {
        "urdf_sha256": digest(source),
        "resolved_urdf_sha256": digest(target),
        "meshes": meshes,
    }


def create_hand(side: str) -> tuple[Articulation, dict[str, Any]]:
    """Import a seven-joint Dex3 articulation and retain asset provenance."""
    urdf, provenance = resolved_urdf(side)
    cfg = ArticulationCfg(
        prim_path=f"/World/Hands/{side}",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf),
            fix_base=False,
            merge_fixed_joints=False,
            make_instanceable=False,
            force_usd_conversion=True,
            usd_dir=str(args.out / f"usd_{side}"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0, damping=0
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=0.0, damping=0.0
            )
        },
    )
    return Articulation(cfg=cfg), provenance


def main() -> None:
    """Run the command with validated inputs and explicit output artifacts."""
    start = time.monotonic()
    fps = 1 / float(np.median(np.diff(data["timestamps_s"])))
    # CPU physics writes visual transforms to USD. Fabric is required for GPU
    # physics but its CPU transform path is unsupported in this Lab release.
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1 / fps, device=args.device, use_fabric=args.device != "cpu"
        )
    )
    light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.85, 0.85, 0.85))
    light.func("/World/Light", light)
    hands, hand_assets = {}, {}
    for side in SIDES:
        hands[side], hand_assets[side] = create_hand(side)
        # The native importer still instances each visual/collision mesh even
        # when the converter-level instance option is disabled. This single
        # environment needs authored review materials and disabled colliders.
        sim_utils.make_uninstanceable(hands[side].cfg.prim_path)
        sim_utils.modify_collision_properties(
            hands[side].cfg.prim_path,
            sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )
        material_path = f"/World/ReviewMaterials/{side}"
        material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.07, 0.25, 0.7) if side == "left" else (0.7, 0.15, 0.04),
            metallic=0.0,
            roughness=0.65,
        )
        material.func(material_path, material)
        sim_utils.bind_visual_material(
            hands[side].cfg.prim_path, material_path, stronger_than_descendants=True
        )
    objects = json.loads((args.bundle / "manifest.json").read_text())["objects"]
    object_ops, all_bounds = {}, []
    for obj in objects:
        index = obj["index"]
        parent = UsdGeom.Xform.Define(sim.stage, f"/World/Objects/object_{index}")
        object_ops[index] = parent.AddTransformOp()
        mesh = UsdGeom.Mesh.Define(sim.stage, f"/World/Objects/object_{index}/Mesh")
        vertices = data[f"object_{index}_vertices"]
        faces = data[f"object_{index}_faces"]
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray(faces.astype(np.int32).reshape(-1).tolist())
        )
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr(
            [(0.65, 0.72, 0.8) if index == 0 else (0.8, 0.75, 0.6)]
        )
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        corners = np.array(
            [
                [x, y, z]
                for x in bounds[:, 0]
                for y in bounds[:, 1]
                for z in bounds[:, 2]
            ]
        )
        poses = np.einsum("ij,tjk->tik", WORLD_TO_STAGE, data[f"object_{index}_world"])
        world = (
            np.einsum("tij,kj->tki", poses[:, :3, :3], corners)
            * data[f"object_{index}_scale"][:, None, None]
            + poses[:, None, :3, 3]
        )
        all_bounds.append(world.reshape(-1, 3))
    camera = Camera(
        CameraCfg(
            prim_path="/World/ReviewCamera",
            update_period=0.0,
            width=args.width,
            height=args.height,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=24.0,
                clipping_range=(0.01, 100.0),
            ),
        )
    )
    sim.reset()
    if args.device == "cpu":
        sim.set_setting("/physics/updateToUsd", True)
    mesh_diagnostics: dict[str, list[dict[str, Any]]] = {}
    for side, hand in hands.items():
        records: list[dict[str, Any]] = []
        for prim in Usd.PrimRange(
            sim.stage.GetPrimAtPath(hand.cfg.prim_path), Usd.TraverseInstanceProxies()
        ):
            if prim.GetTypeName() != "Mesh" or "/visuals/" not in str(prim.GetPath()):
                continue
            points_attr = prim.GetAttribute("points").Get()
            review_color = (0.07, 0.25, 0.7) if side == "left" else (0.7, 0.15, 0.04)
            display = UsdGeom.Mesh(prim).CreateDisplayColorAttr(
                [Gf.Vec3f(*review_color)]
            )
            display.SetMetadata("interpolation", "constant")
            binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            records.append(
                {
                    "path": str(prim.GetPath()),
                    "point_count": len(points_attr) if points_attr is not None else 0,
                    "visibility": str(UsdGeom.Imageable(prim).ComputeVisibility()),
                    "purpose": str(UsdGeom.Imageable(prim).ComputePurpose()),
                    "material": str(binding.GetPath()),
                }
            )
        mesh_diagnostics[side] = records
        if not records or not all(record["point_count"] > 0 for record in records):
            raise ValueError(f"Imported {side} hand has missing visual geometry")
    write_json(args.out / "imported-visual-geometry.json", mesh_diagnostics)
    joint_order = {}
    for side, hand in hands.items():
        declared = contract["joints"][side]["finger_joint_names"]
        if len(hand.joint_names) != len(declared) or set(hand.joint_names) != set(
            declared
        ):
            raise ValueError(
                f"Simulator joint names differ from retarget model: {side}"
            )
        joint_order[side] = [declared.index(name) for name in hand.joint_names]
    points = np.concatenate(all_bounds)
    target = (points.min(axis=0) + points.max(axis=0)) / 2
    span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    eye = target + np.array([0.65, -1.0, 0.7]) * max(span, 0.45)
    camera.set_world_poses_from_view(
        torch.tensor(eye[None], dtype=torch.float32, device=sim.device),
        torch.tensor(target[None], dtype=torch.float32, device=sim.device),
    )
    sim.set_camera_view(eye.tolist(), target.tolist())
    for _ in range(12):
        sim.render()
    camera.update(1 / fps, force_recompute=True)
    observed = np.full_like(trajectory["robot_qpos"], np.nan)
    root_error = np.full((len(frames), 2), np.nan)
    joint_error = root_error.copy()
    orientation_error = root_error.copy()
    usd_link_error = root_error.copy()
    physics_link_error = root_error.copy()
    color_pixels = np.zeros((len(frames), 2), dtype=int)
    video = args.out / "replay.mp4"
    raw = args.out / "frames"
    raw.mkdir()
    writer = imageio.get_writer(
        video, fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    try:
        for row, frame in enumerate(frames):
            targets = {}
            for side_id, side in enumerate(SIDES):
                hand = hands[side]
                imageable = UsdGeom.Imageable(
                    sim.stage.GetPrimAtPath(hand.cfg.prim_path)
                )
                if not valid[row, side_id]:
                    imageable.MakeInvisible()
                    continue
                imageable.MakeVisible()
                q = trajectory["robot_qpos"][row, side_id]
                pose = WORLD_TO_STAGE[:3, :3] @ Rotation.from_quat(q[3:7]).as_matrix()
                position = WORLD_TO_STAGE[:3, :3] @ q[:3]
                root = np.r_[
                    position, Rotation.from_matrix(pose).as_quat(scalar_first=True)
                ]
                joints = q[7:][joint_order[side]]
                targets[side] = (root, joints)
                hand.write_root_pose_to_sim(
                    torch.tensor(root[None], dtype=torch.float32, device=sim.device)
                )
                hand.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
                joint_tensor = torch.tensor(
                    joints[None], dtype=torch.float32, device=sim.device
                )
                hand.write_joint_state_to_sim(
                    joint_tensor, torch.zeros_like(joint_tensor)
                )
                hand.set_joint_position_target(joint_tensor)
                hand.write_data_to_sim()
                # Teleporting joint/base state requires an explicit FK refresh
                # before the visual step in this CPU camera pipeline.
                _ = hand.data.body_link_pose_w
            for obj in objects:
                index = obj["index"]
                pose = WORLD_TO_STAGE @ data[f"object_{index}_world"][row]
                pose[:3, :3] *= float(data[f"object_{index}_scale"][row])
                object_ops[index].Set(Gf.Matrix4d(pose.T.tolist()))
            sim.step(render=False)
            for side_id, side in enumerate(SIDES):
                hand = hands[side]
                hand.update(1 / fps)
                if side not in targets:
                    continue
                root, joints = targets[side]
                actual_root = hand.data.root_state_w[0, :7].cpu().numpy()
                actual_joints = hand.data.joint_pos[0].cpu().numpy()
                if (
                    not np.isfinite(actual_root).all()
                    or not np.isfinite(actual_joints).all()
                ):
                    raise ValueError(f"Nonfinite simulator state: {side}/{frame}")
                root_error[row, side_id] = np.linalg.norm(actual_root[:3] - root[:3])
                joint_error[row, side_id] = np.max(np.abs(actual_joints - joints))
                orientation_error[row, side_id] = (
                    Rotation.from_quat(root[3:], scalar_first=True).inv()
                    * Rotation.from_quat(actual_root[3:], scalar_first=True)
                ).magnitude()
                actual_source_xyz = WORLD_TO_STAGE[:3, :3].T @ actual_root[:3]
                actual_source_rot = (
                    WORLD_TO_STAGE[:3, :3].T
                    @ Rotation.from_quat(actual_root[3:], scalar_first=True).as_matrix()
                )
                observed[row, side_id, :7] = np.r_[
                    actual_source_xyz, Rotation.from_matrix(actual_source_rot).as_quat()
                ]
                observed[row, side_id, 7:] = actual_joints[
                    np.argsort(joint_order[side])
                ]
                body_poses = hand.data.body_link_pose_w[0].cpu().numpy()
                frame_poses = trajectory[f"robot_{side}_frame_poses"][row]
                names = contract["joints"][side]["frame_names"]
                differences = []
                for record in mesh_diagnostics[side]:
                    link_name = record["path"].split("/visuals/")[0].split("/")[-1]
                    expected_position = (
                        WORLD_TO_STAGE[:3, :3] @ frame_poses[names.index(link_name), :3]
                    )
                    differences.append(
                        np.linalg.norm(
                            body_poses[hand.body_names.index(link_name), :3]
                            - expected_position
                        )
                    )
                physics_link_error[row, side_id] = max(differences)
            if args.device == "cpu":
                # This is the same explicit PhysX-to-USD writeback used by
                # Isaac's camera tests. The renderer receives actual physics
                # body poses, not a separately animated target mesh.
                get_physx_interface().update_transformations(False, True, False)
            sim.render()
            for side_id, side in enumerate(SIDES):
                if side not in targets:
                    continue
                hand = hands[side]
                cache = UsdGeom.XformCache()
                names = contract["joints"][side]["frame_names"]
                frame_poses = trajectory[f"robot_{side}_frame_poses"][row]
                positions = []
                for record in mesh_diagnostics[side]:
                    link = record["path"].split("/visuals/")[0]
                    link_name = link.split("/")[-1]
                    rendered_transform = np.asarray(
                        cache.GetLocalToWorldTransform(sim.stage.GetPrimAtPath(link))
                    ).T
                    expected_position = (
                        WORLD_TO_STAGE[:3, :3] @ frame_poses[names.index(link_name), :3]
                    )
                    positions.append(
                        np.linalg.norm(rendered_transform[:3, 3] - expected_position)
                    )
                usd_link_error[row, side_id] = max(positions)
            camera.update(1 / fps, force_recompute=True)
            rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            if rgb.shape != (args.height, args.width, 3) or np.std(rgb) < 1.0:
                raise ValueError(f"Missing or blank renderer output at frame {frame}")
            colors = rgb.astype(float)
            color_pixels[row, 0] = np.count_nonzero(
                (colors[..., 2] > 1.5 * colors[..., 0])
                & (colors[..., 2] > 1.3 * colors[..., 1])
                & (colors[..., 2] > 40)
            )
            color_pixels[row, 1] = np.count_nonzero(
                (colors[..., 0] > 1.5 * colors[..., 2])
                & (colors[..., 0] > 1.2 * colors[..., 1])
                & (colors[..., 0] > 40)
            )
            Image.fromarray(rgb).save(raw / f"{int(frame):06d}.png")
            labelled = Image.fromarray(rgb)
            draw = ImageDraw.Draw(labelled)
            draw.rectangle((0, 0, args.width, 57), fill=(20, 20, 20))
            if (
                gate_manifest.get("collision_model_mode")
                == "documented_derived_voxel_material"
            ):
                if gate["fully_checked_derived_model_pass"][row]:
                    state = "DERIVED MODEL PASS"
                elif gate["measured_bound_violation"][row]:
                    state = "REJECTED: >2 CM"
                elif gate["bound_uncertain"][row]:
                    state = "REJECTED: UNCERTAIN BOUND"
                else:
                    state = "UNSUPPORTED: MISSING HAND"
            else:
                state = (
                    "REJECTED"
                    if gate["rejected"][row]
                    else "OBJECT CHECKS EXCLUDED / REVIEW"
                )
            draw.text(
                (8, 5),
                f"Episode 2 frame {frame} / {len(frames)-1} | kinematic Isaac Lab replay",
                fill="white",
            )
            draw.text(
                (8, 22),
                f"Left {'observed' if valid[row,0] else 'UNKNOWN'} | Right {'observed' if valid[row,1] else 'UNKNOWN'}",
                fill="white",
            )
            draw.text((8, 39), state, fill=(255, 190, 80))
            writer.append_data(np.asarray(labelled))
            if (row + 1) % 25 == 0 or row + 1 == len(frames):
                message = {
                    "rendered_frames": row + 1,
                    "elapsed_seconds": time.monotonic() - start,
                }
                with (args.out / "progress.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(message) + "\n")
                print(json.dumps(message), flush=True)
    finally:
        writer.close()
    with imageio.get_reader(video) as reader:
        decoded_frames = reader.count_frames()
        encoded_fps = reader.get_meta_data()["fps"]
    if decoded_frames != len(frames) or abs(encoded_fps - fps) > 0.01:
        raise ValueError("Encoded video did not retain the complete source timeline")
    numerical = {
        "maximum_root_error_m": float(np.nanmax(root_error)),
        "maximum_joint_error_rad": float(np.nanmax(joint_error)),
        "maximum_usd_link_error_m": float(np.nanmax(usd_link_error)),
        "maximum_physics_link_error_m": float(np.nanmax(physics_link_error)),
        "maximum_color_pixels": color_pixels.max(axis=0).tolist(),
    }
    write_json(args.out / "numerical-checks.json", numerical)
    np.savez_compressed(
        args.out / "simulator-readback.npz",
        frame_indices=frames,
        robot_valid=valid,
        robot_qpos=observed,
        root_position_error_m=root_error,
        joint_error_rad=joint_error,
        root_orientation_error_rad=orientation_error,
        usd_link_position_error_m=usd_link_error,
        physics_link_position_error_m=physics_link_error,
        hand_review_color_pixels=color_pixels,
    )
    if (
        np.nanmax(root_error) > 1e-4
        or np.nanmax(joint_error) > 1e-4
        or np.nanmax(orientation_error) > 1e-4
    ):
        raise ValueError("Simulator state differs from the frozen kinematic targets")
    if np.nanmax(usd_link_error) > 1e-4:
        raise ValueError(
            "Rendered USD link transforms do not match the retargeted hand geometry"
        )
    if np.nanmax(physics_link_error) > 1e-4:
        raise ValueError(
            "Simulator link transforms do not match the retargeted hand geometry"
        )
    if any(color_pixels[valid[:, side_id], side_id].max() < 20 for side_id in range(2)):
        raise ValueError(
            "A supported robot hand never appears with its assigned review color"
        )
    report = {
        "status": "complete kinematic renderer replay; full visual acceptance pending",
        "source_frames": len(frames),
        "decoded_video_frames": decoded_frames,
        "fps": encoded_fps,
        "bundle_manifest_sha256": verified["manifest_sha256"],
        "retarget_payload_sha256": contract["payload"]["sha256"],
        "gate_manifest_sha256": digest(args.gate / "manifest.json"),
        "video_sha256": digest(video),
        "collision_model_mode": gate_manifest.get(
            "collision_model_mode", "original_capsules_with_object_hull_exclusions"
        ),
        "fully_checked_derived_model_accepted_frames": gate_manifest.get(
            "fully_checked_derived_model_accepted_frames", 0
        ),
        "readback_sha256": digest(args.out / "simulator-readback.npz"),
        "source_sha256": digest(Path(__file__)),
        "world_to_stage": WORLD_TO_STAGE.tolist(),
        "world_to_stage_is_measured_gravity": False,
        "camera_eye": eye.tolist(),
        "camera_target": target.tolist(),
        "hand_assets": hand_assets,
        "maximum_root_position_error_m": float(np.nanmax(root_error)),
        "maximum_joint_error_rad": float(np.nanmax(joint_error)),
        "maximum_root_orientation_error_rad": float(np.nanmax(orientation_error)),
        "maximum_rendered_usd_link_position_error_m": float(np.nanmax(usd_link_error)),
        "maximum_physics_link_position_error_m": float(np.nanmax(physics_link_error)),
        "per_hand_maximum_review_color_pixels": color_pixels.max(axis=0).tolist(),
        "runtime": {
            name: importlib.metadata.version(name)
            for name in (
                "isaaclab",
                "isaacsim",
                "torch",
                "numpy",
                "scipy",
                "imageio",
                "imageio-ffmpeg",
            )
        },
        "isaaclab_release": lab_release,
        "isaaclab_release_file_sha256": digest(lab_release_file),
        "simulation_device": sim.device,
        "gravity_enabled": False,
        "collision_resolution_enabled": False,
        "render_transform_sync": "CPU PhysX link FK refreshed and explicitly written to USD; USD scene delegate",
        "review_colors": {
            "left": "blue",
            "right": "orange",
            "meaning": "side identity only; original material appearance not reconstructed",
        },
        "batch_shutdown": "supported SimulationApp.close(skip_cleanup=True) after synchronous artifact completion",
        "learned_policy_rollout": False,
        "visual_review": "pending",
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json(args.out / "complete.json", report)
    print(json.dumps(report, indent=2), flush=True)


try:
    main()
except Exception as error:
    write_json(
        args.out / "failure.json",
        {
            "status": "failed; no accepted replay receipt",
            "error": str(error),
            "exception": type(error).__name__,
            "traceback": traceback.format_exc(),
            "source_sha256": digest(Path(__file__)),
        },
    )
    print(traceback.format_exc(), flush=True)
    raise
finally:
    # All output writes and decode validation finish before this supported
    # batch exit. Normal close stalls in the unused Replicator orchestration
    # on this local camera experience, including when wait is disabled.
    app.close(skip_cleanup=True)
