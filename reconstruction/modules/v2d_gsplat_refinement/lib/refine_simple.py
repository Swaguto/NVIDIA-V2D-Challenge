# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal 2DGS hand/object/camera refinement.

This is intentionally much smaller than ``refine.py``. It optimizes only:
  - static Gaussian appearance params: scale, color, opacity;
  - per-frame root pose of each object (one object via the single-object
    flags, or several via ``--objects_root``);
  - optional global scale per object;
  - per-frame hand wrist/root pose (MANO global_orient + cam_t only);
  - per-frame background/camera pose for the static depth-initialized scene.

It does not optimize Gaussian offsets or Gaussian orientation. MANO finger
pose, MANO shape, hand scale, and object scale are optional via their LR flags.
Image losses are full-frame
L1 photometric error plus optional L1 segmentation-mask supervision, with
optional relative depth-gradient supervision from MoGe depth, optional
absolute supervision from metric (e.g. stereo) depth, and with optional
temporal smoothness on the absolute and hand/object-relative dynamic poses,
measured in the camera frame or (``--smooth_in_world``) the world frame.

Run inside the gsplat refinement environment, for example:

    python -m v2d.gsplat_refinement.lib.refine_simple \
        --frames_dir data/outputs/clip/frames \
        --depth_dir data/outputs/clip/depth \
        --intrinsics_path data/outputs/clip/intrinsics_stable.json \
        --object_mesh_path data/outputs/clip/mesh_scaled.obj \
        --object_poses_dir data/outputs/clip/poses \
        --object_mask_dir data/outputs/clip/masks/1 \
        --right_hand_pose_dir data/outputs/clip/hamer_aligned_filled/2 \
        --right_hand_mask_dir data/outputs/clip/masks/2 \
        --mano_assets_root data/weights/hamer/_DATA/data \
        --refined_object_poses_dir data/outputs/clip/poses_refined_simple \
        --refined_right_hand_pose_dir data/outputs/clip/hamer_refined_simple/2 \
        --refined_camera_poses_dir data/outputs/clip/camera_refined_simple \
        --overlay_path data/outputs/clip/refined_simple.mp4
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from v2d.common.datatypes import Transform3d

from .background import (
    BackgroundGaussians,
    BackgroundPoseField,
    init_background_from_depth,
    init_background_multiframe,
)
from .constraints import PoseTrustRegion
from .depth import load_inverse_depth
from .gaussians import (
    GaussianFrame,
    axis_angle_to_quat,
    concat_frames,
    init_hand_face_gaussians,
    init_object_face_gaussians_from_mesh,
    quat_mul,
)
from .io import (
    HandPoseTrack,
    ObjectPoseTrack,
    load_hand_poses,
    load_intrinsics,
    load_object_mesh,
    load_object_poses,
    save_hand_poses,
    save_object_poses,
)
from .losses import (
    depth_gradient_loss,
    depth_loss,
    quat_acceleration,
    quat_smoothness,
    rotation_smoothness,
    temporal_smoothness,
    temporal_acceleration,
)
from .pose_fields import HandPoseField, ObjectPoseField, _quat_to_axis_angle
from .render_features import pad_render_features

_MESH_SUFFIXES = (".glb", ".gltf", ".obj", ".ply")


@dataclass
class HandSlot:
    side: str
    track: HandPoseTrack
    pose_field: HandPoseField
    gaussians: torch.nn.Module
    mask_dir: str
    output_pose_dir: str | None
    frame_to_pos: dict[int, int]


@dataclass
class ObjectSlot:
    name: str
    pose_field: ObjectPoseField
    gaussians: torch.nn.Module
    verts: torch.Tensor
    faces: np.ndarray
    output_pose_dir: str | None
    sdf: "ObjectSDFGrid | None" = None


@dataclass
class ObjectSpec:
    name: str
    mesh_path: str
    poses_dir: str
    mask_dir: str
    output_pose_dir: str | None


def _object_specs(
    object_mesh_path: str | None,
    object_poses_dir: str | None,
    object_mask_dir: str | None,
    refined_object_poses_dir: str | None,
    objects_root: str | None,
    refined_objects_root: str | None,
) -> list[ObjectSpec]:
    """Resolve the single-object flags and/or a multi-object root.

    ``objects_root/<name>/`` holds one mesh file named ``mesh.<ext>`` plus
    ``poses/`` (object→camera Transform3d JSONs) and ``masks/``. Refined poses
    go to ``refined_objects_root/<name>/``.
    """
    specs: list[ObjectSpec] = []
    single = (object_mesh_path, object_poses_dir, object_mask_dir)
    if any(v is not None for v in single):
        if any(v is None for v in single):
            raise ValueError(
                "object_mesh_path, object_poses_dir and object_mask_dir must be given together"
            )
        specs.append(ObjectSpec(
            "object", object_mesh_path, object_poses_dir, object_mask_dir, refined_object_poses_dir,
        ))
    if objects_root is not None:
        for name in sorted(os.listdir(objects_root)):
            obj_dir = os.path.join(objects_root, name)
            if not os.path.isdir(obj_dir):
                continue
            meshes = [
                os.path.join(obj_dir, f"mesh{ext}") for ext in _MESH_SUFFIXES
                if os.path.exists(os.path.join(obj_dir, f"mesh{ext}"))
            ]
            if len(meshes) != 1:
                raise FileNotFoundError(f"Expected exactly one mesh.<ext> in {obj_dir}, found {meshes}")
            out_dir = os.path.join(refined_objects_root, name) if refined_objects_root else None
            specs.append(ObjectSpec(
                name, meshes[0], os.path.join(obj_dir, "poses"), os.path.join(obj_dir, "masks"), out_dir,
            ))
    if not specs:
        raise ValueError("No objects: pass object_mesh_path/object_poses_dir/object_mask_dir or objects_root")
    if len({s.name for s in specs}) != len(specs):
        raise ValueError(f"Duplicate object names: {[s.name for s in specs]}")
    return specs


class SimpleFrameCache:
    """CPU cache for the small set of supervision this script uses.

    ``obj_mask_known[k][t]`` is 0 when object ``k`` has no mask file for frame
    ``t`` and ``missing_object_mask_unknown`` is set, so the segmentation loss
    skips that channel instead of treating the object as absent.
    """

    def __init__(
        self,
        frame_indices: list[int],
        frames_dir: str,
        depth_dir: str | None,
        object_mask_dirs: list[str],
        hand_mask_dirs: list[str],
        height: int,
        width: int,
        valid_mask_threshold: float = 0.04,
        valid_mask_erode_iters: int = 2,
        missing_object_mask_unknown: bool = False,
    ) -> None:
        self.frame_indices = list(frame_indices)
        self.height = int(height)
        self.width = int(width)
        n = len(frame_indices)
        self.rgb = torch.zeros((n, height, width, 3), dtype=torch.float32)
        self.depth = (
            torch.empty((n, height, width), dtype=torch.float32)
            if depth_dir is not None else None
        )
        self.obj_masks = [
            torch.zeros((n, height, width), dtype=torch.float32)
            for _ in object_mask_dirs
        ]
        self.obj_mask_known = [torch.ones((n,), dtype=torch.float32) for _ in object_mask_dirs]
        self.hand_masks = [
            torch.zeros((n, height, width), dtype=torch.float32)
            for _ in hand_mask_dirs
        ]

        for t, fidx in enumerate(tqdm(frame_indices, ncols=80, desc="caching", unit="frame")):
            self.rgb[t] = _load_rgb_resized(_find_frame_file(frames_dir, fidx), width, height)
            if self.depth is not None and depth_dir is not None:
                self.depth[t] = _load_depth_resized(os.path.join(depth_dir, f"{fidx:06d}.png"), width, height)
            for k, mask_dir in enumerate(object_mask_dirs):
                self.obj_masks[k][t] = _load_mask_or_zero(mask_dir, fidx, width, height)
                if missing_object_mask_unknown and not os.path.exists(
                    os.path.join(mask_dir, f"{fidx:06d}.png")
                ):
                    self.obj_mask_known[k][t] = 0.0
            for k, mask_dir in enumerate(hand_mask_dirs):
                self.hand_masks[k][t] = _load_mask_or_zero(mask_dir, fidx, width, height)

        if valid_mask_threshold <= 0:
            valid = torch.ones((height, width), dtype=torch.float32)
        else:
            max_brightness = self.rgb.amax(dim=0).amax(dim=-1)
            valid = (max_brightness > float(valid_mask_threshold)).float()
            if valid_mask_erode_iters > 0:
                m = valid.unsqueeze(0).unsqueeze(0)
                for _ in range(int(valid_mask_erode_iters)):
                    m = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
                valid = m.squeeze(0).squeeze(0)
        self.valid_mask = valid

        try:
            self.rgb = self.rgb.pin_memory()
            if self.depth is not None:
                self.depth = self.depth.pin_memory()
            self.obj_masks = [m.pin_memory() for m in self.obj_masks]
            self.hand_masks = [m.pin_memory() for m in self.hand_masks]
            self.valid_mask = self.valid_mask.pin_memory()
        except RuntimeError:
            pass

    def union_mask(self, t: int, device: torch.device | str) -> torch.Tensor:
        m = torch.zeros((self.height, self.width), dtype=torch.float32, device=device)
        for mask in self.obj_masks + self.hand_masks:
            m = torch.maximum(m, mask[t].to(device, non_blocking=True))
        return m


def _target_rgb(
    cache: SimpleFrameCache,
    t: int,
    device: torch.device | str,
    mask_background: bool,
) -> torch.Tensor:
    target = cache.rgb[t].to(device, non_blocking=True)
    if mask_background:
        target = target * cache.union_mask(t, device).unsqueeze(-1)
    return target


def _find_frame_file(frames_dir: str, fidx: int) -> str:
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(frames_dir, f"{fidx:06d}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Frame {fidx:06d} not found in {frames_dir}")


def _list_frame_indices(frames_dir: str) -> set[int]:
    out: set[int] = set()
    for path in glob.glob(os.path.join(frames_dir, "*.png")) + glob.glob(os.path.join(frames_dir, "*.jpg")):
        try:
            out.add(int(os.path.splitext(os.path.basename(path))[0]))
        except ValueError:
            continue
    return out


def _load_rgb_resized(path: str, width: int, height: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _load_depth_resized(path: str, width: int, height: int) -> torch.Tensor:
    return torch.from_numpy(load_inverse_depth(path, width, height))


def _load_mask_or_zero(mask_dir: str, fidx: int, width: int, height: int) -> torch.Tensor:
    path = os.path.join(mask_dir, f"{fidx:06d}.png")
    if not os.path.exists(path):
        return torch.zeros((height, width), dtype=torch.float32)
    img = Image.open(path)
    if img.size != (width, height):
        img = img.resize((width, height), Image.NEAREST)
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.0:
        arr = arr / 255.0
    return torch.from_numpy((arr > 0.5).astype(np.float32))


def _restrict_object_track(track: ObjectPoseTrack, keep: set[int]) -> ObjectPoseTrack:
    idxs = sorted(
        (i for i, fidx in enumerate(track.frame_indices) if fidx in keep),
        key=lambda i: track.frame_indices[i],
    )
    if not idxs:
        raise RuntimeError("No object pose frames overlap the available RGB frames.")
    return ObjectPoseTrack(
        rotations=track.rotations[idxs],
        translations=track.translations[idxs],
        scales=track.scales[idxs],
        frame_indices=[track.frame_indices[i] for i in idxs],
    )


def _restrict_hand_track(track: HandPoseTrack, keep: set[int]) -> HandPoseTrack:
    idxs = [i for i, fidx in enumerate(track.frame_indices) if fidx in keep]
    if not idxs:
        raise RuntimeError("A hand pose track has no frames overlapping the training frames.")
    return HandPoseTrack(
        global_orient=track.global_orient[idxs],
        hand_pose=track.hand_pose[idxs],
        betas=track.betas[idxs],
        cam_t=track.cam_t[idxs],
        is_right=track.is_right,
        frame_indices=[track.frame_indices[i] for i in idxs],
        raw_records=[track.raw_records[i] for i in idxs],
        hand_scale=track.hand_scale,
    )


def _freeze_all(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def _enable(module: torch.nn.Module, names: list[str]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, torch.nn.Parameter):
            value.requires_grad_(True)
            params.append(value)
    return params


def _trainable_gaussian_params(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    _freeze_all(module)
    return _enable(module, ["_log_scale", "_opacity_logit", "_color"])


def _load_relative_w2c_poses(
    poses_dir: str,
    frame_indices: list[int],
    ref_t: int,
) -> dict[int, np.ndarray]:
    ref_fidx = frame_indices[ref_t]
    ref_path = os.path.join(poses_dir, f"{ref_fidx:06d}.json")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"background_pose_init_dir is set but reference pose {ref_path} is missing."
        )
    M_ref = Transform3d.load(ref_path).to_matrix()
    out: dict[int, np.ndarray] = {ref_t: np.eye(4)}
    for t, fidx in enumerate(frame_indices):
        if t == ref_t:
            continue
        path = os.path.join(poses_dir, f"{fidx:06d}.json")
        if not os.path.exists(path):
            continue
        tf = Transform3d.load(path)
        if any(abs(s - 1.0) > 1e-3 for s in tf.scale):
            continue
        out[t] = np.linalg.inv(tf.to_matrix()) @ M_ref
    return out


def _seed_background_pose_field(
    bg_pose_field: BackgroundPoseField,
    poses: dict[int, np.ndarray],
) -> tuple[int, int]:
    device = bg_pose_field.axis_angle.device
    aa = bg_pose_field.axis_angle.detach().clone()
    tr = bg_pose_field.translation.detach().clone()
    loaded = missing = 0
    for t in range(aa.shape[0]):
        if t not in poses:
            missing += 1
            continue
        M = poses[t]
        R = torch.from_numpy(M[:3, :3]).to(device=device, dtype=torch.float32)
        q = _rotmat_to_quat(R)
        aa[t] = _quat_to_axis_angle(q)
        tr[t] = torch.from_numpy(M[:3, 3]).to(device=device, dtype=torch.float32)
        loaded += 1
    with torch.no_grad():
        bg_pose_field.axis_angle.copy_(aa)
        bg_pose_field.translation.copy_(tr)
    return loaded, missing


def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    import roma

    q_xyzw = roma.rotmat_to_unitquat(R)
    return torch.cat([q_xyzw[..., 3:4], q_xyzw[..., :3]], dim=-1)


def _anchor_camera_reference(bg_pose_field: BackgroundPoseField) -> None:
    ref_t = int(bg_pose_field.ref_frame_t)
    with torch.no_grad():
        bg_pose_field.axis_angle[ref_t].zero_()
        bg_pose_field.translation[ref_t].zero_()


@dataclass
class ObjectSDFGrid:
    """Differentiable object-space SDF grid.

    ``values`` stores signed distance with positive outside the object and
    negative inside. Trimesh uses the opposite sign convention, so construction
    flips the sign once. Querying uses ``grid_sample`` so gradients flow to the
    queried points, hence to hand/object pose and scale parameters.
    """

    values: torch.Tensor      # (1, 1, D, H, W), D=z, H=y, W=x
    bounds_min: torch.Tensor  # (3,) object-frame xyz
    bounds_max: torch.Tensor  # (3,) object-frame xyz

    def query(self, points_obj: torch.Tensor) -> torch.Tensor:
        if points_obj.numel() == 0:
            return points_obj.new_zeros((0,))
        span = (self.bounds_max - self.bounds_min).clamp_min(1e-8)
        coords = 2.0 * (points_obj - self.bounds_min) / span - 1.0
        grid = coords.view(1, -1, 1, 1, 3)
        sampled = F.grid_sample(
            self.values.to(device=points_obj.device, dtype=points_obj.dtype),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.view(-1)


def _build_object_sdf_grid(
    vertices: torch.Tensor,
    faces: np.ndarray,
    resolution: int,
    margin: float,
    device: torch.device,
) -> ObjectSDFGrid:
    res = int(resolution)
    if res < 8:
        raise ValueError("object_sdf_resolution must be >= 8")
    verts_np = vertices.detach().cpu().numpy().astype(np.float64)
    faces_np = np.asarray(faces, dtype=np.int64)
    try:
        import trimesh
        from trimesh.proximity import signed_distance
    except ImportError as exc:
        raise RuntimeError(
            "Hand/object penetration loss requires trimesh and its proximity dependencies."
        ) from exc

    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    if not bool(getattr(mesh, "is_watertight", False)):
        print("Warning: object mesh is not watertight; hand/object SDF signs may be approximate.")

    bounds_min_np = verts_np.min(axis=0)
    bounds_max_np = verts_np.max(axis=0)
    extent = bounds_max_np - bounds_min_np
    pad = max(float(margin) * 4.0, float(extent.max()) * 0.05, 1e-3)
    bounds_min_np = bounds_min_np - pad
    bounds_max_np = bounds_max_np + pad

    xs = np.linspace(bounds_min_np[0], bounds_max_np[0], res, dtype=np.float32)
    ys = np.linspace(bounds_min_np[1], bounds_max_np[1], res, dtype=np.float32)
    zs = np.linspace(bounds_min_np[2], bounds_max_np[2], res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=-1)

    sdf = np.empty((points.shape[0],), dtype=np.float32)
    chunk = 200_000
    print(f"Building object SDF grid: resolution={res}, points={points.shape[0]}, padding={pad:.4g}")
    try:
        for start in tqdm(range(0, points.shape[0], chunk), ncols=80, desc="object_sdf", unit="chunk"):
            end = min(start + chunk, points.shape[0])
            # trimesh signed_distance is positive inside, negative outside.
            sdf[start:end] = -signed_distance(mesh, points[start:end]).astype(np.float32)
    except BaseException as exc:
        raise RuntimeError(
            "Failed to build object SDF grid for hand/object penetration. "
            "This usually means trimesh's optional rtree dependency is missing "
            "or the object mesh is invalid."
        ) from exc

    sdf_xyz = torch.from_numpy(sdf.reshape(res, res, res))
    values = sdf_xyz.permute(2, 1, 0).contiguous().view(1, 1, res, res, res).to(device)
    bounds_min = torch.tensor(bounds_min_np, dtype=torch.float32, device=device)
    bounds_max = torch.tensor(bounds_max_np, dtype=torch.float32, device=device)
    return ObjectSDFGrid(values=values, bounds_min=bounds_min, bounds_max=bounds_max)


def _hand_object_penetration_loss(
    t: int,
    fidx: int,
    obj_slot: ObjectSlot,
    hand_slots: list[HandSlot],
    margin: float,
    max_hand_verts: int,
) -> torch.Tensor:
    obj_gaussians = obj_slot.gaussians
    object_sdf = obj_slot.sdf
    R_obj, t_obj = obj_slot.pose_field(t)
    if object_sdf is None:
        return R_obj.sum() * 0.0
    if hasattr(obj_gaussians, "object_scale"):
        s_obj = obj_gaussians.object_scale().clamp_min(1e-6)
    else:
        s_obj = R_obj.new_ones(())
    losses: list[torch.Tensor] = []
    for slot in hand_slots:
        local_t = slot.frame_to_pos.get(fidx)
        if local_t is None:
            continue
        verts_cam, _ = slot.pose_field.posed_verts_and_rotmats_camera(local_t)
        if max_hand_verts > 0 and verts_cam.shape[0] > int(max_hand_verts):
            idx = torch.linspace(
                0,
                verts_cam.shape[0] - 1,
                int(max_hand_verts),
                device=verts_cam.device,
            ).round().to(torch.long)
            verts_cam = verts_cam[idx]
        verts_obj = ((verts_cam - t_obj) @ R_obj) / s_obj
        sdf_metric = object_sdf.query(verts_obj) * s_obj
        penetration = (float(margin) - sdf_metric).clamp_min(0.0)
        losses.append(penetration.square().mean())
    if not losses:
        return R_obj.sum() * 0.0
    return torch.stack(losses).mean()


class VGG16PerceptualLoss(torch.nn.Module):
    """VGG16 feature-space L1 loss for RGB images in [0, 1]."""

    def __init__(
        self,
        weights_path: str | None = None,
        resize: int = 224,
        layer_indices: tuple[int, ...] = (3, 8, 15, 22),
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights, vgg16
        except ImportError as exc:
            raise RuntimeError(
                "VGG perceptual loss requires torchvision in the gsplat refinement environment."
            ) from exc

        self.layer_indices = set(int(i) for i in layer_indices)
        self.resize = int(resize)
        if weights_path is not None:
            weights_path = os.path.abspath(os.path.expanduser(weights_path))
            model = vgg16(weights=None)
            if not os.path.exists(weights_path):
                weights = VGG16_Weights.IMAGENET1K_V1
                os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
                print(f"Downloading VGG16 ImageNet weights -> {weights_path}")
                try:
                    state = torch.hub.load_state_dict_from_url(
                        weights.url,
                        model_dir=os.path.dirname(weights_path) or ".",
                        file_name=os.path.basename(weights_path),
                        map_location="cpu",
                        progress=True,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to download torchvision VGG16 ImageNet weights to {weights_path}."
                    ) from exc
            else:
                state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
        else:
            try:
                model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load torchvision VGG16 ImageNet weights. "
                    "Provide --vgg_weights_path with a local vgg16 state_dict, "
                    "or ensure torchvision's weight cache/network access is available."
                ) from exc
        self.features = model.features[: max(self.layer_indices) + 1].eval()
        for param in self.features.parameters():
            param.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def _preprocess(
        self,
        image: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = image.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0)
        if valid_mask is not None:
            x = x * valid_mask.to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        if self.resize > 0 and (x.shape[-2] != self.resize or x.shape[-1] != self.resize):
            x = F.interpolate(x, size=(self.resize, self.resize), mode="bilinear", align_corners=False)
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                outputs.append(x)
        return outputs

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_features = self._features(self._preprocess(pred_rgb, valid_mask))
        with torch.no_grad():
            target_features = self._features(self._preprocess(target_rgb, valid_mask))
        loss = pred_rgb.new_zeros(())
        for pred_f, target_f in zip(pred_features, target_features):
            loss = loss + F.l1_loss(pred_f, target_f)
        return loss / float(max(len(pred_features), 1))


def _render_2dgs_features(
    frame: GaussianFrame,
    K: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    colors: torch.Tensor | None = None,
    clamp_colors: bool = True,
) -> torch.Tensor:
    try:
        from gsplat.rendering import rasterization_2dgs
    except ImportError as exc:
        raise RuntimeError(
            "refine_simple.py requires gsplat.rendering.rasterization_2dgs. "
            "Install/run in the gsplat refinement Docker image with gsplat>=1.4."
        ) from exc

    device = frame.means.device
    render_colors = frame.colors if colors is None else colors
    if clamp_colors:
        render_colors = render_colors.clamp(0.0, 1.0)
    n_channels = int(render_colors.shape[-1])
    render_colors = pad_render_features(render_colors)
    padded_channels = int(render_colors.shape[-1])
    viewmats = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    kwargs = dict(
        means=frame.means,
        quats=frame.quats,
        scales=frame.scales,
        opacities=frame.opacities,
        colors=render_colors,
        viewmats=viewmats,
        Ks=K.unsqueeze(0),
        width=int(width),
        height=int(height),
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        render_mode="RGB",
        backgrounds=torch.zeros(1, padded_channels, device=device, dtype=torch.float32),
    )
    try:
        result = rasterization_2dgs(**kwargs)
    except TypeError:
        # Some gsplat builds expose a narrower 2DGS signature. Retry with the
        # universally required raster inputs before surfacing the error.
        kwargs.pop("render_mode", None)
        result = rasterization_2dgs(**kwargs)

    colors = result[0] if isinstance(result, tuple) else result
    if colors.ndim == 4:
        colors = colors[0]
    return colors[..., :n_channels]


def _render_2dgs_rgb(
    frame: GaussianFrame,
    K: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> torch.Tensor:
    colors = _render_2dgs_features(frame, K, width, height, near_plane, far_plane)
    return colors[..., :3]


def _render_2dgs_depth(
    frame: GaussianFrame,
    K: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth_feature = frame.means[:, 2:3].clamp_min(float(near_plane))
    alpha_feature = torch.ones_like(depth_feature)
    rendered = _render_2dgs_features(
        frame,
        K,
        width,
        height,
        near_plane=near_plane,
        far_plane=far_plane,
        colors=torch.cat([depth_feature, alpha_feature], dim=-1),
        clamp_colors=False,
    )
    depth_accum = rendered[..., 0]
    alpha = rendered[..., 1].clamp(0.0, 1.0)
    depth = depth_accum / alpha.clamp_min(1e-4)
    return depth, alpha


def _current_frame(
    t: int,
    fidx: int,
    obj_slots: list[ObjectSlot],
    hand_slots: list[HandSlot],
    bg_gaussians: BackgroundGaussians | None,
    bg_pose_field: BackgroundPoseField | None,
) -> GaussianFrame:
    frames = []
    for obj in obj_slots:
        R_obj, t_obj = obj.pose_field(t)
        frames.append(obj.gaussians(R_obj, t_obj))
    for slot in hand_slots:
        local_t = slot.frame_to_pos.get(fidx)
        if local_t is None:
            continue
        verts, Rv = slot.pose_field.posed_verts_and_rotmats_camera(local_t)
        frames.append(slot.gaussians(verts, Rv))
    if bg_gaussians is not None and bg_pose_field is not None:
        R_bg, t_bg = bg_pose_field(t)
        frames.append(bg_gaussians(R_bg, t_bg))
    return concat_frames(frames)


def _label_block(
    n: int,
    n_classes: int,
    class_idx: int | None,
    device: torch.device,
) -> torch.Tensor:
    labels = torch.zeros((int(n), int(n_classes)), device=device, dtype=torch.float32)
    if class_idx is not None and n > 0:
        labels[:, int(class_idx)] = 1.0
    return labels


def _current_frame_with_labels(
    t: int,
    fidx: int,
    obj_slots: list[ObjectSlot],
    hand_slots: list[HandSlot],
    bg_gaussians: BackgroundGaussians | None,
    bg_pose_field: BackgroundPoseField | None,
    n_classes: int,
) -> tuple[GaussianFrame, torch.Tensor]:
    """Class channels are objects ``0..M-1`` followed by hands ``M..M+H-1``."""
    frames = []
    labels = []
    for obj_i, obj in enumerate(obj_slots):
        R_obj, t_obj = obj.pose_field(t)
        obj_frame = obj.gaussians(R_obj, t_obj)
        frames.append(obj_frame)
        labels.append(_label_block(obj_frame.means.shape[0], n_classes, obj_i, obj_frame.means.device))

    n_obj = len(obj_slots)
    for slot_i, slot in enumerate(hand_slots):
        local_t = slot.frame_to_pos.get(fidx)
        if local_t is None:
            continue
        verts, Rv = slot.pose_field.posed_verts_and_rotmats_camera(local_t)
        hand_frame = slot.gaussians(verts, Rv)
        frames.append(hand_frame)
        labels.append(_label_block(
            hand_frame.means.shape[0], n_classes, n_obj + slot_i, hand_frame.means.device
        ))

    if bg_gaussians is not None and bg_pose_field is not None:
        R_bg, t_bg = bg_pose_field(t)
        bg_frame = bg_gaussians(R_bg, t_bg)
        frames.append(bg_frame)
        labels.append(_label_block(bg_frame.means.shape[0], n_classes, None, bg_frame.means.device))

    return concat_frames(frames), torch.cat(labels, dim=0).contiguous()


def _segmentation_mask_loss(
    rendered_class: torch.Tensor,
    cache: SimpleFrameCache,
    t: int,
    valid_mask: torch.Tensor,
    valid_denom: torch.Tensor,
) -> torch.Tensor:
    n_classes = int(rendered_class.shape[-1])
    n_obj = len(cache.obj_masks)
    target = rendered_class.new_zeros(rendered_class.shape)
    channel_known = rendered_class.new_ones((n_classes,))
    for obj_i, obj_mask in enumerate(cache.obj_masks):
        target[..., obj_i] = obj_mask[t].to(rendered_class.device, non_blocking=True)
        channel_known[obj_i] = float(cache.obj_mask_known[obj_i][t])
    for slot_i, hand_mask in enumerate(cache.hand_masks):
        if n_obj + slot_i >= n_classes:
            break
        target[..., n_obj + slot_i] = hand_mask[t].to(rendered_class.device, non_blocking=True)
    diff = (rendered_class - target).abs() * channel_known
    return (diff * valid_mask.unsqueeze(-1)).sum() / (
        valid_denom * float(max(n_classes, 1))
    )


def _normalized_smooth(loss: torch.Tensor, n: int) -> torch.Tensor:
    return loss / float(max(n - 1, 1))


def _observed_smooth(fn, values: torch.Tensor, frames: list[int]) -> torch.Tensor:
    """Only consecutive observed pairs contribute; reacquisition gaps stay unknown."""
    pairs = sum(b == a + 1 for a, b in zip(frames, frames[1:]))
    return fn(values, frames) / float(max(pairs, 1))


def _cosine_lr_scale(step: int, total_steps: int, min_factor: float) -> float:
    if total_steps <= 1:
        return float(min_factor)
    progress = min(max(int(step), 0), int(total_steps) - 1) / float(int(total_steps) - 1)
    return float(min_factor) + 0.5 * (1.0 - float(min_factor)) * (1.0 + math.cos(math.pi * progress))


def _first_nonfinite_tensor(
    tensors: list[tuple[str, torch.Tensor | None]],
) -> str | None:
    """Return the first named tensor containing NaN or Inf, if any."""
    for name, tensor in tensors:
        if tensor is not None and not bool(torch.isfinite(tensor).all()):
            return name
    return None


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _hand_root_quats_camera(slot: HandSlot, t_idx: torch.Tensor) -> torch.Tensor:
    q_hand = axis_angle_to_quat(slot.pose_field.global_orient[t_idx])
    if not slot.pose_field.is_right:
        q_hand = torch.cat([q_hand[..., :2], -q_hand[..., 2:]], dim=-1)
    return q_hand


def _hand_object_relative_smoothness(
    obj_pose_field: ObjectPoseField,
    slot: HandSlot,
    w_smooth_hand_object_relative_rot: float,
    w_smooth_hand_object_relative_trans: float,
) -> torch.Tensor:
    loss = obj_pose_field.axis_angle.new_zeros(())
    if (
        float(w_smooth_hand_object_relative_rot) <= 0.0
        and float(w_smooth_hand_object_relative_trans) <= 0.0
    ):
        return loss

    object_frame_to_pos = {fidx: i for i, fidx in enumerate(obj_pose_field.frame_indices)}
    paired: list[tuple[int, int]] = []
    for hand_t, fidx in enumerate(slot.pose_field.frame_indices):
        obj_t = object_frame_to_pos.get(fidx)
        if obj_t is not None:
            paired.append((hand_t, obj_t))
    if len(paired) < 2:
        return loss

    device = obj_pose_field.axis_angle.device
    hand_idx = torch.tensor([h for h, _ in paired], device=device, dtype=torch.long)
    obj_idx = torch.tensor([o for _, o in paired], device=device, dtype=torch.long)
    paired_frames = [slot.pose_field.frame_indices[h] for h, _ in paired]

    R_obj, t_obj = obj_pose_field.batched_forward(obj_idx)
    q_obj = axis_angle_to_quat(obj_pose_field.axis_angle[obj_idx])
    q_hand = _hand_root_quats_camera(slot, hand_idx)
    t_hand = slot.pose_field.cam_t[hand_idx]

    if float(w_smooth_hand_object_relative_rot) > 0.0:
        q_rel = quat_mul(_quat_conj(q_obj), q_hand)
        loss = loss + float(w_smooth_hand_object_relative_rot) * _normalized_smooth(
            quat_smoothness(q_rel, paired_frames),
            len(paired),
        )
    if float(w_smooth_hand_object_relative_trans) > 0.0:
        # Express hand root translation in the object frame, removing common
        # camera-frame motion before applying temporal smoothness.
        t_rel = torch.matmul((t_hand - t_obj).unsqueeze(1), R_obj).squeeze(1)
        loss = loss + float(w_smooth_hand_object_relative_trans) * _normalized_smooth(
            temporal_smoothness(t_rel, paired_frames),
            len(paired),
        )
    return loss


def _camera_pose_to_world(
    bg_pose_field: BackgroundPoseField,
    t_idx: torch.Tensor,
    q_cam: torch.Tensor,
    t_cam: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map camera-frame poses (B,4)/(B,3) to the background world frame."""
    R_w2c, t_w2c = bg_pose_field.batched_forward(t_idx)
    q_w2c = axis_angle_to_quat(bg_pose_field.axis_angle[t_idx])
    q_world = quat_mul(_quat_conj(q_w2c), q_cam)
    # R_w2c^T (t_cam - t_w2c), written in row-vector form.
    t_world = torch.matmul((t_cam - t_w2c).unsqueeze(1), R_w2c).squeeze(1)
    return q_world, t_world


def _pose_smoothness(
    obj_slots: list[ObjectSlot],
    hand_slots: list[HandSlot],
    bg_pose_field: BackgroundPoseField | None,
    w_smooth_object_rot: float,
    w_smooth_object_trans: float,
    w_smooth_hand_rot: float,
    w_smooth_hand_articulation: float,
    w_smooth_hand_trans: float,
    w_smooth_hand_object_relative_rot: float,
    w_smooth_hand_object_relative_trans: float,
    w_smooth_camera_rot: float,
    w_smooth_camera_trans: float,
    smooth_in_world: bool = False,
    frame_to_t: dict[int, int] | None = None,
    w_camera_acceleration_trans: float = 0.0,
    w_camera_acceleration_rot: float = 0.0,
) -> torch.Tensor:
    """Temporal smoothness on all dynamic poses.

    With ``smooth_in_world`` (and a background pose field), object and hand
    root poses are smoothed after mapping into the background world frame.
    On a head-mounted camera this stops the camera-frame smoothness terms from
    penalizing the wearer's own head motion, and couples them to the camera
    trajectory so all three are cleaned jointly.
    """
    loss = obj_slots[0].pose_field.axis_angle.new_zeros(())
    in_world = bool(smooth_in_world) and bg_pose_field is not None
    for obj in obj_slots:
        pf = obj.pose_field
        T = pf.num_frames()
        if in_world:
            idx = torch.arange(T, device=pf.axis_angle.device)
            q_w, t_w = _camera_pose_to_world(
                bg_pose_field, idx, axis_angle_to_quat(pf.axis_angle), pf.translation,
            )
            rot_term = _observed_smooth(quat_smoothness, q_w, pf.frame_indices)
            trans_term = _observed_smooth(temporal_smoothness, t_w, pf.frame_indices)
        else:
            rot_term = _observed_smooth(rotation_smoothness, pf.axis_angle, pf.frame_indices)
            trans_term = _observed_smooth(temporal_smoothness, pf.translation, pf.frame_indices)
        loss = loss + float(w_smooth_object_rot) * rot_term
        loss = loss + float(w_smooth_object_trans) * trans_term
    if bg_pose_field is not None:
        T = bg_pose_field.num_frames()
        loss = loss + float(w_smooth_camera_rot) * _normalized_smooth(
            rotation_smoothness(bg_pose_field.axis_angle), T
        )
        loss = loss + float(w_smooth_camera_trans) * _normalized_smooth(
            temporal_smoothness(bg_pose_field.translation), T
        )
        if w_camera_acceleration_trans > 0 or w_camera_acceleration_rot > 0:
            # The camera centre moves in world coordinates. Penalizing the
            # world-to-camera translation instead couples jitter to orientation.
            idx = torch.arange(T, device=bg_pose_field.axis_angle.device)
            R_w2c, t_w2c = bg_pose_field.batched_forward(idx)
            centre = -torch.matmul(t_w2c.unsqueeze(1), R_w2c).squeeze(1)
            loss = loss + float(w_camera_acceleration_trans) * _normalized_smooth(
                temporal_acceleration(centre), T
            )
            loss = loss + float(w_camera_acceleration_rot) * _normalized_smooth(
                quat_acceleration(axis_angle_to_quat(bg_pose_field.axis_angle)), T
            )
    for slot in hand_slots:
        n = slot.pose_field.num_frames()
        if in_world and frame_to_t is not None:
            idx = torch.tensor(
                [frame_to_t[f] for f in slot.pose_field.frame_indices],
                device=slot.pose_field.cam_t.device, dtype=torch.long,
            )
            local = torch.arange(n, device=idx.device)
            q_w, t_w = _camera_pose_to_world(
                bg_pose_field, idx, _hand_root_quats_camera(slot, local), slot.pose_field.cam_t,
            )
            rot_term = _observed_smooth(quat_smoothness, q_w, slot.pose_field.frame_indices)
            trans_term = _observed_smooth(temporal_smoothness, t_w, slot.pose_field.frame_indices)
        else:
            rot_term = _observed_smooth(rotation_smoothness, slot.pose_field.global_orient, slot.pose_field.frame_indices)
            trans_term = _observed_smooth(temporal_smoothness, slot.pose_field.cam_t, slot.pose_field.frame_indices)
        loss = loss + float(w_smooth_hand_rot) * rot_term
        loss = loss + float(w_smooth_hand_articulation) * _observed_smooth(
            rotation_smoothness, slot.pose_field.hand_pose.view(n, 15, 3), slot.pose_field.frame_indices
        )
        loss = loss + float(w_smooth_hand_trans) * trans_term
        for obj in obj_slots:
            loss = loss + _hand_object_relative_smoothness(
                obj.pose_field,
                slot,
                w_smooth_hand_object_relative_rot,
                w_smooth_hand_object_relative_trans,
            )
    return loss


def _hand_verts_joints_camera(
    pose_field: HandPoseField,
    chunk: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """All-frame MANO vertices (T,778,3) and joints (T,21,3) in camera frame.

    Applies the same mirroring, centroid ``hand_scale`` and ``cam_t`` as
    ``HandPoseField.batched_posed_verts_and_rotmats_camera``.
    """
    verts_out, joints_out = [], []
    n = pose_field.num_frames()
    with torch.no_grad():
        for start in range(0, n, chunk):
            idx = torch.arange(start, min(start + chunk, n), device=pose_field.cam_t.device)
            pose_aa = torch.cat([pose_field.global_orient[idx], pose_field.hand_pose[idx]], dim=-1)
            out = pose_field.mano(pose_aa, pose_field.betas.unsqueeze(0).expand(len(idx), -1))
            verts, joints = out.verts, out.joints
            if not pose_field.is_right:
                mirror = verts.new_tensor([-1.0, 1.0, 1.0])
                verts, joints = verts * mirror, joints * mirror
            c = verts.mean(dim=1, keepdim=True)
            s = pose_field.hand_scale
            verts = (verts - c) * s + c
            joints = (joints - c) * s + c
            shift = pose_field.cam_t[idx].unsqueeze(1)
            verts_out.append((verts + shift).cpu().numpy())
            joints_out.append((joints + shift).cpu().numpy())
    return np.concatenate(verts_out).astype(np.float32), np.concatenate(joints_out).astype(np.float32)


def _pose_matrices(R: torch.Tensor, t: torch.Tensor, scale: torch.Tensor | float = 1.0) -> np.ndarray:
    M = torch.eye(4, device=R.device, dtype=R.dtype).repeat(R.shape[0], 1, 1)
    M[:, :3, :3] = R * scale
    M[:, :3, 3] = t
    return M.detach().cpu().numpy().astype(np.float64)


def _scene_state(
    obj_slots: list[ObjectSlot],
    hand_slots: list[HandSlot],
    bg_pose_field: BackgroundPoseField | None,
    n_frames: int,
) -> dict[str, np.ndarray]:
    """Snapshot every dynamic pose as plain arrays (used before and after)."""
    state: dict[str, np.ndarray] = {}
    with torch.no_grad():
        if bg_pose_field is not None:
            idx = torch.arange(n_frames, device=bg_pose_field.axis_angle.device)
            state["camera_w2c"] = _pose_matrices(*bg_pose_field.batched_forward(idx))
        else:
            state["camera_w2c"] = np.tile(np.eye(4), (n_frames, 1, 1))
        for i, obj in enumerate(obj_slots):
            idx = torch.arange(obj.pose_field.num_frames(), device=obj.pose_field.axis_angle.device)
            R, t = obj.pose_field.batched_forward(idx)
            state[f"object_{i}_o2c"] = _pose_matrices(R, t, obj.gaussians.object_scale())
        for j, slot in enumerate(hand_slots):
            verts, joints = _hand_verts_joints_camera(slot.pose_field)
            state[f"hand_{j}_verts"] = verts
            state[f"hand_{j}_joints"] = joints
    return state


def _export_scene(
    path: str,
    initial: dict[str, np.ndarray] | None,
    refined: dict[str, np.ndarray],
    obj_slots: list[ObjectSlot],
    hand_slots: list[HandSlot],
    bg_gaussians: BackgroundGaussians | None,
    frame_indices: list[int],
    K: torch.Tensor,
    width: int,
    height: int,
    ref_frame_index: int,
    fps: float,
    max_bg_points: int,
) -> None:
    """Write initial/refined geometry and the optimized Gaussian state.

    World is the background reference camera (``camera_w2c`` is world→camera).
    Object poses are object→camera with the learned global scale folded into
    the rotation block; hand vertices and joints are in camera metres.
    """
    out: dict[str, np.ndarray] = {
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "ref_frame_index": np.asarray(ref_frame_index, dtype=np.int64),
        "fps": np.asarray(fps, dtype=np.float64),
        "K": K.detach().cpu().numpy().astype(np.float64),
        "image_size": np.asarray([width, height], dtype=np.int64),
        "object_names": np.asarray([obj.name for obj in obj_slots]),
        "hand_sides": np.asarray([slot.side for slot in hand_slots]),
        "gaussian_state_format": np.asarray("module_state_dict_v1"),
    }
    def save_gaussians(label, module):
        for key, tensor in module.state_dict().items():
            out[f"refined_gaussian_{label}__{key}"] = tensor.detach().cpu().numpy()

    for i, obj in enumerate(obj_slots):
        out[f"object_{i}_verts"] = obj.verts.detach().cpu().numpy().astype(np.float32)
        out[f"object_{i}_faces"] = np.asarray(obj.faces, dtype=np.int32)
        save_gaussians(f"object_{i}", obj.gaussians)
    for j, slot in enumerate(hand_slots):
        out[f"hand_{j}_frame_indices"] = np.asarray(slot.pose_field.frame_indices, dtype=np.int64)
        out[f"hand_{j}_faces"] = slot.pose_field.mano.th_faces.detach().cpu().numpy().astype(np.int32)
        save_gaussians(f"hand_{j}", slot.gaussians)
    if bg_gaussians is not None:
        save_gaussians("background", bg_gaussians)
        with torch.no_grad():
            pts = (bg_gaussians.anchor + bg_gaussians._delta_p).detach()
            keep = torch.sigmoid(bg_gaussians._opacity_logit.detach()) > 0.2
            pts, cols = pts[keep], bg_gaussians._color.detach()[keep].clamp(0.0, 1.0)
            if max_bg_points > 0 and pts.shape[0] > max_bg_points:
                sel = torch.randperm(pts.shape[0], device=pts.device)[:max_bg_points]
                pts, cols = pts[sel], cols[sel]
        out["background_points"] = pts.cpu().numpy().astype(np.float32)
        out["background_colors"] = cols.cpu().numpy().astype(np.float32)
    for prefix, state in (("initial", initial), ("refined", refined)):
        if state is None:
            continue
        for key, value in state.items():
            out[f"{prefix}_{key}"] = value
    for key, value in out.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise FloatingPointError(f"Non-finite scene output: {key}")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **out)


def _save_camera_poses(
    bg_pose_field: BackgroundPoseField,
    frame_indices: list[int],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    q = axis_angle_to_quat(bg_pose_field.axis_angle.detach()).cpu().tolist()
    tr = bg_pose_field.translation.detach().cpu().tolist()
    for t, fidx in enumerate(frame_indices):
        Transform3d(rotation=q[t], translation=tr[t], scale=[1.0, 1.0, 1.0]).save(
            os.path.join(output_dir, f"{fidx:06d}.json")
        )


def _write_overlay_video(
    output_path: str,
    cache: SimpleFrameCache,
    K: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    obj_slots: list[ObjectSlot],
    hand_slots: list[HandSlot],
    bg_gaussians: BackgroundGaussians | None,
    bg_pose_field: BackgroundPoseField | None,
    fps: float = 30.0,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        with torch.no_grad():
            for t, fidx in enumerate(tqdm(cache.frame_indices, desc="render overlay", ncols=80)):
                frame = _current_frame(
                    t, fidx, obj_slots, hand_slots,
                    bg_gaussians, bg_pose_field,
                )
                rgb = _render_2dgs_rgb(frame, K, width, height)
                arr = (rgb.clamp(0.0, 1.0) * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(arr).save(os.path.join(tmp, f"{t:06d}.png"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-r", str(fps),
            "-i", os.path.join(tmp, "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            output_path,
        ], check=True)


def _class_render_to_rgb(rendered_class: torch.Tensor) -> torch.Tensor:
    """Colorize alpha-composited class channels for progress visualization."""
    if rendered_class.numel() == 0 or rendered_class.shape[-1] == 0:
        h, w = rendered_class.shape[:2]
        return rendered_class.new_zeros((h, w, 3))
    palette = rendered_class.new_tensor([
        [1.00, 0.24, 0.12],  # object
        [0.10, 0.76, 1.00],  # hand slot 0
        [1.00, 0.22, 0.86],  # hand slot 1
        [0.70, 1.00, 0.18],
        [1.00, 0.78, 0.12],
    ])
    n_classes = int(rendered_class.shape[-1])
    if n_classes > palette.shape[0]:
        repeats = (n_classes + palette.shape[0] - 1) // palette.shape[0]
        palette = palette.repeat(repeats, 1)
    return (rendered_class.clamp(0.0, 1.0) @ palette[:n_classes]).clamp(0.0, 1.0)


def _dump_progress_frame(
    progress_dir: str,
    step: int,
    t: int,
    cache: SimpleFrameCache,
    K: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    obj_slots: list[ObjectSlot],
    hand_slots: list[HandSlot],
    bg_gaussians: BackgroundGaussians | None,
    bg_pose_field: BackgroundPoseField | None,
    mask_background: bool = False,
) -> None:
    os.makedirs(progress_dir, exist_ok=True)
    with torch.no_grad():
        fidx = cache.frame_indices[t]
        target = _target_rgb(cache, t, device, mask_background)
        n_mask_classes = len(obj_slots) + len(hand_slots)
        frame, labels = _current_frame_with_labels(
            t, fidx, obj_slots, hand_slots,
            bg_gaussians, bg_pose_field, n_mask_classes,
        )
        render_frame = replace(
            frame,
            colors=torch.cat([frame.colors, labels.to(frame.colors.dtype)], dim=-1),
        )
        rendered = _render_2dgs_features(render_frame, K, width, height)
        pred = rendered[..., :3].clamp(0.0, 1.0)
        rendered_mask = _class_render_to_rgb(rendered[..., 3:])
        err = (pred - target).abs().mul(4.0).clamp(0.0, 1.0)
        panel = torch.cat([target, pred, rendered_mask, err], dim=1)
        arr = (panel * 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(arr).save(os.path.join(progress_dir, f"{step:06d}.png"))


def _stitch_progress_video(progress_dir: str, fps: float = 12.0) -> None:
    if not os.path.isdir(progress_dir):
        return
    pngs = [p for p in os.listdir(progress_dir) if p.endswith(".png")]
    if not pngs:
        return
    out_path = os.path.join(os.path.dirname(progress_dir.rstrip("/")), "progress_simple.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-r", str(fps),
        "-pattern_type", "glob",
        "-i", os.path.join(progress_dir, "*.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        out_path,
    ], check=True)
    print(f"Wrote simple-refinement progress video -> {out_path}")


def refine_simple(
    frames_dir: str,
    depth_dir: str | None,
    intrinsics_path: str,
    object_mesh_path: str | None,
    object_poses_dir: str | None,
    object_mask_dir: str | None,
    refined_object_poses_dir: str | None,
    overlay_path: str,
    objects_root: str | None = None,
    refined_objects_root: str | None = None,
    missing_object_mask_unknown: bool = False,
    left_hand_pose_dir: str | None = None,
    left_hand_mask_dir: str | None = None,
    right_hand_pose_dir: str | None = None,
    right_hand_mask_dir: str | None = None,
    refined_left_hand_pose_dir: str | None = None,
    refined_right_hand_pose_dir: str | None = None,
    refined_camera_poses_dir: str | None = None,
    mano_assets_root: str | None = None,
    background_pose_init_dir: str | None = None,
    n_epochs: int = 20,
    batch_size: int = 4,
    train_resolution_scale: float = 0.5,
    lr_gaussians: float = 1e-2,
    lr_object_pose: float = 1e-3,
    lr_object_scale: float = 0.0,
    lr_hand_pose: float = 1e-3,
    lr_hand_articulation: float = 0.0,
    lr_hand_shape: float = 0.0,
    lr_hand_scale: float = 0.0,
    lr_camera_pose: float = 1e-4,
    lr_schedule: str = "constant",
    lr_cosine_min_factor: float = 0.0,
    w_smooth_object_rot: float = 0.01,
    w_smooth_object_trans: float = 0.01,
    w_smooth_hand_rot: float = 0.01,
    w_smooth_hand_articulation: float = 0.01,
    w_smooth_hand_trans: float = 0.01,
    w_smooth_hand_object_relative_rot: float = 0.0,
    w_smooth_hand_object_relative_trans: float = 0.0,
    w_smooth_camera_rot: float = 0.01,
    w_smooth_camera_trans: float = 0.01,
    smooth_in_world: bool = False,
    w_mask: float = 1.0,
    w_relative_depth: float = 0.0,
    w_metric_depth: float = 0.0,
    metric_depth_huber_delta: float = 0.02,
    metric_depth_max_m: float = 3.0,
    w_perceptual: float = 0.0,
    vgg_weights_path: str | None = None,
    perceptual_resize: int = 224,
    w_hand_object_penetration: float = 0.0,
    hand_object_penetration_margin: float = 0.003,
    object_sdf_resolution: int = 96,
    hand_object_penetration_max_verts: int = 0,
    mask_background: bool = False,
    bg_ref_frame: int | None = None,
    bg_init_stride: int = 10,
    bg_voxel_size: float = 0.005,
    bg_max_points: int = 50000,
    valid_mask_threshold: float = 0.04,
    valid_mask_erode_iters: int = 2,
    face_normal_thin_factor_obj: float = 0.25,
    face_normal_thin_factor_hand: float = 0.25,
    init_opacity_obj: float = 0.9,
    init_opacity_hand: float = 0.9,
    init_opacity_bg: float = 0.9,
    init_gaussian_scale_factor: float = 1.0,
    render_every: int = 25,
    progress_dir: str | None = None,
    debug_frame_idx: int | None = None,
    scene_export_path: str | None = None,
    scene_export_bg_points: int = 20000,
    fps: float = 30.0,
    seed: int = 0,
    device: str = "cuda",
    max_object_translation_change: float = 0.0,
    max_object_rotation_change_deg: float = 0.0,
    max_hand_translation_change: float = 0.0,
    max_hand_rotation_change_deg: float = 0.0,
    max_hand_articulation_change_deg: float = 0.0,
    max_camera_translation_change: float = 0.0,
    max_camera_rotation_change_deg: float = 0.0,
    w_camera_acceleration_trans: float = 0.0,
    w_camera_acceleration_rot: float = 0.0,
) -> None:
    object_specs = _object_specs(
        object_mesh_path, object_poses_dir, object_mask_dir, refined_object_poses_dir,
        objects_root, refined_objects_root,
    )
    if (left_hand_pose_dir or right_hand_pose_dir) and mano_assets_root is None:
        raise ValueError("mano_assets_root is required when refining hand tracks")
    if left_hand_pose_dir and left_hand_mask_dir is None:
        raise ValueError("left_hand_mask_dir is required with left_hand_pose_dir")
    if right_hand_pose_dir and right_hand_mask_dir is None:
        raise ValueError("right_hand_mask_dir is required with right_hand_pose_dir")
    if not mask_background and depth_dir is None:
        raise ValueError("depth_dir is required unless --mask-background is set")
    if float(w_relative_depth) > 0.0 and depth_dir is None:
        raise ValueError("depth_dir is required when --w_relative_depth is nonzero")
    if float(w_metric_depth) > 0.0 and depth_dir is None:
        raise ValueError("depth_dir is required when --w_metric_depth is nonzero")
    lr_schedule = str(lr_schedule).lower()
    if lr_schedule not in {"constant", "cosine"}:
        raise ValueError("lr_schedule must be 'constant' or 'cosine'")
    lr_cosine_min_factor = float(lr_cosine_min_factor)
    if lr_cosine_min_factor < 0.0 or lr_cosine_min_factor > 1.0:
        raise ValueError("lr_cosine_min_factor must be in [0, 1]")

    torch.manual_seed(seed)
    if any(not math.isfinite(w) or w < 0 for w in
           (w_camera_acceleration_trans, w_camera_acceleration_rot)):
        raise ValueError("Camera acceleration weights must be finite and nonnegative")
    device_t = torch.device(device)

    K, width, height = load_intrinsics(intrinsics_path, str(device_t))
    full_K, full_width, full_height = K.clone(), int(width), int(height)
    if train_resolution_scale != 1.0:
        s = float(train_resolution_scale)
        K = K.clone()
        K[0, 0] *= s
        K[1, 1] *= s
        K[0, 2] *= s
        K[1, 2] *= s
        width = max(1, int(round(width * s)))
        height = max(1, int(round(height * s)))
        print(f"Training/render resolution: {width}x{height} (scale={s:.3f})")

    available_frames = _list_frame_indices(frames_dir)
    # The training frames are those where every object has a pose and an RGB
    # frame exists; each object field is then indexed by the same positions.
    raw_object_tracks = [load_object_poses(spec.poses_dir, str(device_t)) for spec in object_specs]
    keep = set(available_frames)
    for track in raw_object_tracks:
        keep &= set(track.frame_indices)
    object_tracks = [_restrict_object_track(track, keep) for track in raw_object_tracks]
    frame_indices = list(object_tracks[0].frame_indices)
    keep = set(frame_indices)
    frame_to_t = {fidx: t for t, fidx in enumerate(frame_indices)}
    print(f"Objects: {[spec.name for spec in object_specs]} over {len(frame_indices)} frames")

    raw_hand_specs: list[tuple[str, str, str, str | None]] = []
    if left_hand_pose_dir is not None:
        raw_hand_specs.append(("left", left_hand_pose_dir, left_hand_mask_dir, refined_left_hand_pose_dir))
    if right_hand_pose_dir is not None:
        raw_hand_specs.append(("right", right_hand_pose_dir, right_hand_mask_dir, refined_right_hand_pose_dir))

    hand_tracks: list[tuple[str, HandPoseTrack, str, str | None]] = []
    for side, pose_dir, mask_dir, out_dir in raw_hand_specs:
        ht = _restrict_hand_track(load_hand_poses(pose_dir, str(device_t)), keep)
        hand_tracks.append((side, ht, mask_dir, out_dir))

    cache = SimpleFrameCache(
        frame_indices=frame_indices,
        frames_dir=frames_dir,
        depth_dir=depth_dir if (
            not mask_background or float(w_relative_depth) > 0.0 or float(w_metric_depth) > 0.0
        ) else None,
        object_mask_dirs=[spec.mask_dir for spec in object_specs],
        hand_mask_dirs=[h[2] for h in hand_tracks],
        height=height,
        width=width,
        valid_mask_threshold=valid_mask_threshold,
        valid_mask_erode_iters=valid_mask_erode_iters,
        missing_object_mask_unknown=missing_object_mask_unknown,
    )
    valid_mask = cache.valid_mask.to(device_t, non_blocking=True)
    valid_denom = valid_mask.sum().clamp_min(1.0)

    obj_slots: list[ObjectSlot] = []
    obj_g_params: list[torch.nn.Parameter] = []
    for spec, track in zip(object_specs, object_tracks):
        pose_field = ObjectPoseField(track).to(device_t)
        obj_verts, obj_colors, obj_faces = load_object_mesh(spec.mesh_path, str(device_t))
        obj_gaussians = init_object_face_gaussians_from_mesh(
            obj_verts, obj_faces, obj_colors,
            normal_thin_factor=face_normal_thin_factor_obj,
            init_opacity=init_opacity_obj,
            init_scale_factor=init_gaussian_scale_factor,
        ).to(device_t)
        obj_g_params.extend(_trainable_gaussian_params(obj_gaussians))
        if float(lr_object_scale) > 0.0 and hasattr(obj_gaussians, "_log_scale_global"):
            obj_gaussians._log_scale_global.requires_grad_(True)
        object_trainable = "appearance"
        if getattr(obj_gaussians, "_log_scale_global", None) is not None and obj_gaussians._log_scale_global.requires_grad:
            object_trainable += ", global scale"
        print(f"Object {spec.name}: {obj_gaussians.num_gaussians()} face-anchored Gaussians "
              f"(trainable: {object_trainable})")
        obj_slots.append(ObjectSlot(
            name=spec.name,
            pose_field=pose_field,
            gaussians=obj_gaussians,
            verts=obj_verts,
            faces=obj_faces,
            output_pose_dir=spec.output_pose_dir,
        ))

    hand_slots: list[HandSlot] = []
    for slot_i, (side, ht, mask_dir, out_dir) in enumerate(hand_tracks):
        pf = HandPoseField(
            ht,
            mano_assets_root,
            device=device_t,
            learn_hand_scale=float(lr_hand_scale) > 0.0,
        ).to(device_t)
        _freeze_all(pf)
        if float(lr_hand_pose) > 0.0:
            pf.global_orient.requires_grad_(True)
            pf.cam_t.requires_grad_(True)
        if float(lr_hand_articulation) > 0.0:
            pf.hand_pose.requires_grad_(True)
        if float(lr_hand_shape) > 0.0:
            pf.betas.requires_grad_(True)
        if float(lr_hand_scale) > 0.0:
            pf.hand_scale.requires_grad_(True)
        with torch.no_grad():
            zero_pose = torch.zeros(1, 48, device=device_t)
            zero_betas = torch.zeros(1, 10, device=device_t)
            rest = pf.mano(zero_pose, zero_betas).verts[0].detach().clone()
            if not ht.is_right:
                rest = rest * rest.new_tensor([-1.0, 1.0, 1.0])
        faces_np = pf.mano.th_faces.detach().cpu().numpy()
        hg = init_hand_face_gaussians(
            rest_vertices=rest,
            faces=faces_np,
            is_right=ht.is_right,
            normal_thin_factor=face_normal_thin_factor_hand,
            hand_scale_init=float(getattr(ht, "hand_scale", 1.0) or 1.0),
            init_opacity=init_opacity_hand,
            init_scale_factor=init_gaussian_scale_factor,
            device=device_t,
        ).to(device_t)
        if cache.hand_masks[slot_i].sum() > 0:
            mask_sum = cache.hand_masks[slot_i].sum().clamp_min(1.0)
            mean_rgb = (cache.rgb * cache.hand_masks[slot_i].unsqueeze(-1)).sum(dim=(0, 1, 2)) / mask_sum
            with torch.no_grad():
                hg._color.copy_(mean_rgb.to(device_t).expand_as(hg._color))
        hg_params = _trainable_gaussian_params(hg)
        trainable_hand = []
        if pf.global_orient.requires_grad or pf.cam_t.requires_grad:
            trainable_hand.append("root")
        if pf.hand_pose.requires_grad:
            trainable_hand.append("fingers")
        if pf.betas.requires_grad:
            trainable_hand.append("shape")
        if pf.hand_scale.requires_grad:
            trainable_hand.append("scale")
        trainable_desc = ", ".join(trainable_hand) if trainable_hand else "none"
        print(f"Hand {side}: {hg.num_gaussians()} face-anchored Gaussians "
              f"(trainable: {trainable_desc})")
        hand_slots.append(HandSlot(
            side=side,
            track=ht,
            pose_field=pf,
            gaussians=hg,
            mask_dir=mask_dir,
            output_pose_dir=out_dir,
            frame_to_pos={fidx: i for i, fidx in enumerate(ht.frame_indices)},
        ))
        del hg_params  # collected below by module reference

    if bg_ref_frame is not None and int(bg_ref_frame) in frame_indices:
        ref_t = frame_indices.index(int(bg_ref_frame))
    else:
        ref_t = 0
    bg_gaussians: BackgroundGaussians | None = None
    bg_pose_field: BackgroundPoseField | None = None
    bg_g_params: list[torch.nn.Parameter] = []
    if mask_background:
        if background_pose_init_dir is not None:
            print("Ignoring background_pose_init_dir because --mask-background omits background Gaussians.")
        if refined_camera_poses_dir is not None:
            print("Ignoring refined_camera_poses_dir because --mask-background omits camera/background pose refinement.")
        print("Mask background: target background is black and no background Gaussians are rendered.")
    else:
        if cache.depth is None:
            raise RuntimeError("Depth cache is missing; depth_dir is required for background refinement.")
        slam_poses = None
        if background_pose_init_dir is not None:
            slam_poses = _load_relative_w2c_poses(background_pose_init_dir, frame_indices, ref_t)

        def union_for_bg(t: int) -> torch.Tensor:
            m = cache.union_mask(t, device_t)
            m = torch.maximum(m, 1.0 - valid_mask)
            return m

        if slam_poses is not None and bg_init_stride > 1:
            positions = sorted({ref_t} | set(range(0, len(frame_indices), int(bg_init_stride))))
            positions = [t for t in positions if t in slam_poses]
            T_w2c_list = [torch.from_numpy(slam_poses[t]).to(device_t, dtype=torch.float32) for t in positions]
            anchors, colors, init_scales = init_background_multiframe(
                rgbs=[cache.rgb[t].to(device_t) for t in positions],
                depths=[cache.depth[t].to(device_t) for t in positions],
                union_masks=[union_for_bg(t) for t in positions],
                T_w2c_list=T_w2c_list,
                K=K,
                voxel_size=bg_voxel_size,
                max_points=bg_max_points,
            )
            print(f"Background: initialized from {len(positions)} frames")
        else:
            anchors, colors, init_scales = init_background_from_depth(
                rgb=cache.rgb[ref_t].to(device_t),
                depth=cache.depth[ref_t].to(device_t),
                union_mask=union_for_bg(ref_t),
                K=K,
                max_points=bg_max_points,
            )
            print(f"Background: initialized from frame {frame_indices[ref_t]:06d}")
        bg_gaussians = BackgroundGaussians(
            anchors, colors, init_scales,
            init_opacity=init_opacity_bg,
            init_scale_factor=init_gaussian_scale_factor,
        ).to(device_t)
        bg_g_params = _trainable_gaussian_params(bg_gaussians)
        bg_pose_field = BackgroundPoseField(len(frame_indices), device=device_t, ref_frame_t=ref_t).to(device_t)
        if slam_poses is not None:
            loaded, missing = _seed_background_pose_field(bg_pose_field, slam_poses)
            print(f"Camera/background pose seed: loaded {loaded}, missing {missing}")
        _anchor_camera_reference(bg_pose_field)
        print(f"Background: {bg_gaussians.num_gaussians()} depth-initialized Gaussians")

    use_mask_loss = float(w_mask) > 0.0
    n_mask_classes = len(obj_slots) + len(hand_slots)
    if use_mask_loss:
        print(f"Segmentation mask L1: weight={float(w_mask):.4g}, classes={n_mask_classes}")

    use_relative_depth_loss = float(w_relative_depth) > 0.0
    if use_relative_depth_loss:
        if cache.depth is None:
            raise RuntimeError("Depth cache is missing; depth_dir is required for relative depth loss.")
        region = "foreground masks" if mask_background else "valid image region"
        print(f"Relative MoGe depth-gradient loss: weight={float(w_relative_depth):.4g}, "
              f"region={region}")

    use_metric_depth_loss = float(w_metric_depth) > 0.0
    if use_metric_depth_loss:
        if cache.depth is None:
            raise RuntimeError("Depth cache is missing; depth_dir is required for metric depth loss.")
        print(f"Metric depth Huber loss: weight={float(w_metric_depth):.4g}, "
              f"delta={float(metric_depth_huber_delta):.4g} m, max={float(metric_depth_max_m):.3g} m")
    if smooth_in_world:
        if bg_pose_field is None:
            print("Ignoring smooth_in_world because there is no camera/background pose field.")
        else:
            print("Object and hand root smoothness measured in the background world frame.")

    if (
        float(w_smooth_hand_object_relative_rot) > 0.0
        or float(w_smooth_hand_object_relative_trans) > 0.0
    ):
        if hand_slots:
            print(
                "Hand/object relative smoothness: "
                f"rot={float(w_smooth_hand_object_relative_rot):.4g}, "
                f"trans={float(w_smooth_hand_object_relative_trans):.4g}"
            )
        else:
            print("Ignoring hand/object relative smoothness because no hand tracks are present.")

    use_perceptual_loss = float(w_perceptual) > 0.0
    perceptual_loss_fn: VGG16PerceptualLoss | None = None
    if use_perceptual_loss:
        perceptual_loss_fn = VGG16PerceptualLoss(
            weights_path=vgg_weights_path,
            resize=perceptual_resize,
        ).to(device_t)
        print(f"VGG16 perceptual L1: weight={float(w_perceptual):.4g}, "
              f"resize={int(perceptual_resize)}")

    use_penetration_loss = float(w_hand_object_penetration) > 0.0 and bool(hand_slots)
    if float(w_hand_object_penetration) > 0.0:
        if hand_slots:
            for obj in obj_slots:
                obj.sdf = _build_object_sdf_grid(
                    obj.verts,
                    obj.faces,
                    object_sdf_resolution,
                    hand_object_penetration_margin,
                    device_t,
                )
            print(
                f"Hand/object penetration: weight={float(w_hand_object_penetration):.4g}, "
                f"margin={float(hand_object_penetration_margin):.4g}, "
                f"sdf_res={int(object_sdf_resolution)}"
            )
        else:
            print("Ignoring hand/object penetration loss because no hand tracks are present.")

    progress_t = ref_t
    if debug_frame_idx is not None:
        try:
            progress_t = frame_indices.index(int(debug_frame_idx))
        except ValueError:
            print(f"Warning: debug_frame_idx={debug_frame_idx} not in sequence; "
                  f"using frame {frame_indices[progress_t]:06d} for progress.")
    if render_every > 0 and progress_dir is None:
        progress_dir = os.path.join(
            os.path.dirname(os.path.abspath(overlay_path)) or ".",
            "refined_simple_progress",
        )
    if render_every > 0 and progress_dir is not None:
        print(f"Simple-refinement progress frames -> {progress_dir} "
              f"(frame {frame_indices[progress_t]:06d}, every {render_every} steps)")
        _dump_progress_frame(
            progress_dir, 0, progress_t, cache, K, width, height, device_t,
            obj_slots, hand_slots, bg_gaussians, bg_pose_field,
            mask_background=mask_background,
        )

    initial_state = (
        _scene_state(obj_slots, hand_slots, bg_pose_field, len(frame_indices))
        if scene_export_path is not None else None
    )

    pose_trust = PoseTrustRegion()
    for obj in obj_slots:
        pose_trust.add(f"object/{obj.name}/translation", obj.pose_field.translation,
                       max_object_translation_change)
        pose_trust.add(f"object/{obj.name}/rotation", obj.pose_field.axis_angle,
                       math.radians(max_object_rotation_change_deg))
    for slot in hand_slots:
        pose_trust.add(f"hand/{slot.side}/translation", slot.pose_field.cam_t,
                       max_hand_translation_change)
        pose_trust.add(f"hand/{slot.side}/rotation", slot.pose_field.global_orient,
                       math.radians(max_hand_rotation_change_deg))
        pose_trust.add(f"hand/{slot.side}/articulation", slot.pose_field.hand_pose,
                       math.radians(max_hand_articulation_change_deg))
    if bg_pose_field is not None:
        pose_trust.add("camera/translation", bg_pose_field.translation,
                       max_camera_translation_change)
        pose_trust.add("camera/rotation", bg_pose_field.axis_angle,
                       math.radians(max_camera_rotation_change_deg))

    param_groups: list[dict] = []
    if obj_g_params:
        param_groups.append({"params": obj_g_params, "lr": lr_gaussians})
    for slot in hand_slots:
        params = [p for p in (slot.gaussians._log_scale, slot.gaussians._opacity_logit, slot.gaussians._color) if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr_gaussians})
    if bg_g_params:
        param_groups.append({"params": bg_g_params, "lr": lr_gaussians})
    for obj in obj_slots:
        if float(lr_object_scale) > 0.0 and hasattr(obj.gaussians, "_log_scale_global"):
            param_groups.append({"params": [obj.gaussians._log_scale_global], "lr": lr_object_scale})
        param_groups.extend([
            {"params": [obj.pose_field.axis_angle], "lr": lr_object_pose},
            {"params": [obj.pose_field.translation], "lr": lr_object_pose},
        ])
    if bg_pose_field is not None:
        param_groups.extend([
            {"params": [bg_pose_field.axis_angle], "lr": lr_camera_pose},
            {"params": [bg_pose_field.translation], "lr": lr_camera_pose},
        ])
    for slot in hand_slots:
        if float(lr_hand_pose) > 0.0:
            param_groups.append({"params": [slot.pose_field.global_orient], "lr": lr_hand_pose})
            param_groups.append({"params": [slot.pose_field.cam_t], "lr": lr_hand_pose})
        if float(lr_hand_articulation) > 0.0:
            param_groups.append({"params": [slot.pose_field.hand_pose], "lr": lr_hand_articulation})
        if float(lr_hand_shape) > 0.0:
            param_groups.append({"params": [slot.pose_field.betas], "lr": lr_hand_shape})
        if float(lr_hand_scale) > 0.0:
            param_groups.append({"params": [slot.pose_field.hand_scale], "lr": lr_hand_scale})

    optimizer = torch.optim.Adam(param_groups)
    # The rasterizer can occasionally emit an invalid backward value for a
    # plausible initial trajectory. Keep that one mini-batch from poisoning
    # every pose through the sequence-wide smoothness terms.
    trainable_params = [
        param for group in optimizer.param_groups for param in group["params"]
    ]
    max_grad_norm = 1.0
    print(f"Gradient clipping: global norm <= {max_grad_norm:g}")
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    T = len(frame_indices)
    n_batches = max(1, (T + int(batch_size) - 1) // int(batch_size))
    total_steps = int(n_epochs) * n_batches
    lr_scale = 1.0
    if lr_schedule == "cosine":
        print(f"LR schedule: cosine, min_factor={lr_cosine_min_factor:.4g}, steps={total_steps}")
    pbar = tqdm(total=total_steps, ncols=100, desc="refine_simple")
    successful_steps = 0
    trace_path = None
    if progress_dir is not None:
        os.makedirs(progress_dir, exist_ok=True)
        trace_path = os.path.join(progress_dir, "optimization_trace.jsonl")

    def record_step(status, batch, **values):
        if trace_path is not None:
            with open(trace_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, "status": status,
                                         "frames": [frame_indices[t] for t in batch],
                                         **values}, allow_nan=False) + "\n")

    step = 0
    for _epoch in range(int(n_epochs)):
        order = torch.randperm(T).tolist()
        for start in range(0, T, int(batch_size)):
            batch = order[start:start + int(batch_size)]
            if lr_schedule == "cosine":
                lr_scale = _cosine_lr_scale(step, total_steps, lr_cosine_min_factor)
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * lr_scale
            optimizer.zero_grad(set_to_none=True)
            zero = obj_slots[0].pose_field.axis_angle.new_zeros(())
            photo = zero
            mask_l1 = zero
            relative_depth = zero
            metric_depth = zero
            perceptual = zero
            penetration = zero
            for t in batch:
                fidx = frame_indices[t]
                target = _target_rgb(cache, t, device_t, mask_background)
                if use_mask_loss:
                    frame, labels = _current_frame_with_labels(
                        t, fidx, obj_slots, hand_slots,
                        bg_gaussians, bg_pose_field, n_mask_classes,
                    )
                    render_frame = replace(
                        frame,
                        colors=torch.cat([frame.colors, labels.to(frame.colors.dtype)], dim=-1),
                    )
                    rendered = _render_2dgs_features(render_frame, K, width, height)
                    pred = rendered[..., :3]
                    class_pred = rendered[..., 3:]
                    mask_l1 = mask_l1 + _segmentation_mask_loss(
                        class_pred, cache, t, valid_mask, valid_denom,
                    )
                else:
                    frame = _current_frame(
                        t, fidx, obj_slots, hand_slots,
                        bg_gaussians, bg_pose_field,
                    )
                    pred = _render_2dgs_rgb(frame, K, width, height)
                per_pixel = (pred - target).abs().sum(dim=-1)
                photo = photo + (per_pixel * valid_mask).sum() / valid_denom
                if perceptual_loss_fn is not None:
                    perceptual = perceptual + perceptual_loss_fn(pred, target, valid_mask)
                if use_relative_depth_loss or use_metric_depth_loss:
                    rendered_depth, rendered_alpha = _render_2dgs_depth(frame, K, width, height)
                    target_depth = cache.depth[t].to(device_t, non_blocking=True)
                    depth_mask = (
                        cache.union_mask(t, device_t)
                        if mask_background else valid_mask
                    )
                if use_relative_depth_loss:
                    relative_depth = relative_depth + depth_gradient_loss(
                        rendered_depth,
                        target_depth,
                        depth_mask * (rendered_alpha > 1e-3).to(depth_mask.dtype),
                    )
                if use_metric_depth_loss:
                    # Stereo depth is metric but sparse: zero/inf pixels and
                    # values past the stereo range are unknown, not far away.
                    metric_valid = (
                        torch.isfinite(target_depth)
                        & (target_depth > 0.0)
                        & (target_depth < float(metric_depth_max_m))
                        & (rendered_alpha > 0.5)
                    ).to(depth_mask.dtype)
                    metric_depth = metric_depth + depth_loss(
                        rendered_depth,
                        torch.where(metric_valid > 0, target_depth, torch.zeros_like(target_depth)),
                        depth_mask * metric_valid,
                        huber_delta=float(metric_depth_huber_delta),
                    )
                if use_penetration_loss:
                    for obj in obj_slots:
                        penetration = penetration + _hand_object_penetration_loss(
                            t,
                            fidx,
                            obj,
                            hand_slots,
                            hand_object_penetration_margin,
                            hand_object_penetration_max_verts,
                        )
            photo = photo / float(max(len(batch), 1))
            mask_l1 = mask_l1 / float(max(len(batch), 1))
            relative_depth = relative_depth / float(max(len(batch), 1))
            metric_depth = metric_depth / float(max(len(batch), 1))
            perceptual = perceptual / float(max(len(batch), 1))
            penetration = penetration / float(max(len(batch), 1))
            smooth = _pose_smoothness(
                obj_slots, hand_slots, bg_pose_field,
                w_smooth_object_rot, w_smooth_object_trans,
                w_smooth_hand_rot, w_smooth_hand_articulation, w_smooth_hand_trans,
                w_smooth_hand_object_relative_rot, w_smooth_hand_object_relative_trans,
                w_smooth_camera_rot, w_smooth_camera_trans,
                smooth_in_world=smooth_in_world,
                frame_to_t=frame_to_t,
                w_camera_acceleration_trans=w_camera_acceleration_trans,
                w_camera_acceleration_rot=w_camera_acceleration_rot,
            ) / float(n_batches)
            loss = (
                photo
                + float(w_mask) * mask_l1
                + float(w_relative_depth) * relative_depth
                + float(w_metric_depth) * metric_depth
                + float(w_perceptual) * perceptual
                + float(w_hand_object_penetration) * penetration
                + smooth
            )
            if not bool(torch.isfinite(loss)):
                record_step("skipped", batch, reason="nonfinite_loss")
                print(
                    f"WARNING: skipping non-finite refinement loss at step={step}, "
                    f"frames={[frame_indices[t] for t in batch]}"
                )
                optimizer.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)
                continue

            loss.backward()
            bad_gradient = _first_nonfinite_tensor([
                (f"gradient for group={group_i} param={param_i} shape={tuple(param.shape)}", param.grad)
                for group_i, group in enumerate(optimizer.param_groups)
                for param_i, param in enumerate(group["params"])
            ])
            if bad_gradient is not None:
                record_step("skipped", batch, reason="nonfinite_gradient", detail=bad_gradient)
                print(
                    f"WARNING: skipping non-finite refinement gradient at step={step}, "
                    f"frames={[frame_indices[t] for t in batch]}: {bad_gradient}"
                )
                optimizer.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=max_grad_norm, error_if_nonfinite=False
            )
            if not bool(torch.isfinite(grad_norm)):
                record_step("skipped", batch, reason="nonfinite_gradient_norm")
                print(
                    f"WARNING: skipping non-finite gradient norm at step={step}, "
                    f"frames={[frame_indices[t] for t in batch]}"
                )
                optimizer.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)
                continue

            optimizer.step()
            bad_parameter = _first_nonfinite_tensor([
                (f"group={group_i} param={param_i} shape={tuple(param.shape)}", param)
                for group_i, group in enumerate(optimizer.param_groups)
                for param_i, param in enumerate(group["params"])
            ])
            if bad_parameter is not None:
                raise FloatingPointError(
                    f"Non-finite refinement parameter after optimizer step {step}, "
                    f"frames={[frame_indices[t] for t in batch]}: {bad_parameter}"
                )
            if bg_pose_field is not None:
                _anchor_camera_reference(bg_pose_field)
            trust_clipped = pose_trust.project()
            successful_steps += 1
            record_step("updated", batch, epoch=_epoch, loss=float(loss.detach()),
                        photo=float(photo.detach()), mask=float(mask_l1.detach()),
                        relative_depth=float(relative_depth.detach()), metric_depth=float(metric_depth.detach()),
                        smooth=float(smooth.detach()), gradient_norm=float(grad_norm.detach()), lr_scale=lr_scale,
                        pose_trust_clipped=trust_clipped)
            step += 1
            if render_every > 0 and progress_dir is not None and step % int(render_every) == 0:
                _dump_progress_frame(
                    progress_dir, step, progress_t, cache, K, width, height, device_t,
                    obj_slots, hand_slots, bg_gaussians, bg_pose_field,
                    mask_background=mask_background,
                )
            pbar.update(1)
            pbar.set_postfix(
                photo=f"{float(photo.detach()):.4f}",
                mask=f"{float(mask_l1.detach()):.4f}",
                rdepth=f"{float(relative_depth.detach()):.4f}",
                mdepth=f"{float(metric_depth.detach()):.4f}",
                perc=f"{float(perceptual.detach()):.4f}",
                pen=f"{float(penetration.detach()):.4f}",
                lr=f"{lr_scale:.3g}",
                smooth=f"{float(smooth.detach()):.4f}",
            )
    pbar.close()
    if successful_steps == 0:
        raise FloatingPointError("Refinement produced zero successful optimizer updates")
    print(f"Optimizer updates: {successful_steps}/{total_steps}; skipped: {total_steps - successful_steps}")
    if render_every > 0 and progress_dir is not None:
        _dump_progress_frame(
            progress_dir, step, progress_t, cache, K, width, height, device_t,
            obj_slots, hand_slots, bg_gaussians, bg_pose_field,
            mask_background=mask_background,
        )
        _stitch_progress_video(progress_dir)

    for obj in obj_slots:
        refined_obj_track = obj.pose_field.export_track()
        s_obj_learned = float(obj.gaussians.object_scale().detach())
        refined_obj_track.scales = refined_obj_track.scales * s_obj_learned
        if float(lr_object_scale) > 0.0:
            print(f"Learned {obj.name} scale: {s_obj_learned:.4f}")
        if obj.output_pose_dir is None:
            continue
        save_object_poses(refined_obj_track, obj.output_pose_dir)
        print(f"Wrote refined {obj.name} poses -> {obj.output_pose_dir}")
    if scene_export_path is not None:
        _export_scene(
            scene_export_path,
            initial_state,
            _scene_state(obj_slots, hand_slots, bg_pose_field, len(frame_indices)),
            obj_slots,
            hand_slots,
            bg_gaussians,
            frame_indices,
            full_K,
            full_width,
            full_height,
            frame_indices[ref_t],
            fps,
            scene_export_bg_points,
        )
        print(f"Wrote initial/refined scene -> {scene_export_path}")
    for slot in hand_slots:
        if slot.output_pose_dir is None:
            continue
        save_hand_poses(slot.pose_field.export_track(slot.track.raw_records), slot.output_pose_dir)
        print(f"Wrote refined {slot.side} hand poses -> {slot.output_pose_dir}")
    if refined_camera_poses_dir is not None and bg_pose_field is not None:
        _save_camera_poses(bg_pose_field, frame_indices, refined_camera_poses_dir)
        print(f"Wrote refined camera/background poses -> {refined_camera_poses_dir}")

    _write_overlay_video(
        output_path=overlay_path,
        cache=cache,
        K=K,
        width=width,
        height=height,
        device=device_t,
        obj_slots=obj_slots,
        hand_slots=hand_slots,
        bg_gaussians=bg_gaussians,
        bg_pose_field=bg_pose_field,
        fps=fps,
    )
    print(f"Wrote overlay -> {overlay_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--depth_dir", default=None,
                   help="Depth frames for background initialization. Required unless --mask-background is set.")
    p.add_argument("--intrinsics_path", required=True)
    p.add_argument("--object_mesh_path", default=None)
    p.add_argument("--object_poses_dir", default=None)
    p.add_argument("--object_mask_dir", default=None)
    p.add_argument("--refined_object_poses_dir", default=None)
    p.add_argument("--overlay_path", required=True)
    p.add_argument("--objects_root", default=None,
                   help="Multi-object input: <root>/<name>/{mesh.<ext>, poses/, masks/}. "
                        "May be combined with the single-object flags.")
    p.add_argument("--refined_objects_root", default=None,
                   help="Refined poses for --objects_root go to <root>/<name>/.")
    p.add_argument("--missing_object_mask_unknown", action="store_true",
                   help="Treat a missing object mask file as unknown (skip that channel) "
                        "instead of as an empty mask.")
    p.add_argument("--left_hand_pose_dir", default=None)
    p.add_argument("--left_hand_mask_dir", default=None)
    p.add_argument("--right_hand_pose_dir", default=None)
    p.add_argument("--right_hand_mask_dir", default=None)
    p.add_argument("--refined_left_hand_pose_dir", default=None)
    p.add_argument("--refined_right_hand_pose_dir", default=None)
    p.add_argument("--refined_camera_poses_dir", default=None)
    p.add_argument("--mano_assets_root", default=None)
    p.add_argument("--background_pose_init_dir", default=None)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--train_resolution_scale", type=float, default=0.5)
    p.add_argument("--lr_gaussians", type=float, default=1e-2)
    p.add_argument("--lr_object_pose", type=float, default=1e-3)
    p.add_argument("--lr_object_scale", type=float, default=0.0,
                   help="LR for global object scale. 0 disables.")
    p.add_argument("--lr_hand_pose", type=float, default=1e-3,
                   help="LR for hand root global_orient and cam_t. 0 disables root pose refinement.")
    p.add_argument("--lr_hand_articulation", type=float, default=0.0,
                   help="LR for MANO finger articulation hand_pose. 0 disables.")
    p.add_argument("--lr_hand_shape", type=float, default=0.0,
                   help="LR for shared MANO betas/shape per hand track. 0 disables.")
    p.add_argument("--lr_hand_scale", type=float, default=0.0,
                   help="LR for shared per-track hand_scale. 0 disables.")
    p.add_argument("--lr_camera_pose", type=float, default=1e-4)
    p.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant",
                   help="Learning-rate schedule for all optimizer groups.")
    p.add_argument("--lr_cosine_min_factor", type=float, default=0.0,
                   help="Final LR multiplier for --lr_schedule cosine. 0 decays to zero.")
    p.add_argument("--w_smooth_object_rot", type=float, default=0.01)
    p.add_argument("--w_smooth_object_trans", type=float, default=0.01)
    p.add_argument("--w_smooth_hand_rot", type=float, default=0.01)
    p.add_argument("--w_smooth_hand_articulation", type=float, default=0.01,
                   help="Temporal rotation smoothness for MANO finger hand_pose.")
    p.add_argument("--w_smooth_hand_trans", type=float, default=0.01)
    p.add_argument("--w_smooth_hand_object_relative_rot", type=float, default=0.0,
                   help="Temporal smoothness for hand root rotation relative to object pose. 0 disables.")
    p.add_argument("--w_smooth_hand_object_relative_trans", type=float, default=0.0,
                   help="Temporal smoothness for hand root translation in the object frame. "
                        "0 disables.")
    p.add_argument("--w_smooth_camera_rot", type=float, default=0.01)
    p.add_argument("--w_smooth_camera_trans", type=float, default=0.01)
    p.add_argument("--w_camera_acceleration_trans", type=float, default=0.0,
                   help="Second-difference penalty on the world camera centre; 0 disables.")
    p.add_argument("--w_camera_acceleration_rot", type=float, default=0.0,
                   help="Penalty on changes of camera relative rotation; 0 disables.")
    p.add_argument("--smooth_in_world", action="store_true",
                   help="Measure object/hand root smoothness in the background world frame "
                        "instead of the camera frame.")
    p.add_argument("--w_mask", type=float, default=1.0,
                   help="Weight for L1 segmentation mask loss. 0 disables.")
    p.add_argument("--w_relative_depth", type=float, default=0.0,
                   help="Weight for log-depth-gradient loss against MoGe depth. 0 disables.")
    p.add_argument("--w_metric_depth", type=float, default=0.0,
                   help="Weight for absolute Huber depth loss against metric (e.g. stereo) depth. 0 disables.")
    p.add_argument("--metric_depth_huber_delta", type=float, default=0.02,
                   help="Huber transition for --w_metric_depth, in metres.")
    p.add_argument("--metric_depth_max_m", type=float, default=3.0,
                   help="Ignore target depth at or beyond this range for --w_metric_depth.")
    p.add_argument("--w_perceptual", type=float, default=0.0,
                   help="Weight for VGG16 perceptual feature L1 loss. 0 disables.")
    p.add_argument("--vgg_weights_path", default=None,
                   help="Optional local VGG16 state_dict path. If missing, torchvision ImageNet weights are downloaded there.")
    p.add_argument("--perceptual_resize", type=int, default=224,
                   help="Square resize for VGG perceptual loss. <=0 uses training resolution.")
    p.add_argument("--w_hand_object_penetration", type=float, default=0.0,
                   help="Weight for 3D hand/object SDF penetration loss. 0 disables.")
    p.add_argument("--hand_object_penetration_margin", type=float, default=0.003,
                   help="Signed-distance margin in meters/object units for hand/object separation.")
    p.add_argument("--object_sdf_resolution", type=int, default=96,
                   help="Resolution per axis for the object SDF grid used by hand/object penetration.")
    p.add_argument("--hand_object_penetration_max_verts", type=int, default=0,
                   help="Max MANO verts per hand for penetration loss. 0 uses all vertices.")
    p.add_argument("--mask-background", "--mask_background", dest="mask_background", action="store_true",
                   help="Black out target background and omit background Gaussians/camera refinement.")
    p.add_argument("--bg_ref_frame", type=int, default=None)
    p.add_argument("--bg_init_stride", type=int, default=10)
    p.add_argument("--bg_voxel_size", type=float, default=0.005)
    p.add_argument("--bg_max_points", type=int, default=50000)
    p.add_argument("--valid_mask_threshold", type=float, default=0.04)
    p.add_argument("--valid_mask_erode_iters", type=int, default=2)
    p.add_argument("--face_normal_thin_factor_obj", type=float, default=0.25)
    p.add_argument("--face_normal_thin_factor_hand", type=float, default=0.25)
    p.add_argument("--init_opacity_obj", type=float, default=0.9)
    p.add_argument("--init_opacity_hand", type=float, default=0.9)
    p.add_argument("--init_opacity_bg", type=float, default=0.9)
    p.add_argument("--init_gaussian_scale_factor", type=float, default=1.0,
                   help="Multiplier for all initial Gaussian scales/radii.")
    p.add_argument("--render_every", type=int, default=25,
                   help="Write a target/render/error progress panel every N optimizer steps. 0 disables.")
    p.add_argument("--progress_dir", default=None,
                   help="Directory for simple-refinement progress PNGs. Defaults next to overlay_path.")
    p.add_argument("--debug_frame_idx", type=int, default=None,
                   help="Frame index to use for intermediate progress visualizations.")
    p.add_argument("--scene_export_path", default=None,
                   help="Write an .npz with initial and refined camera/object/hand state for reporting.")
    p.add_argument("--scene_export_bg_points", type=int, default=20000,
                   help="Max background Gaussian centres in --scene_export_path. 0 keeps all.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    for name in ("object", "hand", "camera"):
        p.add_argument(f"--max_{name}_translation_change", type=float, default=0.0,
                       help="Maximum initialization-relative translation change in metres;0 disables.")
        p.add_argument(f"--max_{name}_rotation_change_deg", type=float, default=0.0,
                       help="Maximum initialization-relative axis-angle change in degrees;0 disables.")
    p.add_argument("--max_hand_articulation_change_deg", type=float, default=0.0,
                   help="Maximum per-finger-joint change from initialization in degrees;0 disables.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    refine_simple(**vars(args))


if __name__ == "__main__":
    main()
