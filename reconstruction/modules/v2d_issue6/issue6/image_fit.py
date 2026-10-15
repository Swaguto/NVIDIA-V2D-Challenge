"""Fit MANO to source hand landmarks and stereo while regularizing world motion.

Landmarks are saved model predictions, not independent ground truth. MANO shape,
CAD, calibrated intrinsics and fitted camera poses are fixed throughout.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def initialize_trailing_pose(poses, records, rotations, is_right):
    """Warm start trailing poses, then fit them to new image/stereo observations."""
    poses = np.array(poses, float, copy=True)
    observed = [i for i, r in enumerate(records) if r.get("directly_observed", True)]
    initialized = []
    if observed:
        anchor = observed[-1]
        mirror = np.diag([1 if is_right else -1, 1, 1])
        root = Rotation.from_rotvec(poses[anchor, :3]).as_matrix()
        for i in range(anchor + 1, len(records)):
            if records[i].get("flow_depth_supported", False):
                relative = mirror @ rotations[i] @ rotations[anchor].T @ mirror
                poses[i, :3] = Rotation.from_matrix(relative @ root).as_rotvec()
                poses[i, 3:] = poses[anchor, 3:]
                initialized.append(i)
    return poses, initialized


def refine_sequence(
    mano,
    records,
    keypoints,
    depth_targets,
    depth_weights,
    rotations,
    camera_translations,
    camera,
    is_right,
    iterations=200,
):
    device = next(mano.buffers()).device

    def tensor(x):
        return torch.as_tensor(np.array(x), dtype=torch.float32, device=device)

    pose_prior, trailing_initializations = initialize_trailing_pose(
        [r["mano"]["global_orient"] + r["mano"]["hand_pose"] for r in records], records, np.asarray(rotations), is_right
    )
    initial_pose = tensor(pose_prior)
    initial_t = tensor([r["cam_t"] for r in records])
    betas = tensor([r["mano"]["betas"] for r in records])
    scale = tensor([r["hand_scale"] for r in records])[:, None, None]
    points = tensor(keypoints)
    depths = tensor(depth_targets)
    dw = tensor(depth_weights)
    R = tensor(rotations)
    camera_t = tensor(camera_translations)
    mirror = tensor([1 if is_right else -1, 1, 1])
    fx, fy, cx, cy = [camera[k] for k in ("fx", "fy", "cx", "cy")]
    stereo_initializations = []
    with torch.no_grad():
        model = mano(initial_pose, betas)
        local_v, local_j = model.verts * mirror, model.joints * mirror
        centre = local_v.mean(dim=1, keepdim=True)
        local_j = (local_j - centre) * scale + centre
        for i, record in enumerate(records):
            valid = (dw[i] > 0) & (points[i, :, 2] > 0)
            if record.get("flow_supported", False) and torch.count_nonzero(valid) >= 4:
                z = depths[i, valid]
                measured = torch.stack(
                    [(points[i, valid, 0] - cx) * z / fx, (points[i, valid, 1] - cy) * z / fy, z], dim=-1
                )
                initial_t[i] = (measured - local_j[i, valid]).median(dim=0).values
                stereo_initializations.append(i)
    delta_pose = torch.nn.Parameter(torch.zeros_like(initial_pose))
    delta_t = torch.nn.Parameter(torch.zeros_like(initial_t))
    optimizer = torch.optim.Adam([{"params": [delta_pose], "lr": 0.005}, {"params": [delta_t], "lr": 0.001}])

    def geometry():
        result = mano(initial_pose + delta_pose, betas)
        v, j = result.verts * mirror, result.joints * mirror
        centroid = v.mean(dim=1, keepdim=True)
        v = (v - centroid) * scale + centroid + (initial_t + delta_t)[:, None]
        j = (j - centroid) * scale + centroid + (initial_t + delta_t)[:, None]
        return v, j

    def project(j):
        z = j[..., 2].clamp(min=0.03)
        return torch.stack([fx * j[..., 0] / z + cx, fy * j[..., 1] / z + cy], dim=-1)

    def error(j):
        residual = torch.linalg.vector_norm(project(j) - points[..., :2], dim=-1)
        return residual[points[..., 2] > 0]

    with torch.no_grad():
        _, initial_j = geometry()
        before = error(initial_j)
    for iteration in range(iterations):
        optimizer.zero_grad()
        _, joints = geometry()
        projected = project(joints)
        valid = points[..., 2] > 0
        image_residual = (projected - points[..., :2]) / 8
        image_loss = (
            torch.nn.functional.huber_loss(
                image_residual, torch.zeros_like(image_residual), delta=1, reduction="none"
            ).sum(-1)
            * points[..., 2]
        ).sum() / valid.sum().clamp(min=1)
        depth_residual = (joints[..., 2] - depths) / 0.015
        depth_loss = (
            torch.nn.functional.huber_loss(depth_residual, torch.zeros_like(depth_residual), delta=1, reduction="none")
            * dw
        ).sum() / dw.sum().clamp(min=1)
        world = torch.einsum("nji,nkj->nki", R, joints - camera_t[:, None])
        # A 2 mm discrete acceleration penalty at the original ~20 Hz axis.
        acceleration = world[2:] - 2 * world[1:-1] + world[:-2]
        smooth = torch.nn.functional.huber_loss(
            acceleration / 0.002, torch.zeros_like(acceleration), delta=1, reduction="mean"
        )
        prior = (delta_pose / 0.15).square().mean() + (delta_t / 0.15).square().mean()
        loss = image_loss + depth_loss + 0.3 * smooth + 0.03 * prior
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite image/stereo sequence fit")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta_t.clamp_(-0.25, 0.25)
            delta_pose[:, :3].clamp_(-0.52, 0.52)
            delta_pose[:, 3:].clamp_(-0.35, 0.35)
        if iteration % 50 == 0:
            print("image/stereo refinement", iteration, float(loss.detach()), flush=True)
    with torch.no_grad():
        _, final_j = geometry()
        after = error(final_j)
    receipt = {
        "iterations": iterations,
        "landmark_source": "saved wearer HaMeR predictions",
        "median_keypoint_error_px_before": float(before.median()) if len(before) else None,
        "median_keypoint_error_px_after": float(after.median()) if len(after) else None,
        "p95_keypoint_error_px_after": float(torch.quantile(after, 0.95)) if len(after) else None,
        "depth_landmarks": int(torch.count_nonzero(dw)),
        "inferred_frames_initialized_from_stereo": stereo_initializations,
        "trailing_pose_initializations": trailing_initializations,
        "independent_accuracy_claimed": False,
    }
    return (initial_pose + delta_pose).detach().cpu().numpy(), (initial_t + delta_t).detach().cpu().numpy(), receipt


def refine_available_sequence(
    mano,
    records,
    keypoints,
    depth_targets,
    depth_weights,
    rotations,
    camera_translations,
    camera,
    is_right,
    iterations=200,
):
    """Fit contiguous finite priors without smoothing across unknown geometry."""
    if all(record is not None for record in records):
        return refine_sequence(
            mano,
            records,
            keypoints,
            depth_targets,
            depth_weights,
            rotations,
            camera_translations,
            camera,
            is_right,
            iterations,
        )
    n = len(records)
    poses = np.full((n, 48), np.nan)
    translations = np.full((n, 3), np.nan)
    available = np.array([record is not None for record in records])
    boundaries = np.flatnonzero(np.diff(np.r_[False, available, False]))
    segments = []
    for start, end in boundaries.reshape(-1, 2):
        selected = records[start:end]
        if end - start < 3:
            poses[start:end] = [r["mano"]["global_orient"] + r["mano"]["hand_pose"] for r in selected]
            translations[start:end] = [r["cam_t"] for r in selected]
            receipt = {"optimized": False, "reason": "fewer_than_three_finite_frames"}
        else:
            poses[start:end], translations[start:end], receipt = refine_sequence(
                mano,
                selected,
                keypoints[start:end],
                depth_targets[start:end],
                depth_weights[start:end],
                rotations[start:end],
                camera_translations[start:end],
                camera,
                is_right,
                iterations,
            )
        segments.append({"start_index": int(start), "end_index_exclusive": int(end), "receipt": receipt})
    return (
        poses,
        translations,
        {
            "segments": segments,
            "unknown_frame_indices": np.flatnonzero(~available).tolist(),
            "independent_accuracy_claimed": False,
        },
    )
