#!/usr/bin/env python3
"""Track one public CAD object with FoundationPose and rectified stereo depth.

Use masks from temporal_masks.py. Missing or depth-poor masks remain gaps; a
later world-frame inference stage may fill them with explicit provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("V2D_RUNTIME", str(ROOT / "reconstruction/modules/v2d_stereo_depth/lib")))
from stereo_depth import load_rig, stereo_depth  # noqa: E402


def mesh_for(data: Path, name: str) -> Path:
    folder = data / "mesh" / name
    for suffix in ("", "_visual", "_collision"):
        path = folder / f"{name}{suffix}.glb"
        if path.is_file():
            return path
    raise FileNotFoundError(f"CAD missing for {name}")


def centered_tracker_pose(pose, model_center):
    """FoundationPose stores centered-mesh poses; exports use the unchanged raw mesh."""
    result = np.asarray(pose, float).copy()
    result[:3, 3] += result[:3, :3] @ np.asarray(model_center, float)
    return result


def foreground_depth(depth, valid, mask):
    """Exclude other surfaces from refinement without changing measured depths.

    The separate acceptance gate still uses the original depth and foreground.
    This optional experiment supplies no depth outside the observed object mask.
    """
    depth, valid, mask = np.asarray(depth), np.asarray(valid), np.asarray(mask)
    if depth.ndim != 2 or valid.shape != depth.shape or mask.shape != depth.shape:
        raise ValueError("Depth, validity and foreground must share the calibrated grid")
    selected = valid & (mask > 127) & np.isfinite(depth) & (depth > 0)
    return np.where(selected, depth, 0).astype(np.float32)


def pose_rejection(pose, depth, valid, mask, K, bounds, bootstrap, jump_prior):
    """Use the same measured-depth and 15cm/40degree gates on every attempt."""
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    support = None
    if not np.isfinite(pose).all() or pose[2, 3] <= 0.05:
        return {"error": "invalid pose"}, support
    if bootstrap:
        from reconstruct_stereo_object import fusion_depth
        _, support = fusion_depth(depth, valid, mask, K, pose, bounds)
        if not support['accepted']:
            return dict(error='insufficient observed depth support', depth_support=support), support
    if jump_prior is not None:
        step = np.linalg.inv(jump_prior) @ pose
        if (np.linalg.norm(step[:3, 3]) > 0.15 or
                Rotation.from_matrix(step[:3, :3]).magnitude() > np.deg2rad(40)):
            return {"error": "implausible pose jump"}, support
    return None, support


def stereo_frames(left, right, anchor=None, max_frames=0):
    """Visit every source frame once, anchored forwards then backwards if requested."""
    prefix = []
    index = 0
    while not max_frames or index < max_frames:
        ok_l, raw_l = left.read()
        ok_r, raw_r = right.read()
        if ok_l != ok_r:
            raise ValueError("stereo frame-count mismatch")
        if not ok_l:
            break
        if anchor is not None and index <= anchor:
            packed = []
            for raw in (raw_l, raw_r):
                if np.array_equal(raw[:, :, 0], raw[:, :, 1]) and np.array_equal(raw[:, :, 0], raw[:, :, 2]):
                    packed.append(raw[:, :, 0].copy())
                else:
                    packed.append(raw.copy())
            prefix.append(packed)
        if anchor is None or index >= anchor:
            yield index, raw_l, raw_r, index == anchor
        index += 1
    if anchor is not None:
        if anchor < 0 or anchor >= index:
            raise ValueError("Reconstruction anchor is outside the decoded frame axis")
        for frame in range(anchor-1, -1, -1):
            raw_l, raw_r = [cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if raw.ndim == 2 else raw for raw in prefix[frame]]
            yield frame, raw_l, raw_r, frame == anchor-1


def run(episode: int, name: str, data: Path, masks: Path, out: Path,
        register_iterations=5, track_iterations=2, max_frames=0,
        reinit_every=0, bootstrap_manifest=None, camera_root=None,
        foreground_depth_only=False, foreground_translation_prior=False,
        foreground_translation_fallback=False):
    target = (out / f"e{episode:03d}" / "ego_object_poses" / f"e{episode:03d}" /
              name / "object_to_cam" / "ego_cam_c")
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"object output must be fresh: {target}")
    # The FoundationPose installation and weights are supplied through the VM.
    import torch
    torch.backends.cudnn.enabled = False
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

    rig = load_rig()
    mesh_path = mesh_for(data, name)
    mesh_sha256 = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    anchor = None
    anchor_pose = None
    calibrated_anchor = False
    calibrated_anchor_sha256 = None
    if foreground_translation_prior and foreground_translation_fallback:
        raise ValueError('Choose always-on foreground initialization or rejection fallback separately')
    if (foreground_translation_prior or foreground_translation_fallback) and bootstrap_manifest is None:
        raise ValueError('Foreground translation prior requires an own reconstructed body anchor')
    if bootstrap_manifest is not None:
        proof = json.loads(bootstrap_manifest.read_text())
        if (proof.get('provided_meshes_read') is not False or proof.get('reference_pose_labels_read') is not False
                or proof['episode'] != episode or proof['object'] != name
                or proof['mesh_sha256'] != mesh_sha256 or proof.get('body_origin_frozen') is not True
                or reinit_every != 0):
            raise ValueError('Bootstrap tracking needs the exact reconstructed body and no periodic re-registration')
        anchor = int(proof['seed_frame'])
        anchor_pose = np.asarray(proof['seed_camera_from_object'], float)
        if (anchor_pose.shape != (4, 4) or not np.isfinite(anchor_pose).all()
                or not np.allclose(anchor_pose[3], [0, 0, 0, 1])
                or not np.allclose(anchor_pose[:3, :3].T @ anchor_pose[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(anchor_pose[:3, :3]), 1, atol=1e-5)):
            raise ValueError('Invalid measured reconstruction anchor')
        from calibrated_anchor import calibrated_seed
        calibrated_anchor, calibrated_anchor_sha256 = calibrated_seed(bootstrap_manifest, proof)
    # FoundationPose's renderer expects SimpleMaterial.image, while the public
    # GLBs carry PBRMaterial.baseColorTexture.
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        material = mesh.visual.material
        image = getattr(material, "baseColorTexture", None)
        if image is not None:
            mesh.visual.material = trimesh.visual.material.SimpleMaterial(image=image)
        else:
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh, vertex_colors=np.tile([128, 128, 128, 255], (len(mesh.vertices), 1)))
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    tracker = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                             mesh=mesh, scorer=scorer, refiner=refiner,
                             glctx=dr.RasterizeCudaContext(), debug=0)
    base = data / "videos/chunk-000"
    left = cv2.VideoCapture(str(base / f"observation.images.{rig.left}" /
                                f"episode_{episode:06d}.mp4"))
    right = cv2.VideoCapture(str(base / f"observation.images.{rig.right}" /
                                 f"episode_{episode:06d}.mp4"))
    if not left.isOpened() or not right.isOpened():
        raise FileNotFoundError(f"stereo episode {episode}")
    camera_poses = None
    camera_sha256 = None
    if camera_root is not None:
        if bootstrap_manifest is None:
            raise ValueError('Independent camera transport is an explicit reconstructed-body experiment')
        from camera_prior import load_sequence
        camera_poses, camera_sha256 = load_sequence(camera_root, episode,
            int(left.get(cv2.CAP_PROP_FRAME_COUNT)), rig.left)
    K = np.asarray(rig.P1[:3, :3], dtype=np.float64)
    directory = (out / f"e{episode:03d}" / "ego_object_poses" / f"e{episode:03d}" /
                 name / "object_to_cam" / rig.left)
    directory.mkdir(parents=True)
    last_frame = None
    last_pose = None
    last_registration = None
    last_pose_frame = None
    last_surface = None
    observed, missing, rejected = [], [], []
    fallback_attempted, fallback_accepted = [], []
    frames_seen = 0
    for frame, raw_l, raw_r, reset_anchor in stereo_frames(left, right, anchor, max_frames):
        frames_seen = max(frames_seen, frame+1)
        if reset_anchor:
            last_frame, last_pose = anchor, anchor_pose.copy()
            last_pose_frame = anchor
            last_surface = None
        rect_l, _ = rig.rectify_pair(raw_l, raw_r)
        rgb = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)
        depth, valid = stereo_depth(rig, raw_l, raw_r)
        mask_path = masks / name / f"{frame:06d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
        if mask is None or mask.shape != depth.shape or ((mask > 127) & valid).sum() < 100:
            missing.append(frame)
            last_frame = None
            continue
        refinement_depth = foreground_depth(depth, valid, mask) if foreground_depth_only else depth
        surface = None
        if foreground_translation_prior or foreground_translation_fallback:
            from foreground_prior import measured_surface
            surface = measured_surface(depth, valid, mask, K)
        try:
            prior_pose = last_pose
            if camera_poses is not None:
                from camera_prior import transport
                prior_pose = transport(last_pose, camera_poses[last_pose_frame], camera_poses[frame])
            acceptance_prior_pose = prior_pose
            surface_shift = 0.0
            if foreground_translation_prior and surface is not None and last_surface is not None:
                from foreground_prior import translated_prior
                camera_delta = (camera_poses[frame] @ np.linalg.inv(camera_poses[last_pose_frame])) if camera_poses is not None else None
                prior_pose = translated_prior(prior_pose, last_surface, surface, camera_delta)
                surface_shift = float(np.linalg.norm(prior_pose[:3, 3] - acceptance_prior_pose[:3, 3]))
            periodic = (reinit_every > 0 and last_registration is not None and
                        frame - last_registration >= reinit_every)
            if anchor is not None:
                tracker.pose_last = torch.as_tensor(centered_tracker_pose(prior_pose, tracker.model_center), device='cuda', dtype=torch.float)
                if frame == anchor:
                    if calibrated_anchor:
                        pose = anchor_pose.copy()
                        method = 'calibrated_measured_surface_anchor'
                    elif proof.get('stage') == 'sam3d_metric_depth':
                        pose = tracker.track_one(rgb=rgb, depth=refinement_depth, K=K,
                                                 iteration=max(register_iterations, track_iterations))
                        method = 'refine_sam3d_anchor'
                    else:
                        pose = anchor_pose.copy()
                        method = 'stereo_seed_anchor'
                else:
                    pose = tracker.track_one(rgb=rgb, depth=refinement_depth, K=K, iteration=track_iterations)
                    method = 'track_from_reconstructed_seed' if last_frame is not None else 'reacquire_from_seed_prior'
            elif last_frame is None or periodic:
                pose = tracker.register(K=K, rgb=rgb, depth=refinement_depth, ob_mask=mask > 127,
                                        iteration=register_iterations)
                method = "register_periodic" if periodic and last_frame is not None else "register"
            else:
                pose = tracker.track_one(rgb=rgb, depth=refinement_depth, K=K,
                                         iteration=track_iterations)
                method = "track"
        except Exception as error:  # keep the batch moving; report each failure
            traceback.print_exc()
            rejected.append({"frame": frame, "error": f"{type(error).__name__}: {error}"})
            last_frame = None
            continue
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        jump_prior = (acceptance_prior_pose if camera_poses is not None else last_pose) if last_pose is not None and last_frame is not None else None
        rejection, support = pose_rejection(pose, depth, valid, mask, K, mesh.bounds, anchor is not None, jump_prior)
        initial_rejection = rejection
        if rejection is not None and foreground_translation_fallback and surface is not None and last_surface is not None:
            # Successful normal tracks never enter this branch. Retry once, then
            # apply exactly the original gate reference and measured data again.
            from foreground_prior import translated_prior
            fallback_attempted.append(frame)
            camera_delta = (camera_poses[frame] @ np.linalg.inv(camera_poses[last_pose_frame])) if camera_poses is not None else None
            fallback_pose = translated_prior(acceptance_prior_pose, last_surface, surface, camera_delta)
            surface_shift = float(np.linalg.norm(fallback_pose[:3, 3] - acceptance_prior_pose[:3, 3]))
            tracker.pose_last = torch.as_tensor(centered_tracker_pose(fallback_pose, tracker.model_center), device='cuda', dtype=torch.float)
            try:
                pose = np.asarray(tracker.track_one(rgb=rgb, depth=refinement_depth, K=K, iteration=track_iterations), dtype=np.float64).reshape(4, 4)
                rejection, support = pose_rejection(pose, depth, valid, mask, K, mesh.bounds, anchor is not None, jump_prior)
                if rejection is None:
                    fallback_accepted.append(frame)
                    method = 'recover_from_measured_foreground'
            except Exception as error:
                traceback.print_exc()
                rejection = dict(error=f'{type(error).__name__}: {error}')
        if rejection is not None:
            rejected.append(dict(frame=frame, **rejection))
            last_frame = None
            continue
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        record = {"episode": episode, "frame_index": frame, "object": name,
                  "mesh_sha256": mesh_sha256,
                  "translation": pose[:3, 3].tolist(),
                  "rotation": quat[[3, 0, 1, 2]].tolist(),
                  "scale": 1.0, "status": "observed", "gt_use": "none",
                  "coordinate_frame": "rectified_camera", "camera_frame": rig.left,
                  "method": method if method in {'stereo_seed_anchor', 'calibrated_measured_surface_anchor'} else f"foundationpose_{method}",
                  "mask_pixels": int((mask > 127).sum()),
                  "depth_pixels": int(((mask > 127) & valid).sum())}
        record['foreground_depth_only'] = foreground_depth_only
        record['foreground_translation_prior'] = foreground_translation_prior
        record['foreground_translation_fallback'] = foreground_translation_fallback
        if initial_rejection is not None:
            record['normal_attempt_rejection'] = initial_rejection
        record['measured_surface_initialization_shift_m'] = surface_shift
        if support is not None:
            record['depth_support'] = support
        if camera_poses is not None:
            record.update(independent_camera_prior=True,camera_prior_frame=last_pose_frame,
                          camera_trajectory_sha256=camera_sha256)
        (directory / f"{frame:06d}.json").write_text(json.dumps(record))
        observed.append(frame)
        last_frame, last_pose = frame, pose
        last_pose_frame = frame
        last_surface = surface
        if method.startswith("register"):
            last_registration = frame
        if frame % 25 == 0:
            print(f"  FP {name} frame {frame}: {method}", flush=True)
    left.release()
    right.release()
    report = {"episode": episode, "object": name, "frames": frames_seen,
              "mesh_sha256": mesh_sha256,
              "observed": len(observed), "missing_mask_or_depth": missing,
              "rejected": rejected, "gt_use": "none",
              "reinit_every": reinit_every, "register_iterations": register_iterations,
              "track_iterations": track_iterations,
              "foreground_depth_only": foreground_depth_only,
              "foreground_translation_prior": foreground_translation_prior,
              "foreground_translation_fallback": foreground_translation_fallback,
              "fallback_attempted_frames": fallback_attempted,
              "fallback_accepted_frames": fallback_accepted,
              "acceptance_prior_excludes_foreground_translation": True,
              "acceptance_uses_original_measured_depth": True}
    if bootstrap_manifest is not None:
        report.update(bootstrap_manifest_sha256=hashlib.sha256(bootstrap_manifest.read_bytes()).hexdigest(),
                      seed_frame=anchor, initialization='bidirectional tracking from measured reconstruction body gauge',
                      no_arbitrary_bootstrap_rotation_registration=True)
        report.update(calibrated_seed_preserved=calibrated_anchor,
                      calibrated_seed_report_sha256=calibrated_anchor_sha256,
                      calibrated_seed_still_uses_original_acceptance_gates=True)
    report.update(independent_camera_prior=camera_poses is not None,camera_trajectory_sha256=camera_sha256)
    (out / f"e{episode:03d}" / f"{name}_fp_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get(
        "V2D_PUBLIC", "/home/ubuntu/v2d_track3/data/hf/track_3/public")))
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--register-iterations", type=int, default=5)
    parser.add_argument("--track-iterations", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--reinit-every", type=int, default=0,
                        help="re-register from the current mask every N frames; 0 disables")
    parser.add_argument('--bootstrap-manifest',type=Path,
                        help='Own reconstructed mesh anchor; preserve its measured seed-frame body gauge')
    parser.add_argument('--camera-root',type=Path,
                        help='Explicit experiment: per-frame independent rectified camera JSON folder')
    parser.add_argument('--foreground-depth-only', action='store_true',
                        help='Experiment: refine using measured object-mask depth only; acceptance gates unchanged')
    parser.add_argument('--foreground-translation-prior', action='store_true',
                        help='Experiment: initialize displacement from measured foreground; original pose-jump/depth gates unchanged')
    parser.add_argument('--foreground-translation-fallback', action='store_true',
                        help='Experiment: retry rejected normal tracks once with measured displacement; identical acceptance gates')
    args = parser.parse_args()
    print(run(args.episode, args.object, args.data, args.masks, args.out,
              args.register_iterations, args.track_iterations, args.max_frames,
              args.reinit_every, args.bootstrap_manifest, args.camera_root,
              args.foreground_depth_only, args.foreground_translation_prior,
              args.foreground_translation_fallback), flush=True)


if __name__ == "__main__":
    main()
