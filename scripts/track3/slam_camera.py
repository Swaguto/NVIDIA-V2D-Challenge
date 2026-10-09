#!/usr/bin/env python3
"""Run metric stereo-depth-assisted DROID-SLAM on staged ego_cam_c inputs.

Requires the upstream DROID-SLAM checkout and its CUDA extensions on PYTHONPATH.
Depth is a sensor prior in local and global bundle adjustment. No reference
camera/object trajectory is read. Outputs declare camera_to_world explicitly.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(inputs: Path, weights: Path, out: Path, droid_repo: Path,
        image_size: tuple[int, int] = (240, 384), max_depth_m: float = 3.0,
        filter_thresh: float = 0.5, keyframe_thresh: float = 1.0) -> dict:
    import cv2
    import torch
    from v2d.common.datatypes import CameraIntrinsics, DepthImage
    from v2d.droid_slam.lib._slam import Frame, run_droid_slam

    manifest = json.loads((inputs / 'inputs.json').read_text())
    frames = manifest['frames']
    if manifest['camera_frame'] != 'ego_cam_c' or not frames or frames != sorted(set(frames)):
        raise ValueError('Expected ordered unique ego_cam_c source frames')
    if any(d <= 0 or d % 8 for d in image_size) or not np.isfinite(max_depth_m) or max_depth_m <= 0:
        raise ValueError('Image dimensions must be positive multiples of 8; depth limit must be positive')
    if any(not np.isfinite(v) or v <= 0 for v in (filter_thresh, keyframe_thresh)):
        raise ValueError('Motion/keyframe thresholds must be finite and positive')
    if out.exists():
        raise ValueError(f'Preserve existing SLAM output: {out}')
    intrinsics = CameraIntrinsics.load(str(inputs / 'intrinsics.json'))
    source_paths = [inputs / 'inputs.json', inputs / 'intrinsics.json']
    source_paths += [inputs / folder / f'{frame:06d}.png' for frame in frames for folder in ('frames', 'depth')]
    input_hashes = {str(p): sha(p) for p in source_paths}
    source_files = list(droid_repo.glob('droid_slam/**/*.py')) + list(droid_repo.glob('src/*'))
    source_files += list((Path(__file__).resolve().parents[2] / 'reconstruction/modules/v2d_droid_slam/lib').glob('*.py'))
    source_files += [Path(__file__).resolve()]
    source_hashes = {str(p): sha(p) for p in source_files if p.is_file()}
    # The trajectory filler temporarily appends a batch of non-keyframes.
    settings = {'image_size': list(image_size), 'buffer_size': max(len(frames) + 32, 64),
                'max_prior_depth_m': max_depth_m, 'stride': 1, 'upsample': False,
                'filter_thresh': filter_thresh, 'keyframe_thresh': keyframe_thresh, 'warmup': 8}
    receipt = {'episode': manifest['episode'], 'method': 'stereo_depth_droid_slam',
               'sensor_mode': 'RGB-D using calibrated metric stereo depth', 'camera_frame': 'ego_cam_c',
               'direction': 'camera_to_world', 'translation_units': 'metres', 'gt_use': 'none',
               'settings': settings, 'input_sha256': input_hashes, 'source_sha256': source_hashes,
               'weights_sha256': sha(weights), 'torch': torch.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(), 'status': 'running',
               'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    out.mkdir(parents=True)
    (out / 'run-receipt.json').write_text(json.dumps(receipt, indent=2))

    def stream():
        for position, frame in enumerate(frames):
            if position % 25 == 0:
                print(f'SLAM stream frame {frame} ({position + 1}/{len(frames)})', flush=True)
            image = cv2.imread(str(inputs / 'frames' / f'{frame:06d}.png'))
            if image is None or image.shape[:2] != (intrinsics.height, intrinsics.width):
                raise ValueError(f'Invalid source image dimensions at {frame}')
            depth = DepthImage.load(str(inputs / 'depth' / f'{frame:06d}.png')).depth.astype(np.float32)
            if depth.shape != image.shape[:2]:
                raise ValueError(f'Depth/image grid mismatch at {frame}')
            depth[~np.isfinite(depth) | (depth <= 0) | (depth > max_depth_m)] = 0
            yield Frame(frame_idx=frame, image_bgr=image, intrinsics=intrinsics, prior_depth=depth)

    try:
        details = run_droid_slam(stream, str(weights), str(out / 'camera_c2w'),
                                 trajectory_path=str(out / 'trajectory.tum'),
                                 image_size=image_size, buffer_size=settings['buffer_size'],
                                 filter_thresh=filter_thresh, keyframe_thresh=keyframe_thresh)
        if details['frames'] != frames or details['metric_depth_prior_frames'] != len(frames):
            raise RuntimeError('SLAM did not process every source frame with metric stereo depth')
        if details['keyframes'] < 2:
            raise RuntimeError('SLAM did not establish a multi-view map')
        if not details['frontend_initialized']:
            raise RuntimeError('SLAM frontend never initialized; decrease the motion threshold for low-motion video')
        changed = [str(p) for p in source_paths if sha(p) != input_hashes[str(p)]]
        if changed:
            raise RuntimeError(f'SLAM source inputs changed: {changed}')
        receipt.update(details, status='completed', source_inputs_changed=changed,
                       peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        evidence_id = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
        for frame in frames:
            path = out / 'camera_c2w' / f'{frame:06d}.json'
            pose = json.loads(path.read_text())
            pose.update(method=receipt['method'], gt_use='none', status='estimated',
                        direction='camera_to_world', camera_frame='ego_cam_c',
                        translation_units='metres', slam_run_evidence_sha256=evidence_id)
            path.write_text(json.dumps(pose, indent=2, allow_nan=False))
        receipt['pose_sha256'] = {p.name: sha(p) for p in sorted((out / 'camera_c2w').glob('*.json'))}
    except Exception as exc:
        receipt.update(status='failed', error=repr(exc))
        raise
    finally:
        receipt['ended_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (out / 'run-receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'weights', 'out', 'droid-repo'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image-height', type=int, default=240)
    parser.add_argument('--image-width', type=int, default=384)
    parser.add_argument('--max-depth-m', type=float, default=3.0)
    parser.add_argument('--filter-thresh', type=float, default=0.5)
    parser.add_argument('--keyframe-thresh', type=float, default=1.0)
    args = parser.parse_args()
    receipt = run(args.inputs, args.weights, args.out, args.droid_repo,
                  (args.image_height, args.image_width), args.max_depth_m,
                  args.filter_thresh, args.keyframe_thresh)
    print(json.dumps({k: receipt[k] for k in ('status', 'keyframes', 'peak_gpu_allocated_bytes')}, indent=2))


if __name__ == '__main__':
    main()
