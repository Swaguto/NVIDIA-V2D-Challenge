"""Run pinned ViPE SLAM on calibrated frames with measured stereo depth."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from vipe.slam.system import SLAMSystem
from vipe.streams.base import FrameAttribute, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, required=True)
parser.add_argument('--input-root', type=Path, default=Path('/workspace/issue6-full-inputs'))
parser.add_argument('--hand-root', type=Path, default=Path('/workspace/temporal-hands'))
parser.add_argument('--runtime-root', type=Path, default=Path('/workspace'))
parser.add_argument('--output', type=Path)
args = parser.parse_args()
root = args.input_root / f'e{args.episode:03d}'
manifest = json.loads((root / 'manifest.json').read_text())
camera = manifest['color_camera']
detections = json.loads((args.hand_root / f'e{args.episode:03d}/cleaned_detections.json').read_text())


class StereoStream(VideoStream):
    def __len__(self):
        return len(manifest['frames'])

    def frame_size(self):
        return (camera['height'], camera['width'])

    def name(self):
        return f'e{args.episode:03d}'

    def fps(self):
        return 1 / np.median(np.diff([f['timestamp_s'] for f in manifest['frames']]))

    def attributes(self):
        return {FrameAttribute.INTRINSICS, FrameAttribute.METRIC_DEPTH,
                FrameAttribute.CAMERA_TYPE, FrameAttribute.MASK}

    def __iter__(self):
        for number, frame in enumerate(manifest['frames']):
            rgb = cv2.cvtColor(cv2.imread(str(root / frame['color'])), cv2.COLOR_BGR2RGB)
            depth = np.load(root / frame['depth_path'], allow_pickle=False)
            depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
            static = np.ones(depth.shape, bool)
            for state in frame['objects'].values():
                if state.get('mask'):
                    mask = cv2.imread(state['mask'], cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        assert mask.shape == depth.shape
                        static &= mask == 0
            for side in ('left', 'right'):
                box = detections[number].get(f'{side}_bbox')
                if box is not None:
                    x0, y0, x1, y1 = np.asarray(box).round().astype(int)
                    static[max(0, y0):min(camera['height'], y1+1),
                           max(0, x0):min(camera['width'], x1+1)] = False
            yield VideoFrame(raw_frame_idx=number,
                             rgb=torch.as_tensor(rgb, device='cuda').float() / 255,
                             intrinsics=torch.tensor([camera[k] for k in ('fx', 'fy', 'cx', 'cy')], device='cuda'),
                             metric_depth=torch.as_tensor(depth, device='cuda'),
                             mask=torch.as_tensor(static, device='cuda'),
                             camera_type=CameraType.PINHOLE)


config = OmegaConf.load(args.runtime_root / 'ViPE/configs/slam/default.yaml')
config.optimize_intrinsics = False
config.keyframe_depth = None
config.visualize = False
started = time.time()
result = SLAMSystem(torch.device('cuda'), config).run([StereoStream()], camera_type=CameraType.PINHOLE)
c2w = result.get_view_trajectory(0).matrix().cpu().numpy()
assert c2w.shape == (len(manifest['frames']), 4, 4) and np.isfinite(c2w).all()
out = args.output or args.hand_root / f'e{args.episode:03d}/cameras'
out.mkdir(parents=True, exist_ok=True)
intrinsics = np.tile([camera[k] for k in ('fx', 'fy', 'cx', 'cy')], (len(c2w), 1)).astype(np.float32)
np.savez_compressed(out / 'cameras.npz', height=camera['height'], width=camera['width'],
                    focal=camera['fx'], intrins=intrinsics, w2c=np.linalg.inv(c2w).astype(np.float32))
np.savez_compressed(out / 'vipe_c2w.npz', data=c2w, inds=np.arange(len(c2w)))
receipt = {'episode': args.episode, 'frames': len(c2w), 'elapsed_s': time.time()-started,
           'method': 'ViPE SLAM with known rectified intrinsics and its measured-depth interface',
           'depth_source': 'calibrated left/right stereo disparity', 'gt_object_poses_used': False,
           'dynamic_masks': 'observed CAD masks plus enhanced-tracker hand boxes',
           'vipe_revision': 'fdc79359ec6d063c86b9d203ca3d7054a9636823',
           'torch_version': torch.__version__, 'cuda_version': torch.version.cuda}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
