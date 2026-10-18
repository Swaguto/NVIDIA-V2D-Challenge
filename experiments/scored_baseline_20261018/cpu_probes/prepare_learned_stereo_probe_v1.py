"""Freeze seven exact rectified pairs for a bounded, private depth diagnostic."""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0, str(R/'reconstruction/modules/v2d_stereo_depth/lib'))
from stereo_depth import load_rig, stereo_depth

out = S/'learned-stereo-inputs-v1'
out.mkdir()
root = S/'scored-cohort-14-results/reconstruction-e037'
rig = load_rig(str(S/'dataset/track_3/public/meta/camera_calibration.json'))
caps = [cv2.VideoCapture(str(S/f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{cam}/episode_000037.mp4')) for cam in [rig.left, rig.right]]
with np.load(root/'bundles/e037/episode.npz', allow_pickle=False) as z:
    camera = z['camera_c2w'].copy()
rows = []
for frame in [0, 25, 50, 75, 100, 125, 150]:
    raw = []
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        assert ok
        raw.append(image)
    for side, image in zip(['left', 'right'], rig.rectify_pair(*raw)):
        assert cv2.imwrite(str(out/f'{frame:03d}-{side}.png'), image)
    depth, valid = stereo_depth(rig, *raw)
    mask = cv2.imread(str(root/f'source/e037/masks/wooden_spoon/{frame:06d}.png'), 0)
    assert cv2.imwrite(str(out/f'{frame:03d}-mask.png'), mask)
    np.savez_compressed(out/f'{frame:03d}-control.npz', depth_m=depth, valid=valid, camera_c2w=camera[frame])
    rows.append(frame)
for cap in caps:
    cap.release()
meta = dict(episode=37, frames=rows, rectified_intrinsics=rig.P1[:3, :3].tolist(),
            baseline_m=rig.baseline_m, automatic_masks=True, reference_labels_read=False,
            source_inputs_modified=False, pixels_filled=False, model_depths_are_estimates=True,
            source_code_commit='ef21fe0c6c9db0154cdbfaed96cc4b44ac5e67dd',
            files={str(p.relative_to(out)):dict(size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.iterdir())})
(out/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
print(json.dumps(dict(frames=rows, files=len(meta['files']), bytes=sum(f['size'] for f in meta['files'].values()))))
