"""Validate RGB-only observation-control prompts against frozen frame identity."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_rgb_annotations(path, baseline, baseline_sha256):
    path = Path(path)
    record = json.loads(path.read_text())
    if (record.get('gt_use') != 'none' or record.get('camera_frame') != 'ego_cam_c'
            or record.get('objects') != baseline['objects']
            or record.get('baseline_sha256') != baseline_sha256
            or record.get('calibration_sha256') != baseline['calibration_sha256']):
        raise ValueError('annotation identity/provenance differs from frozen observations')
    episode = record['episode']
    expected = {r['frame_index'] for r in baseline['frames'] if r['episode'] == episode}
    prompts = {int(k): v for k, v in record['frames'].items()}
    if not expected or set(prompts) != expected:
        raise ValueError('annotations must cover every frozen frame of the selected episode')
    reviewed = {}
    for item in record['review_images']:
        frame = item['frame_index']
        if frame in reviewed or frame not in expected:
            raise ValueError('duplicate or unexpected reviewed frame')
        h, w = item['height'], item['width']
        pixel_hash = item['rectified_bgr_sha256']
        if (not isinstance(h, int) or not isinstance(w, int) or h <= 0 or w <= 0
                or len(pixel_hash) != 64 or any(c not in '0123456789abcdef' for c in pixel_hash)):
            raise ValueError('reviewed image dimensions and pixel SHA-256 required')
        if len(prompts[frame]) != len(baseline['objects']):
            raise ValueError('one prompt per object required')
        for box, positive, negative in prompts[frame]:
            box = np.asarray(box, dtype=float)
            if (box.shape != (4,) or not np.isfinite(box).all()
                    or not 0 <= box[0] < box[2] <= w or not 0 <= box[1] < box[3] <= h):
                raise ValueError('prompt box outside reviewed image')
            if not positive:
                raise ValueError('foreground point required')
            points = np.asarray(positive + negative, dtype=float)
            if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                    or (points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any()):
                raise ValueError('prompt point outside reviewed image')
        reviewed[frame] = item
    if set(reviewed) != expected:
        raise ValueError('reviewed RGB images must cover every frozen frame')
    return episode, prompts, reviewed


def verify_review_image(image, record):
    """Bind prompts to decoded rectified BGR pixels, independently of PNG encoding."""
    if (image.shape != (record['height'], record['width'], 3)
            or image.dtype != np.uint8
            or hashlib.sha256(image.tobytes()).hexdigest() != record['rectified_bgr_sha256']):
        raise ValueError('reviewed RGB differs from current rectified source frame')
