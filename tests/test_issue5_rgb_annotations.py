"""External reviewed prompts cannot change frame identity or discard failures."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/track3'))
from rgb_annotations import load_rgb_annotations, verify_review_image


def annotation(tmp_path):
    image = np.zeros((80, 128, 3), np.uint8)
    baseline = {'objects': ['white_pot', 'white_pot_lid'], 'calibration_sha256': 'calibration',
                'frames': [{'episode': 0, 'frame_index': 53}, {'episode': 2, 'frame_index': 49}]}
    prompt = [[10, 10, 80, 80], [[25, 25]], [[100, 50]]]
    record = {'episode': 0, 'gt_use': 'none', 'camera_frame': 'ego_cam_c',
              'objects': baseline['objects'], 'baseline_sha256': 'baseline',
              'calibration_sha256': 'calibration', 'frames': {'53': [prompt, prompt]},
              'review_images': [{'frame_index': 53, 'width': 128, 'height': 80,
                                 'rectified_bgr_sha256': hashlib.sha256(image.tobytes()).hexdigest()}]}
    path = tmp_path / 'annotations.json'
    path.write_text(json.dumps(record))
    return baseline, record, path


def test_reviewed_prompt_frame_identity_and_image_bytes_are_enforced(tmp_path):
    baseline, record, path = annotation(tmp_path)
    episode, prompts, images = load_rgb_annotations(path, baseline, 'baseline')
    assert episode == 0 and set(prompts) == set(images) == {53}
    record['episode'] = 2
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='every frozen frame'):
        load_rgb_annotations(path, baseline, 'baseline')
    record['episode'] = 0
    verify_review_image(np.zeros((80,128,3),np.uint8),images[53])
    with pytest.raises(ValueError, match='reviewed RGB differs'):
        verify_review_image(np.ones((80,128,3),np.uint8),images[53])


def test_missing_prompts_and_outside_points_are_rejected(tmp_path):
    baseline, record, path = annotation(tmp_path)
    baseline['frames'].append({'episode': 0, 'frame_index': 106})
    with pytest.raises(ValueError, match='every frozen frame'):
        load_rgb_annotations(path, baseline, 'baseline')
    baseline['frames'].pop()
    record['frames']['53'][0][1] = [[128, 20]]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='point outside'):
        load_rgb_annotations(path, baseline, 'baseline')
