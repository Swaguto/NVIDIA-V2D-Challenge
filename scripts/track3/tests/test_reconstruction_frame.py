import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from select_reconstruction_frame import mask_features, rank


def test_isolated_object_is_preferred_to_a_larger_fused_foreground():
    empty = np.zeros((100, 100), bool)
    one = empty.copy()
    one[10:60, 10:40] = True
    stacked = one.copy()
    stacked[60:90, 10:40] = True
    other = empty.copy()
    other[60:90, 10:40] = True
    rows = [
        dict(frame=155, **mask_features(stacked, other, empty)),
        dict(frame=0, **mask_features(one, other, empty)),
    ]
    ranked, _ = rank(rows)
    assert ranked[0]["frame"] == 0


def test_hand_occlusion_and_tiny_foreground_do_not_win_frame_selection():
    empty = np.zeros((100, 100), bool)
    target = empty.copy()
    target[10:90, 10:90] = True
    hand = empty.copy()
    hand[50:90, 10:90] = True
    tiny = empty.copy()
    tiny[10:12, 10:12] = True
    rows = [
        dict(frame=0, **mask_features(target, empty, hand)),
        dict(frame=20, **mask_features(target, empty, empty)),
        dict(frame=1, **mask_features(tiny, empty, empty)),
    ]
    ranked, _ = rank(rows)
    assert ranked[0]["frame"] == 20 and not any(r["frame"] == 1 for r in ranked)
    with pytest.raises(ValueError):
        mask_features(target, np.zeros((2, 2)), empty)
