import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from select_reconstruction_frame import mask_features, rank


def test_unknown_other_foreground_is_excluded_without_losing_known_views(
    tmp_path, monkeypatch
):
    import cv2
    import json
    from types import SimpleNamespace
    from select_reconstruction_frame import select

    masks = tmp_path / "masks"
    for name in ["target", "other"]:
        (masks / name).mkdir(parents=True)
    (masks / "report.json").write_text(
        json.dumps(dict(episode=1, objects=["target", "other"]))
    )
    target = np.full((20, 20), 255, np.uint8)
    for frame in [0, 1]:
        cv2.imwrite(str(masks / "target" / f"{frame:06d}.png"), target)
    # Frame 0 is otherwise the earliest clean candidate but its other-object
    # observation is missing. Frame 1 has explicitly observed empty foreground.
    cv2.imwrite(str(masks / "other/000001.png"), np.zeros_like(target))

    class Capture:
        def set(self, *_):
            pass

        def read(self):
            return True, np.zeros((20, 20, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: Capture())
    monkeypatch.setitem(
        sys.modules,
        "stereo_depth",
        SimpleNamespace(
            load_rig=lambda *_: SimpleNamespace(left="left", right="right"),
            stereo_depth=lambda *_: (np.ones((20, 20)), np.ones((20, 20), bool)),
        ),
    )
    result = select(tmp_path, masks, None, 1, "target", tmp_path / "selection.json")
    assert result["selected"]["frame"] == 1
    assert result["excluded_unknown_other_foreground_frames"] == [0]
    assert set(result["selected_foreground_sha256"]) == {"target", "other"}


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
